// Function: sub_E5E4A0
// Address: 0xe5e4a0
//
__int64 __fastcall sub_E5E4A0(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r9
  unsigned int v6; // r12d
  __int64 v8; // rax
  __int64 v9; // rdi
  void (*v10)(); // rax
  __int64 v11; // [rsp+10h] [rbp-B0h] BYREF
  __int64 v12; // [rsp+18h] [rbp-A8h]
  _QWORD v13[2]; // [rsp+20h] [rbp-A0h] BYREF
  _BYTE v14[144]; // [rsp+30h] [rbp-90h] BYREF

  v6 = sub_E5E420(a1, a2);
  if ( (_BYTE)v6 )
  {
    v8 = *(_QWORD *)(a2 + 112);
    v13[0] = v14;
    v11 = v8;
    v12 = *(_QWORD *)(a2 + 120);
    v13[1] = 0x600000000LL;
    if ( *(_DWORD *)(a2 + 136) )
      sub_E5B8D0((__int64)v13, a2 + 128, v3, v4, (__int64)v13, v5);
    v9 = *(_QWORD *)(a1 + 8);
    v10 = *(void (**)())(*(_QWORD *)v9 + 144LL);
    if ( v10 != nullsub_322 )
      ((void (__fastcall *)(__int64, __int64 *, _QWORD))v10)(v9, &v11, *(_QWORD *)(a2 + 32));
    *(_QWORD *)(a2 + 112) = v11;
    *(_QWORD *)(a2 + 120) = v12;
    sub_E5B8D0(a2 + 128, (__int64)v13, v3, v4, (__int64)v13, (__int64)&v11);
    *(_DWORD *)(a2 + 80) = 0;
    *(_QWORD *)(a2 + 48) = 0;
    (*(void (__fastcall **)(_QWORD, __int64 *, __int64, __int64, _QWORD))(**(_QWORD **)(a1 + 16) + 24LL))(
      *(_QWORD *)(a1 + 16),
      &v11,
      a2 + 40,
      a2 + 72,
      *(_QWORD *)(a2 + 32));
    if ( (_BYTE *)v13[0] != v14 )
      _libc_free(v13[0], &v11);
  }
  return v6;
}
