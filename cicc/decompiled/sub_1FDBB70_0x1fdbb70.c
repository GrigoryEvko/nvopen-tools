// Function: sub_1FDBB70
// Address: 0x1fdbb70
//
__int64 __fastcall sub_1FDBB70(_QWORD *a1, __int64 a2, unsigned int a3)
{
  unsigned int v3; // r12d
  __int64 **v5; // rax
  unsigned int v6; // eax
  unsigned int v7; // ebx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v11; // rax
  __int64 *v12; // rax
  unsigned int v13; // r8d
  __int64 *v14; // rax
  bool v15; // al
  __int64 (*v16)(); // r10
  unsigned int v17; // edx
  __int64 v18; // [rsp+0h] [rbp-40h]
  __int64 v19; // [rsp+8h] [rbp-38h]
  unsigned int v20; // [rsp+8h] [rbp-38h]

  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v5 = *(__int64 ***)(a2 - 8);
  else
    v5 = (__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  LOBYTE(v6) = sub_1FD35E0(a1[12], **v5);
  v7 = v6;
  LOBYTE(v8) = sub_1FD35E0(a1[12], *(_QWORD *)a2);
  v9 = v8;
  LOBYTE(v3) = (unsigned __int8)v7 <= 1u || (_BYTE)v8 == 1 || (_BYTE)v8 == 0;
  if ( (_BYTE)v3 )
    return 0;
  v11 = a1[14];
  if ( !*(_QWORD *)(v11 + 8LL * (unsigned __int8)v9 + 120) || !*(_QWORD *)(v11 + 8LL * (unsigned __int8)v7 + 120) )
    return 0;
  v12 = (*(_BYTE *)(a2 + 23) & 0x40) != 0
      ? *(__int64 **)(a2 - 8)
      : (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v19 = v9;
  v13 = sub_1FD8F60(a1, *v12);
  if ( !v13 )
    return 0;
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    v14 = *(__int64 **)(a2 - 8);
  else
    v14 = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v18 = v19;
  v20 = v13;
  v15 = sub_1FD4DC0((__int64)a1, *v14);
  v16 = *(__int64 (**)())(*a1 + 64LL);
  if ( v16 != sub_1FD34C0 )
  {
    v17 = ((__int64 (__fastcall *)(_QWORD *, _QWORD, __int64, _QWORD, _QWORD, bool))v16)(a1, v7, v18, a3, v20, v15);
    if ( v17 )
    {
      v3 = 1;
      sub_1FD5CC0((__int64)a1, a2, v17, 1);
      return v3;
    }
    return 0;
  }
  return v3;
}
