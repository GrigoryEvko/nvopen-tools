// Function: sub_26C3D10
// Address: 0x26c3d10
//
__int64 __fastcall sub_26C3D10(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rsi
  __int64 v4; // rdx
  __int64 v5; // rcx
  _QWORD *v6; // r13
  __int64 v7; // rax
  void (*v8)(); // rax
  void (__fastcall *v9)(_QWORD *); // rax
  __int64 v10; // rsi
  _QWORD v12[3]; // [rsp+8h] [rbp-18h] BYREF

  v3 = *(_QWORD *)(a2 + 1712);
  if ( v3 && (sub_30CC6B0(v12, v3, a3, 0), (v6 = (_QWORD *)v12[0]) != 0) )
  {
    if ( *(_BYTE *)(v12[0] + 56LL) )
    {
      sub_30CACB0(v12[0], v3, v4, v5);
      *(_DWORD *)(a1 + 8) = 0;
      *(_QWORD *)a1 = 0x80000000LL;
      *(_QWORD *)(a1 + 16) = "previously inlined";
      *(_BYTE *)(a1 + 56) = 0;
      *(_BYTE *)(a1 + 64) = 1;
    }
    else
    {
      v7 = *(_QWORD *)v12[0];
      *(_BYTE *)(v12[0] + 57LL) = 1;
      v8 = *(void (**)())(v7 + 40);
      if ( v8 != nullsub_1535 )
        ((void (__fastcall *)(_QWORD *))v8)(v6);
      *(_QWORD *)a1 = 0x7FFFFFFF;
      *(_DWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = "not previously inlined";
      *(_BYTE *)(a1 + 56) = 0;
      *(_BYTE *)(a1 + 64) = 1;
    }
    v9 = *(void (__fastcall **)(_QWORD *))(*v6 + 8LL);
    if ( v9 == sub_2610030 )
    {
      v10 = v6[4];
      *v6 = &unk_4A1F3E0;
      if ( v10 )
        sub_B91220((__int64)(v6 + 4), v10);
      j_j___libc_free_0((unsigned __int64)v6);
      return a1;
    }
    else
    {
      v9(v6);
      return a1;
    }
  }
  else
  {
    *(_QWORD *)(a1 + 64) = 0;
    *(_OWORD *)a1 = 0;
    *(_OWORD *)(a1 + 16) = 0;
    *(_OWORD *)(a1 + 32) = 0;
    *(_OWORD *)(a1 + 48) = 0;
    return a1;
  }
}
