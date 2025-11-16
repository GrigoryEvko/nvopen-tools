// Function: sub_374D270
// Address: 0x374d270
//
__int64 __fastcall sub_374D270(__int64 a1, int a2, unsigned int a3)
{
  __int64 v3; // rsi
  __int64 v5; // r12
  int v7; // r14d
  __int64 v8; // r15
  int v9; // r13d
  __int64 v10; // rbx
  unsigned __int64 v11; // rdi
  bool v12; // cc
  unsigned __int64 v13; // rdi
  __int64 v14; // [rsp+0h] [rbp-50h] BYREF
  int v15; // [rsp+8h] [rbp-48h]
  __int64 v16; // [rsp+10h] [rbp-40h] BYREF
  int v17; // [rsp+18h] [rbp-38h]

  v3 = a2 & 0x7FFFFFFF;
  if ( *(_DWORD *)(a1 + 1096) <= (unsigned int)v3 )
    return 0;
  v5 = *(_QWORD *)(a1 + 1088) + 40 * v3;
  if ( *(char *)(v5 + 3) >= 0 )
    return 0;
  if ( a3 > *(_DWORD *)(v5 + 16) )
  {
    *(_DWORD *)v5 = *(_DWORD *)v5 & 0x80000000 | 1;
    sub_C449B0((__int64)&v16, (const void **)(v5 + 24), a3);
    sub_C449B0((__int64)&v14, (const void **)(v5 + 8), a3);
    v7 = v15;
    v8 = v14;
    v9 = v17;
    v10 = v16;
    if ( *(_DWORD *)(v5 + 16) > 0x40u )
    {
      v11 = *(_QWORD *)(v5 + 8);
      if ( v11 )
        j_j___libc_free_0_0(v11);
    }
    v12 = *(_DWORD *)(v5 + 32) <= 0x40u;
    *(_QWORD *)(v5 + 8) = v8;
    *(_DWORD *)(v5 + 16) = v7;
    if ( !v12 )
    {
      v13 = *(_QWORD *)(v5 + 24);
      if ( v13 )
        j_j___libc_free_0_0(v13);
    }
    *(_QWORD *)(v5 + 24) = v10;
    *(_DWORD *)(v5 + 32) = v9;
  }
  return v5;
}
