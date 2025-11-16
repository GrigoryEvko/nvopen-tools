// Function: sub_1781A00
// Address: 0x1781a00
//
__int64 __fastcall sub_1781A00(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v4; // rax
  unsigned __int64 v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rax
  char v9; // al
  __int64 v10; // rax
  __int64 v11; // rcx
  _BYTE *v12; // rdi
  char v13; // r13
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // r12
  __int16 *v17; // rbx
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rbx
  __int64 i; // r12
  double v22; // [rsp-80h] [rbp-80h]
  __int64 v23[4]; // [rsp-78h] [rbp-78h] BYREF
  __int64 v24; // [rsp-58h] [rbp-58h] BYREF
  void *v25; // [rsp-50h] [rbp-50h] BYREF
  __int64 v26; // [rsp-48h] [rbp-48h]

  v2 = *(_QWORD *)(a2 + 8);
  if ( !v2 )
    return 0;
  if ( *(_QWORD *)(v2 + 8) )
    return 0;
  if ( *(_BYTE *)(a2 + 16) != 78 )
    return 0;
  v4 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v4 + 16) )
    return 0;
  if ( *(_DWORD *)(v4 + 36) != *(_DWORD *)a1 )
    return 0;
  v6 = a2 & 0xFFFFFFFFFFFFFFF8LL;
  v7 = *(_QWORD *)(v6 + 24 * (*(unsigned int *)(a1 + 8) - (unsigned __int64)(*(_DWORD *)(v6 + 20) & 0xFFFFFFF)));
  v8 = *(_QWORD *)(v7 + 8);
  if ( !v8 || *(_QWORD *)(v8 + 8) )
    return 0;
  v9 = *(_BYTE *)(v7 + 16);
  if ( v9 == 40 )
  {
    v14 = *(_QWORD *)(v7 - 48);
    if ( v14 )
    {
      **(_QWORD **)(a1 + 16) = v14;
      if ( (unsigned __int8)sub_13D6AF0((double *)(a1 + 24), *(_QWORD *)(v7 - 24)) )
        return 1;
    }
  }
  else
  {
    if ( v9 != 5 )
      return 0;
    if ( *(_WORD *)(v7 + 18) != 16 )
      return 0;
    v10 = *(_QWORD *)(v7 - 24LL * (*(_DWORD *)(v7 + 20) & 0xFFFFFFF));
    if ( !v10 )
      return 0;
    **(_QWORD **)(a1 + 16) = v10;
    v11 = *(_DWORD *)(v7 + 20) & 0xFFFFFFF;
    v12 = *(_BYTE **)(v7 + 24 * (1 - v11));
    if ( v12[16] == 14 )
    {
      v13 = sub_17802A0((__int64)v12, *(double *)(a1 + 24));
    }
    else
    {
      if ( *(_BYTE *)(*(_QWORD *)v12 + 8LL) != 16 )
        return 0;
      v15 = sub_15A1020(v12, v6, v7, v11);
      v16 = v15;
      if ( !v15 || *(_BYTE *)(v15 + 16) != 14 )
        return 0;
      v22 = *(double *)(a1 + 24);
      v17 = (__int16 *)sub_1698280();
      sub_169D3F0((__int64)v23, v22);
      sub_169E320(&v25, v23, v17);
      sub_1698460((__int64)v23);
      sub_16A3360((__int64)&v24, *(__int16 **)(v16 + 32), 0, (bool *)v23);
      v13 = sub_1594120(v16, (__int64)&v24, v18, v19);
      if ( v25 == sub_16982C0() )
      {
        v20 = v26;
        if ( v26 )
        {
          for ( i = v26 + 32LL * *(_QWORD *)(v26 - 8); v20 != i; sub_127D120((_QWORD *)(i + 8)) )
            i -= 32;
          j_j_j___libc_free_0_0(v20 - 8);
        }
      }
      else
      {
        sub_1698460((__int64)&v25);
      }
    }
    if ( v13 )
      return 1;
  }
  return 0;
}
