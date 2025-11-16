// Function: sub_1B78E20
// Address: 0x1b78e20
//
void __fastcall sub_1B78E20(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v11; // rax
  __int64 *v12; // rbx
  __int64 *v13; // r13
  __int64 v14; // rax
  __int64 v15; // rcx
  unsigned __int64 v16; // rdx
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  int v21; // r9d
  double v22; // xmm4_8
  double v23; // xmm5_8
  __int64 *v24; // rbx
  __int64 *v25; // r13
  __int64 v26; // rsi
  __int64 v27; // r14
  __int64 j; // r15
  __int64 v29; // rsi
  __int64 i; // [rsp+8h] [rbp-38h]

  v11 = 3LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
  {
    v12 = *(__int64 **)(a2 - 8);
    v13 = &v12[v11];
  }
  else
  {
    v13 = (__int64 *)a2;
    v12 = (__int64 *)(a2 - v11 * 8);
  }
  for ( ; v13 != v12; v12 += 3 )
  {
    if ( *v12 )
    {
      v14 = sub_1B75C50(a1, *v12, *(double *)a3.m128_u64, a4, a5);
      if ( *v12 )
      {
        v15 = v12[1];
        v16 = v12[2] & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v16 = v15;
        if ( v15 )
          *(_QWORD *)(v15 + 16) = *(_QWORD *)(v15 + 16) & 3LL | v16;
      }
      *v12 = v14;
      if ( v14 )
      {
        v17 = *(_QWORD *)(v14 + 8);
        v12[1] = v17;
        if ( v17 )
          *(_QWORD *)(v17 + 16) = (unsigned __int64)(v12 + 1) | *(_QWORD *)(v17 + 16) & 3LL;
        v12[2] = (v14 + 8) | v12[2] & 3;
        *(_QWORD *)(v14 + 8) = v12;
      }
    }
  }
  sub_1B78850(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
  if ( *(_QWORD *)(a1 + 8) )
  {
    if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
    {
      sub_15E08E0(a2, a2);
      v24 = *(__int64 **)(a2 + 88);
      v25 = &v24[5 * *(_QWORD *)(a2 + 96)];
      if ( (*(_BYTE *)(a2 + 18) & 1) != 0 )
      {
        sub_15E08E0(a2, a2);
        v24 = *(__int64 **)(a2 + 88);
      }
    }
    else
    {
      v24 = *(__int64 **)(a2 + 88);
      v25 = &v24[5 * *(_QWORD *)(a2 + 96)];
    }
    for ( ;
          v25 != v24;
          *(v24 - 5) = (*(__int64 (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 8) + 24LL))(
                         *(_QWORD *)(a1 + 8),
                         v26) )
    {
      v26 = *v24;
      v24 += 5;
    }
  }
  v27 = *(_QWORD *)(a2 + 80);
  for ( i = a2 + 72; i != v27; v27 = *(_QWORD *)(v27 + 8) )
  {
    if ( !v27 )
      BUG();
    for ( j = *(_QWORD *)(v27 + 24); v27 + 16 != j; j = *(_QWORD *)(j + 8) )
    {
      v29 = j - 24;
      if ( !j )
        v29 = 0;
      sub_1B78950(a1, v29, a3, a4, a5, a6, v22, v23, a9, a10, v18, v19, v20, v21);
    }
  }
}
