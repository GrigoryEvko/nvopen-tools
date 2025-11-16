// Function: sub_1CAC930
// Address: 0x1cac930
//
__int64 __fastcall sub_1CAC930(
        _QWORD *a1,
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
  double v12; // xmm4_8
  double v13; // xmm5_8
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rdx
  unsigned int v17; // r13d
  __int64 v18; // r12
  __int64 v19; // rdi
  __int64 v20; // rax
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax

  sub_1C9BBA0((__int64)a1, a2, a3, a4, a5, a6, a7, a8, a9, a10);
  v14 = *(_QWORD *)(a2 + 80);
  if ( v14 != a2 + 72 )
  {
    while ( 1 )
    {
      if ( !v14 )
        BUG();
      v15 = *(_QWORD *)(v14 + 24);
      if ( v15 != v14 + 16 )
        break;
LABEL_9:
      v14 = *(_QWORD *)(v14 + 8);
      if ( a2 + 72 == v14 )
        goto LABEL_10;
    }
    while ( 1 )
    {
      if ( !v15 )
        BUG();
      if ( *(_BYTE *)(v15 - 8) == 54 )
      {
        v16 = *(_QWORD *)(v15 - 24);
        if ( *(_BYTE *)(v16 + 8) == 15 && !(*(_DWORD *)(v16 + 8) >> 8) )
          break;
      }
      v15 = *(_QWORD *)(v15 + 8);
      if ( v14 + 16 == v15 )
        goto LABEL_9;
    }
    sub_1CAB590(a1, a2);
    if ( byte_4FBDE40 )
      sub_1C9D5C0(a1, a2);
  }
LABEL_10:
  if ( dword_4FBDD60 == 1 )
    v17 = sub_1CA2920(a1, a2, a3, a4, a5, a6, v12, v13, a9, a10);
  else
    v17 = sub_1CA9E90(a1, a2, a3, a4, a5, a6, v12, v13, a9, a10);
  v18 = a1[64];
  while ( v18 )
  {
    sub_1C96740(*(_QWORD *)(v18 + 24));
    v19 = v18;
    v18 = *(_QWORD *)(v18 + 16);
    j_j___libc_free_0(v19, 48);
  }
  a1[64] = 0;
  a1[65] = a1 + 63;
  a1[66] = a1 + 63;
  v20 = a1[3];
  a1[67] = 0;
  if ( v20 != a1[4] )
    a1[4] = v20;
  v21 = a1[6];
  if ( v21 != a1[7] )
    a1[7] = v21;
  v22 = a1[9];
  if ( v22 != a1[10] )
    a1[10] = v22;
  v23 = a1[12];
  if ( v23 != a1[13] )
    a1[13] = v23;
  return v17;
}
