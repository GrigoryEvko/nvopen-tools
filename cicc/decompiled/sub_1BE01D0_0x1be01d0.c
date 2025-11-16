// Function: sub_1BE01D0
// Address: 0x1be01d0
//
__int64 __fastcall sub_1BE01D0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        int a5,
        __m128i a6,
        __m128i a7,
        __m128i a8,
        __m128i a9,
        double a10,
        double a11,
        double a12,
        __m128 a13)
{
  _QWORD *v13; // r13
  __int64 v14; // rbx
  unsigned int v17; // r12d
  __int64 v18; // rsi
  unsigned __int8 v19; // al
  _QWORD *v20; // r14
  _QWORD *v21; // rbx
  __int64 v22; // rax

  v13 = *(_QWORD **)a2;
  v14 = *(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 == v14 )
  {
    v17 = 0;
    goto LABEL_15;
  }
  v17 = 0;
  do
  {
    while ( 1 )
    {
      v18 = *(_QWORD *)(v14 - 8);
      if ( !v18 )
        goto LABEL_4;
      v19 = *(_BYTE *)(v18 + 16);
      if ( v19 <= 0x17u )
        goto LABEL_4;
      if ( v19 != 87 )
        break;
      v17 |= sub_1BDC800(
               a1,
               (__int64 *)v18,
               a3,
               (__int64)a4,
               a6,
               *(double *)a7.m128i_i64,
               *(double *)a8.m128i_i64,
               *(double *)a9.m128i_i64,
               a10,
               a11,
               a12,
               a13);
LABEL_4:
      v14 -= 24;
      if ( v13 == (_QWORD *)v14 )
        goto LABEL_10;
    }
    if ( v19 != 84 )
    {
      if ( (unsigned __int8)(v19 - 75) <= 1u )
        v17 |= sub_1BE0150(a1, v18, a3, a4, a6, a7, a8, a9, a10, a11, a12, a13);
      goto LABEL_4;
    }
    v14 -= 24;
    v17 |= sub_1BDC990(
             a1,
             v18,
             a6,
             *(double *)a7.m128i_i64,
             *(double *)a8.m128i_i64,
             *(double *)a9.m128i_i64,
             a10,
             a11,
             a12,
             a13,
             a3,
             (__int64)a4,
             a5);
  }
  while ( v13 != (_QWORD *)v14 );
LABEL_10:
  v20 = *(_QWORD **)a2;
  v21 = (_QWORD *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8));
  while ( v20 != v21 )
  {
    while ( 1 )
    {
      v22 = *(v21 - 1);
      v21 -= 3;
      if ( v22 == -8 || v22 == 0 || v22 == -16 )
        break;
      sub_1649B30(v21);
      if ( v20 == v21 )
        goto LABEL_15;
    }
  }
LABEL_15:
  *(_DWORD *)(a2 + 8) = 0;
  return v17;
}
