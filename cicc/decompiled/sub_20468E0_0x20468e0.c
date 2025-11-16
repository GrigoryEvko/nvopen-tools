// Function: sub_20468E0
// Address: 0x20468e0
//
__int64 __fastcall sub_20468E0(
        __int64 *a1,
        __int64 a2,
        unsigned __int64 a3,
        __m128i a4,
        double a5,
        __m128i a6,
        __int64 a7,
        __int64 a8)
{
  __int128 v10; // rax
  __int64 *v11; // rax
  unsigned __int64 v12; // rdx
  unsigned __int64 v13; // r15
  __int64 v14; // r14
  __int64 v15; // rax
  unsigned int v16; // edx
  unsigned __int8 v17; // al
  __int128 v18; // rax
  __int64 *v19; // rax
  unsigned __int64 v20; // rdx
  unsigned __int64 v21; // r15
  __int64 v22; // r14
  __int128 v23; // rax
  __int128 v24; // rax

  *(_QWORD *)&v10 = sub_1D38BB0((__int64)a1, 2139095040, a8, 5, 0, 0, a4, a5, a6, 0);
  v11 = sub_1D332F0(a1, 118, a8, 5, 0, 0, *(double *)a4.m128i_i64, a5, a6, a2, a3, v10);
  v13 = v12;
  v14 = (__int64)v11;
  v15 = sub_1E0A0C0(a1[4]);
  v16 = 8 * sub_15A9520(v15, 0);
  if ( v16 == 32 )
  {
    v17 = 5;
  }
  else if ( v16 > 0x20 )
  {
    v17 = 6;
    if ( v16 != 64 )
    {
      v17 = 0;
      if ( v16 == 128 )
        v17 = 7;
    }
  }
  else
  {
    v17 = 3;
    if ( v16 != 8 )
      v17 = 4 * (v16 == 16);
  }
  *(_QWORD *)&v18 = sub_1D38BB0((__int64)a1, 23, a8, v17, 0, 0, a4, a5, a6, 0);
  v19 = sub_1D332F0(a1, 124, a8, 5, 0, 0, *(double *)a4.m128i_i64, a5, a6, v14, v13, v18);
  v21 = v20;
  v22 = (__int64)v19;
  *(_QWORD *)&v23 = sub_1D38BB0((__int64)a1, 127, a8, 5, 0, 0, a4, a5, a6, 0);
  *(_QWORD *)&v24 = sub_1D332F0(a1, 53, a8, 5, 0, 0, *(double *)a4.m128i_i64, a5, a6, v22, v21, v23);
  return sub_1D309E0(a1, 146, a8, 9, 0, 0, *(double *)a4.m128i_i64, a5, *(double *)a6.m128i_i64, v24);
}
