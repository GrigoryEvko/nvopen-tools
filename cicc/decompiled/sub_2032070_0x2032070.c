// Function: sub_2032070
// Address: 0x2032070
//
__int64 __fastcall sub_2032070(__int64 *a1, unsigned __int64 a2, __m128i a3, double a4, __m128i a5)
{
  unsigned __int8 v6; // cl
  unsigned int v7; // ecx
  const __m128i *v8; // r9
  __int64 v9; // r14
  __int64 v10; // rsi
  __int64 *v11; // r11
  unsigned __int64 v12; // r8
  __int64 v13; // r9
  __int64 *v14; // rcx
  unsigned int v15; // edx
  unsigned __int64 v16; // r8
  __int128 v18; // [rsp-10h] [rbp-190h]
  unsigned __int64 v19; // [rsp+0h] [rbp-180h]
  __int64 v20; // [rsp+8h] [rbp-178h]
  __int64 *v21; // [rsp+10h] [rbp-170h]
  __int64 *v22; // [rsp+10h] [rbp-170h]
  unsigned int v23; // [rsp+28h] [rbp-158h]
  __int64 v24; // [rsp+30h] [rbp-150h] BYREF
  int v25; // [rsp+38h] [rbp-148h]
  _BYTE *v26; // [rsp+40h] [rbp-140h] BYREF
  __int64 v27; // [rsp+48h] [rbp-138h]
  _BYTE v28[304]; // [rsp+50h] [rbp-130h] BYREF

  v6 = *(_BYTE *)(a2 + 27);
  v26 = v28;
  v27 = 0x1000000000LL;
  v7 = (v6 >> 2) & 3;
  if ( v7 )
    v9 = (__int64)sub_202F090((__int64)a1, (__int64)&v26, a2, v7);
  else
    v9 = sub_2030B50(a1, (__int64)&v26, a2, a3);
  if ( (unsigned int)v27 == 1 )
  {
    v14 = *(__int64 **)v26;
    v16 = *((unsigned int *)v26 + 2);
  }
  else
  {
    v10 = *(_QWORD *)(a2 + 72);
    v11 = (__int64 *)a1[1];
    v12 = (unsigned __int64)v26;
    v13 = (unsigned int)v27;
    v24 = v10;
    if ( v10 )
    {
      v19 = (unsigned __int64)v26;
      v21 = v11;
      v20 = (unsigned int)v27;
      sub_1623A60((__int64)&v24, v10, 2);
      v12 = v19;
      v13 = v20;
      v11 = v21;
    }
    *((_QWORD *)&v18 + 1) = v13;
    *(_QWORD *)&v18 = v12;
    v25 = *(_DWORD *)(a2 + 64);
    v14 = sub_1D359D0(v11, 2, (__int64)&v24, 1, 0, 0, *(double *)a3.m128i_i64, a4, a5, v18);
    v23 = v15;
    v16 = v15;
    if ( v24 )
    {
      v22 = v14;
      sub_161E7C0((__int64)&v24, v24);
      v16 = v23;
      v14 = v22;
    }
  }
  sub_2013400((__int64)a1, a2, 1, (__int64)v14, (__m128i *)v16, v8);
  if ( v26 != v28 )
    _libc_free((unsigned __int64)v26);
  return v9;
}
