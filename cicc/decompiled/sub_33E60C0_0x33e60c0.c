// Function: sub_33E60C0
// Address: 0x33e60c0
//
_QWORD *__fastcall sub_33E60C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, char a6, __int64 *a7, int a8)
{
  __int64 *v11; // r13
  __int64 v12; // rsi
  __int64 *v13; // r8
  _QWORD *v14; // rax
  _QWORD *v15; // r12
  __int64 v16; // rdx
  unsigned __int64 v17; // rax
  unsigned __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rdx
  __m128i v21; // xmm0
  __m128i *v22; // rax
  __int64 v24; // [rsp+8h] [rbp-58h] BYREF
  __m128i v25; // [rsp+10h] [rbp-50h] BYREF
  __int64 v26; // [rsp+20h] [rbp-40h]

  LODWORD(v26) = a5;
  v11 = *(__int64 **)(a1 + 720);
  v25.m128i_i32[0] = 0;
  v12 = *a7;
  v25.m128i_i64[1] = a4;
  v13 = v11;
  v24 = v12;
  if ( v12 )
  {
    sub_B96E90((__int64)&v24, v12, 1);
    v13 = *(__int64 **)(a1 + 720);
  }
  v14 = (_QWORD *)sub_A777F0(0x40u, v13);
  v15 = v14;
  if ( v14 )
  {
    *v14 = 1;
    v16 = *v11;
    v11[10] += 24;
    v17 = (v16 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v11[1] >= v17 + 24 && v16 )
      *v11 = v17 + 24;
    else
      v17 = sub_9D1E70((__int64)v11, 24, 24, 3);
    v15[1] = v17;
    v15[2] = 0;
    v18 = (*v11 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v11[1] >= v18 && *v11 )
      *v11 = v18;
    else
      v18 = sub_9D1E70((__int64)v11, 0, 0, 3);
    v19 = v24;
    v15[3] = v18;
    v15[4] = a2;
    v15[5] = a3;
    v15[6] = v19;
    if ( v19 )
      sub_B96E90((__int64)(v15 + 6), v19, 1);
    v20 = v26;
    *((_BYTE *)v15 + 60) = a6;
    *((_BYTE *)v15 + 61) = 0;
    v21 = _mm_loadu_si128(&v25);
    *((_DWORD *)v15 + 14) = a8;
    *((_WORD *)v15 + 31) = 0;
    v22 = (__m128i *)v15[1];
    v22[1].m128i_i64[0] = v20;
    *v22 = v21;
  }
  if ( v24 )
    sub_B91220((__int64)&v24, v24);
  return v15;
}
