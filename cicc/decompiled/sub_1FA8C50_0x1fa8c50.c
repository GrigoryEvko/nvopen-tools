// Function: sub_1FA8C50
// Address: 0x1fa8c50
//
__int64 *__fastcall sub_1FA8C50(__int64 a1, __int64 a2, double a3, double a4, __m128i a5)
{
  __int64 v6; // rax
  unsigned __int16 v7; // r9
  __int64 v8; // r12
  unsigned int v9; // r14d
  __int64 v10; // rcx
  __int64 v11; // rsi
  __int64 *v12; // rax
  unsigned __int8 *v13; // rax
  const void **v14; // r8
  __int64 v15; // rcx
  unsigned __int128 v16; // kr00_16
  unsigned int v17; // esi
  __int64 *result; // rax
  __int64 v19; // rdx
  __int64 v20; // r14
  const void *v21; // rsi
  __int64 v22; // rdx
  __int64 *v23; // r11
  unsigned __int64 v24; // r9
  unsigned int v25; // r14d
  __int64 v26; // rax
  __m128i v27; // xmm0
  __int128 *v28; // rcx
  __int64 v29; // rsi
  __int64 v30; // rdx
  const void *v31; // r12
  __int64 v32; // rdx
  __int64 v33; // r13
  int v34; // [rsp+Ch] [rbp-A4h]
  __int64 v35; // [rsp+10h] [rbp-A0h]
  unsigned int v36; // [rsp+10h] [rbp-A0h]
  __int64 v37; // [rsp+18h] [rbp-98h]
  __int64 v38; // [rsp+18h] [rbp-98h]
  unsigned __int16 v39; // [rsp+20h] [rbp-90h]
  __int128 *v40; // [rsp+20h] [rbp-90h]
  __int64 *v41; // [rsp+20h] [rbp-90h]
  __int64 v42; // [rsp+28h] [rbp-88h]
  unsigned __int128 v43; // [rsp+30h] [rbp-80h]
  const void **v44; // [rsp+40h] [rbp-70h]
  __int64 *v45; // [rsp+40h] [rbp-70h]
  __int64 v46; // [rsp+40h] [rbp-70h]
  _QWORD *v47; // [rsp+40h] [rbp-70h]
  __int64 *s1; // [rsp+48h] [rbp-68h]
  __int64 *s1c; // [rsp+48h] [rbp-68h]
  void *s1a; // [rsp+48h] [rbp-68h]
  const void **s1b; // [rsp+48h] [rbp-68h]
  __int64 v52; // [rsp+50h] [rbp-60h] BYREF
  int v53; // [rsp+58h] [rbp-58h]
  __int64 v54; // [rsp+60h] [rbp-50h] BYREF
  unsigned int v55; // [rsp+68h] [rbp-48h]
  __int64 v56; // [rsp+70h] [rbp-40h]
  int v57; // [rsp+78h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 32);
  v7 = *(_WORD *)(a2 + 80);
  v8 = *(_QWORD *)v6;
  v9 = *(_DWORD *)(v6 + 8);
  v10 = *(_QWORD *)(v6 + 40);
  LODWORD(v6) = *(_DWORD *)(v6 + 48);
  v11 = *(_QWORD *)(v8 + 72);
  v55 = v9;
  v57 = v6;
  v34 = v6;
  v12 = *(__int64 **)a1;
  v56 = v10;
  s1 = v12;
  v13 = (unsigned __int8 *)(*(_QWORD *)(v8 + 40) + 16LL * v9);
  v35 = v10;
  v54 = v8;
  v14 = (const void **)*((_QWORD *)v13 + 1);
  v15 = *v13;
  v52 = v11;
  v16 = __PAIR128__(2, &v54);
  if ( v11 )
  {
    *(_QWORD *)&v43 = &v54;
    v37 = v15;
    v39 = v7;
    *((_QWORD *)&v43 + 1) = 2;
    v44 = v14;
    sub_1623A60((__int64)&v52, v11, 2);
    v15 = v37;
    v7 = v39;
    v16 = v43;
    v14 = v44;
  }
  v17 = *(unsigned __int16 *)(a2 + 24);
  v53 = *(_DWORD *)(v8 + 64);
  result = sub_1D39800(s1, v17, (__int64)&v52, v15, v14, v7, a3, a4, a5, (__int64 *)v16, *((__int64 *)&v16 + 1));
  if ( v52 )
  {
    s1c = result;
    sub_161E7C0((__int64)&v52, v52);
    result = s1c;
  }
  if ( !result )
  {
    if ( *(_BYTE *)(a1 + 25)
      && *(_WORD *)(v8 + 24) == 110
      && *(_WORD *)(v35 + 24) == 110
      && sub_1D18C00(v8, 1, v9)
      && sub_1D18C00(v35, 1, v34)
      && *(_WORD *)(*(_QWORD *)(*(_QWORD *)(v8 + 32) + 40LL) + 24LL) == 48
      && *(_WORD *)(*(_QWORD *)(*(_QWORD *)(v35 + 32) + 40LL) + 24LL) == 48
      && (s1a = (void *)sub_1F80640(v8), v20 = v19, v21 = (const void *)sub_1F80640(v35), v20 == v22)
      && (!(4 * v20) || !memcmp(s1a, v21, 4 * v20)) )
    {
      v23 = *(__int64 **)a1;
      v24 = *(unsigned __int16 *)(a2 + 80);
      v25 = **(unsigned __int8 **)(a2 + 40);
      v26 = *(_QWORD *)(v8 + 32);
      s1b = *(const void ***)(*(_QWORD *)(a2 + 40) + 8LL);
      v27 = _mm_loadu_si128((const __m128i *)(v26 + 40));
      v28 = *(__int128 **)(v35 + 32);
      v52 = *(_QWORD *)(a2 + 72);
      if ( v52 )
      {
        v36 = v24;
        v38 = v26;
        v40 = v28;
        v45 = v23;
        sub_1F6CA20(&v52);
        v24 = v36;
        v26 = v38;
        v28 = v40;
        v23 = v45;
      }
      v29 = *(unsigned __int16 *)(a2 + 24);
      v53 = *(_DWORD *)(a2 + 64);
      v41 = sub_1D332F0(
              v23,
              v29,
              (__int64)&v52,
              v25,
              s1b,
              v24,
              *(double *)v27.m128i_i64,
              a4,
              a5,
              *(_QWORD *)v26,
              *(_QWORD *)(v26 + 8),
              *v28);
      v42 = v30;
      sub_17CD270(&v52);
      sub_1FA7E80(a1, *(_QWORD *)(a2 + 48));
      v46 = *(_QWORD *)a1;
      v31 = (const void *)sub_1F80640(v8);
      v33 = v32;
      v52 = *(_QWORD *)(a2 + 72);
      if ( v52 )
        sub_1F6CA20(&v52);
      v53 = *(_DWORD *)(a2 + 64);
      v47 = sub_1D41320(
              v46,
              v25,
              s1b,
              (__int64)&v52,
              (__int64)v41,
              v42,
              *(double *)v27.m128i_i64,
              a4,
              a5,
              v27.m128i_i64[0],
              v27.m128i_i64[1],
              v31,
              v33);
      sub_17CD270(&v52);
      return v47;
    }
    else
    {
      return 0;
    }
  }
  return result;
}
