// Function: sub_6F63B0
// Address: 0x6f63b0
//
void __fastcall sub_6F63B0(__m128i *a1, __int64 a2, __int64 *a3, _DWORD *a4)
{
  __int64 v7; // r12
  char v8; // al
  __int64 v9; // rdi
  __int64 v10; // r15
  __m128i v11; // xmm1
  __m128i v12; // xmm2
  __m128i v13; // xmm3
  __m128i v14; // xmm4
  __m128i v15; // xmm5
  __m128i v16; // xmm6
  __m128i v17; // xmm7
  __m128i v18; // xmm0
  __int8 v19; // al
  char *v20; // rax
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 i; // rax
  __int64 v24; // rcx
  char v25; // al
  __int64 v26; // rdi
  __int64 v27; // r8
  __int64 v28; // r9
  __m128i v29; // xmm2
  __m128i v30; // xmm3
  __m128i v31; // xmm4
  __m128i v32; // xmm5
  __m128i v33; // xmm6
  __m128i v34; // xmm7
  __m128i v35; // xmm1
  __m128i v36; // xmm2
  __m128i v37; // xmm3
  __m128i v38; // xmm4
  __m128i v39; // xmm5
  __m128i v40; // xmm6
  _QWORD *v41; // [rsp+10h] [rbp-1B0h]
  __int64 v42; // [rsp+10h] [rbp-1B0h]
  __int64 v43; // [rsp+20h] [rbp-1A0h] BYREF
  __int64 v44; // [rsp+28h] [rbp-198h] BYREF
  __m128i v45; // [rsp+30h] [rbp-190h] BYREF
  __m128i v46; // [rsp+40h] [rbp-180h]
  __m128i v47; // [rsp+50h] [rbp-170h]
  __m128i v48; // [rsp+60h] [rbp-160h]
  __m128i v49; // [rsp+70h] [rbp-150h] BYREF
  __m128i v50; // [rsp+80h] [rbp-140h]
  __m128i v51; // [rsp+90h] [rbp-130h]
  _OWORD v52[2]; // [rsp+A0h] [rbp-120h] BYREF
  __m128i v53; // [rsp+C0h] [rbp-100h]
  __m128i v54; // [rsp+D0h] [rbp-F0h]
  __m128i v55; // [rsp+E0h] [rbp-E0h]
  __m128i v56; // [rsp+F0h] [rbp-D0h]
  __m128i v57; // [rsp+100h] [rbp-C0h]
  __m128i v58; // [rsp+110h] [rbp-B0h]
  __m128i v59; // [rsp+120h] [rbp-A0h]
  __m128i v60; // [rsp+130h] [rbp-90h]
  __m128i v61; // [rsp+140h] [rbp-80h]
  __m128i v62; // [rsp+150h] [rbp-70h]
  __m128i v63; // [rsp+160h] [rbp-60h]
  __m128i v64; // [rsp+170h] [rbp-50h]
  __m128i v65; // [rsp+180h] [rbp-40h]

  v7 = a1[8].m128i_i64[1];
  if ( (unsigned int)sub_82ED00(a1, a2, a3) )
  {
    sub_6F3BA0(a1, 0);
    if ( a4 )
      *a4 = 1;
    return;
  }
  v8 = *(_BYTE *)(v7 + 80);
  v9 = v7;
  if ( v8 != 16 )
  {
    if ( v8 != 24 )
      goto LABEL_4;
LABEL_7:
    v9 = *(_QWORD *)(v9 + 88);
    v8 = *(_BYTE *)(v9 + 80);
    if ( v8 != 20 )
      goto LABEL_5;
LABEL_8:
    if ( sub_8BFF80(v9, a1[6].m128i_i64[1], &v43) )
    {
      v10 = v43;
LABEL_10:
      v11 = _mm_loadu_si128(a1 + 1);
      v12 = _mm_loadu_si128(a1 + 2);
      v13 = _mm_loadu_si128(a1 + 3);
      v14 = _mm_loadu_si128(a1 + 4);
      v15 = _mm_loadu_si128(a1 + 5);
      v45 = _mm_loadu_si128(a1);
      v16 = _mm_loadu_si128(a1 + 6);
      v17 = _mm_loadu_si128(a1 + 7);
      v46 = v11;
      v18 = _mm_loadu_si128(a1 + 8);
      v19 = a1[1].m128i_i8[0];
      v47 = v12;
      v48 = v13;
      v49 = v14;
      v50 = v15;
      v51 = v16;
      v52[0] = v17;
      v52[1] = v18;
      if ( v19 == 2 )
      {
        v29 = _mm_loadu_si128(a1 + 10);
        v30 = _mm_loadu_si128(a1 + 11);
        v31 = _mm_loadu_si128(a1 + 12);
        v32 = _mm_loadu_si128(a1 + 13);
        v53 = _mm_loadu_si128(a1 + 9);
        v33 = _mm_loadu_si128(a1 + 14);
        v34 = _mm_loadu_si128(a1 + 15);
        v54 = v29;
        v35 = _mm_loadu_si128(a1 + 16);
        v36 = _mm_loadu_si128(a1 + 17);
        v55 = v30;
        v37 = _mm_loadu_si128(a1 + 18);
        v56 = v31;
        v38 = _mm_loadu_si128(a1 + 19);
        v57 = v32;
        v39 = _mm_loadu_si128(a1 + 20);
        v58 = v33;
        v40 = _mm_loadu_si128(a1 + 21);
        v59 = v34;
        v60 = v35;
        v61 = v36;
        v62 = v37;
        v63 = v38;
        v64 = v39;
        v65 = v40;
      }
      else if ( v19 == 5 || v19 == 1 )
      {
        v53.m128i_i64[0] = a1[9].m128i_i64[0];
      }
      v20 = 0;
      if ( (v46.m128i_i8[3] & 2) != 0 )
        v20 = (char *)v52 + 8;
      v44 = v10;
      v41 = v20;
      v21 = sub_8B74F0(v7, &v44, 1, (char *)v49.m128i_i64 + 4);
      v22 = v21;
      if ( a3 )
        *a3 = v21;
      if ( *(_BYTE *)(v21 + 80) != 10 )
        goto LABEL_40;
      for ( i = *(_QWORD *)(*(_QWORD *)(v21 + 88) + 152LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      if ( *(_QWORD *)(*(_QWORD *)(i + 168) + 40LL) )
      {
        sub_6E7350(v22, (v46.m128i_i8[2] & 0x40) != 0, v50.m128i_i64[1], (__int64)a1);
        sub_6E4BC0((__int64)a1, (__int64)&v45);
        if ( (v46.m128i_i8[3] & 4) != 0 )
          a1[1].m128i_i8[2] |= 8u;
        if ( v46.m128i_i8[1] == 2 )
          sub_6EE880((__int64)a1, v41);
      }
      else
      {
LABEL_40:
        sub_6EAB60(
          v22,
          (v46.m128i_i8[2] & 0x40) != 0,
          0,
          &v49.m128i_i32[1],
          (__int64 *)((char *)&v49.m128i_i64[1] + 4),
          v50.m128i_i64[1],
          (__int64)a1);
        if ( v46.m128i_i8[1] == 2 )
          sub_6F5FA0(a1, v41, 0, a2, v27, v28);
        sub_6E4BC0((__int64)a1, (__int64)&v45);
        sub_6E5010(a1, &v45);
      }
    }
    return;
  }
  v9 = **(_QWORD **)(v7 + 88);
  v8 = *(_BYTE *)(v9 + 80);
  if ( v8 == 24 )
    goto LABEL_7;
LABEL_4:
  if ( v8 == 20 )
    goto LABEL_8;
LABEL_5:
  if ( v8 == 17 )
  {
    v24 = sub_82C1B0(v9, 0, 0, &v45);
    if ( v24 )
    {
      v10 = 0;
      v7 = 0;
      do
      {
        v25 = *(_BYTE *)(v24 + 80);
        v26 = v24;
        if ( v25 == 16 )
        {
          v26 = **(_QWORD **)(v24 + 88);
          v25 = *(_BYTE *)(v26 + 80);
        }
        if ( v25 == 24 )
        {
          v26 = *(_QWORD *)(v26 + 88);
          v25 = *(_BYTE *)(v26 + 80);
        }
        if ( v25 == 20 )
        {
          v42 = v24;
          if ( sub_8BFF80(v26, a1[6].m128i_i64[1], &v43) )
          {
            if ( v7 )
              goto LABEL_44;
            v10 = v43;
            v7 = v42;
          }
        }
        v24 = sub_82C230(&v45);
      }
      while ( v24 );
      if ( v7 )
        goto LABEL_10;
LABEL_44:
      if ( v10 )
        sub_725130(v10);
    }
  }
}
