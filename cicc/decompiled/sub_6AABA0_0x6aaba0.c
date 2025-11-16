// Function: sub_6AABA0
// Address: 0x6aaba0
//
__int64 __fastcall sub_6AABA0(_WORD *a1, __m128i *a2, __int64 a3, __int64 a4)
{
  __m128i *v5; // rsi
  __int32 v6; // r13d
  __int16 v7; // bx
  __int64 v8; // rdi
  char v9; // dl
  __int64 v10; // rax
  __int32 v11; // eax
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __m128i v17; // xmm1
  __m128i v18; // xmm2
  __m128i v19; // xmm3
  __m128i v20; // xmm4
  __m128i v21; // xmm5
  __m128i v22; // xmm6
  __m128i v23; // xmm7
  __m128i v24; // xmm0
  __int8 v25; // al
  __int64 i; // rax
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r14
  __m128i v30; // xmm2
  __m128i v31; // xmm3
  __m128i v32; // xmm4
  __m128i v33; // xmm5
  __m128i v34; // xmm6
  __m128i v35; // xmm7
  __m128i v36; // xmm1
  __m128i v37; // xmm2
  __m128i v38; // xmm3
  __m128i v39; // xmm4
  __m128i v40; // xmm5
  __m128i v41; // xmm6
  int v42; // eax
  __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // [rsp+0h] [rbp-1B0h]
  __int64 v46; // [rsp+0h] [rbp-1B0h]
  _BOOL4 v47; // [rsp+Ch] [rbp-1A4h]
  unsigned int v48; // [rsp+Ch] [rbp-1A4h]
  __int64 v49; // [rsp+10h] [rbp-1A0h] BYREF
  __int64 v50; // [rsp+18h] [rbp-198h] BYREF
  __m128i v51; // [rsp+20h] [rbp-190h] BYREF
  __m128i v52; // [rsp+30h] [rbp-180h] BYREF
  __m128i v53; // [rsp+40h] [rbp-170h] BYREF
  __m128i v54; // [rsp+50h] [rbp-160h] BYREF
  __m128i v55; // [rsp+60h] [rbp-150h] BYREF
  __m128i v56; // [rsp+70h] [rbp-140h] BYREF
  __m128i v57; // [rsp+80h] [rbp-130h] BYREF
  __m128i v58; // [rsp+90h] [rbp-120h] BYREF
  __m128i v59; // [rsp+A0h] [rbp-110h] BYREF
  __m128i v60; // [rsp+B0h] [rbp-100h] BYREF
  __m128i v61; // [rsp+C0h] [rbp-F0h] BYREF
  __m128i v62; // [rsp+D0h] [rbp-E0h] BYREF
  __m128i v63; // [rsp+E0h] [rbp-D0h] BYREF
  __m128i v64; // [rsp+F0h] [rbp-C0h] BYREF
  __m128i v65; // [rsp+100h] [rbp-B0h] BYREF
  __m128i v66; // [rsp+110h] [rbp-A0h] BYREF
  __m128i v67; // [rsp+120h] [rbp-90h] BYREF
  __m128i v68; // [rsp+130h] [rbp-80h] BYREF
  __m128i v69; // [rsp+140h] [rbp-70h] BYREF
  __m128i v70; // [rsp+150h] [rbp-60h] BYREF
  __m128i v71; // [rsp+160h] [rbp-50h] BYREF
  __m128i v72[4]; // [rsp+170h] [rbp-40h] BYREF

  if ( a1 )
  {
    v5 = &v51;
    v47 = a1[4] == 144;
    sub_6F8AB0((_DWORD)a1, (unsigned int)&v51, 0, 0, (unsigned int)&v50, (unsigned int)&v49, 0);
    v6 = *(_DWORD *)(*(_QWORD *)a1 + 44LL);
    v7 = *(_WORD *)(*(_QWORD *)a1 + 48LL);
  }
  else
  {
    v50 = *(_QWORD *)&dword_4F063F8;
    v47 = word_4F06418[0] == 144;
    sub_7B8B50(0, a2, a3, a4);
    v5 = 0;
    sub_69ED20((__int64)&v51, 0, 17, 0);
    v6 = v55.m128i_i32[3];
    v7 = v56.m128i_i16[0];
  }
  if ( !v52.m128i_i8[0] )
    goto LABEL_8;
  v8 = v51.m128i_i64[0];
  v9 = *(_BYTE *)(v51.m128i_i64[0] + 140);
  if ( v9 == 12 )
  {
    v10 = v51.m128i_i64[0];
    do
    {
      v10 = *(_QWORD *)(v10 + 160);
      v9 = *(_BYTE *)(v10 + 140);
    }
    while ( v9 == 12 );
  }
  if ( !v9 )
  {
LABEL_8:
    sub_6E6260(a2);
    goto LABEL_9;
  }
  if ( (unsigned int)sub_8D2AC0(v51.m128i_i64[0]) || (v8 = v51.m128i_i64[0], (unsigned int)sub_8D2930(v51.m128i_i64[0])) )
  {
    if ( v47 )
    {
      v17 = _mm_loadu_si128(&v52);
      v18 = _mm_loadu_si128(&v53);
      v19 = _mm_loadu_si128(&v54);
      v20 = _mm_loadu_si128(&v55);
      v21 = _mm_loadu_si128(&v56);
      *a2 = _mm_loadu_si128(&v51);
      v22 = _mm_loadu_si128(&v57);
      v23 = _mm_loadu_si128(&v58);
      a2[1] = v17;
      v24 = _mm_loadu_si128(&v59);
      v25 = v52.m128i_i8[0];
      a2[2] = v18;
      a2[3] = v19;
      a2[4] = v20;
      a2[5] = v21;
      a2[6] = v22;
      a2[7] = v23;
      a2[8] = v24;
      if ( v25 == 2 )
      {
        v30 = _mm_loadu_si128(&v61);
        v31 = _mm_loadu_si128(&v62);
        v32 = _mm_loadu_si128(&v63);
        v33 = _mm_loadu_si128(&v64);
        v34 = _mm_loadu_si128(&v65);
        a2[9] = _mm_loadu_si128(&v60);
        v35 = _mm_loadu_si128(&v66);
        v36 = _mm_loadu_si128(&v67);
        a2[10] = v30;
        a2[11] = v31;
        v37 = _mm_loadu_si128(&v68);
        v38 = _mm_loadu_si128(&v69);
        a2[12] = v32;
        v39 = _mm_loadu_si128(&v70);
        a2[13] = v33;
        v40 = _mm_loadu_si128(&v71);
        a2[14] = v34;
        v41 = _mm_loadu_si128(v72);
        a2[15] = v35;
        a2[16] = v36;
        a2[17] = v37;
        a2[18] = v38;
        a2[19] = v39;
        a2[20] = v40;
        a2[21] = v41;
      }
      else if ( v25 == 5 || v25 == 1 )
      {
        a2[9].m128i_i64[0] = v60.m128i_i64[0];
      }
    }
    else
    {
      v29 = v51.m128i_i64[0];
      v49 = sub_724DC0(v8, v5, v13, v14, v15, v16);
      sub_72BB40(v29, v49);
      sub_6E6A50(v49, a2);
      sub_724E30(&v49);
    }
    if ( (unsigned int)sub_6E53E0(5, 1443, &v50) )
      sub_684B30(0x5A3u, &v50);
  }
  else
  {
    v48 = 34 - v47;
    if ( !(unsigned int)sub_8D2B50(v51.m128i_i64[0]) )
    {
      if ( (unsigned int)sub_8D3D40(v51.m128i_i64[0]) )
      {
        sub_7032B0(v48, &v51, a2, &v50, 0);
        goto LABEL_9;
      }
      sub_6E68E0(1442, &v51);
      goto LABEL_8;
    }
    for ( i = v51.m128i_i64[0]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
      ;
    v27 = sub_72C610(*(unsigned __int8 *)(i + 160));
    v28 = v27;
    if ( v52.m128i_i8[1] != 1 || (v45 = v27, v42 = sub_6ED0A0(&v51), v28 = v45, v42) )
    {
      sub_6FEAC0(v48, &v51, v28, a2, &v50, 0);
    }
    else
    {
      sub_6ECC10(&v51, v5, v45);
      v46 = sub_73CA70(v45, v51.m128i_i64[0]);
      v43 = sub_6F6F40(&v51, 0);
      v44 = sub_73DC30(v48, v46, v43);
      sub_6E7150(v44, a2);
      a2[5].m128i_i64[1] = v56.m128i_i64[1];
    }
  }
LABEL_9:
  v11 = v50;
  a2[4].m128i_i32[3] = v6;
  a2[5].m128i_i16[0] = v7;
  a2[4].m128i_i32[1] = v11;
  a2[4].m128i_i16[4] = WORD2(v50);
  *(_QWORD *)dword_4F07508 = *(__int64 *)((char *)a2[4].m128i_i64 + 4);
  unk_4F061D8 = *(__int64 *)((char *)&a2[4].m128i_i64[1] + 4);
  sub_6E3280(a2, &v50);
  return sub_6E3BA0(a2, &v50, 0, 0);
}
