// Function: sub_6DE240
// Address: 0x6de240
//
__int64 __fastcall sub_6DE240(__m128i *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  char i; // dl
  __int64 v9; // rsi
  __int64 v10; // rdi
  int v11; // ebx
  char v12; // al
  int v13; // r15d
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // r8
  __int32 v17; // edx
  __int16 v18; // ax
  __int64 v19; // rsi
  __int64 v20; // rdx
  __m128i v22; // xmm1
  __m128i v23; // xmm2
  __m128i v24; // xmm3
  __m128i v25; // xmm4
  __m128i v26; // xmm5
  __m128i v27; // xmm6
  __m128i v28; // xmm7
  __m128i v29; // xmm0
  __int8 v30; // al
  __m128i v31; // xmm2
  __m128i v32; // xmm3
  __m128i v33; // xmm4
  __m128i v34; // xmm5
  __m128i v35; // xmm6
  __m128i v36; // xmm7
  __m128i v37; // xmm1
  __m128i v38; // xmm2
  __m128i v39; // xmm3
  __m128i v40; // xmm4
  __m128i v41; // xmm5
  __m128i v42; // xmm6
  char v43; // [rsp+Fh] [rbp-261h]
  __int64 v44; // [rsp+10h] [rbp-260h]
  unsigned int v45; // [rsp+2Ch] [rbp-244h] BYREF
  __int64 v46; // [rsp+30h] [rbp-240h] BYREF
  __int64 v47; // [rsp+38h] [rbp-238h] BYREF
  char v48[160]; // [rsp+40h] [rbp-230h] BYREF
  __m128i v49; // [rsp+E0h] [rbp-190h] BYREF
  __m128i v50; // [rsp+F0h] [rbp-180h] BYREF
  __m128i v51; // [rsp+100h] [rbp-170h] BYREF
  __m128i v52; // [rsp+110h] [rbp-160h] BYREF
  __m128i v53; // [rsp+120h] [rbp-150h] BYREF
  __m128i v54; // [rsp+130h] [rbp-140h] BYREF
  __m128i v55; // [rsp+140h] [rbp-130h] BYREF
  __m128i v56; // [rsp+150h] [rbp-120h] BYREF
  __m128i v57; // [rsp+160h] [rbp-110h] BYREF
  __m128i v58; // [rsp+170h] [rbp-100h] BYREF
  __m128i v59; // [rsp+180h] [rbp-F0h] BYREF
  __m128i v60; // [rsp+190h] [rbp-E0h] BYREF
  __m128i v61; // [rsp+1A0h] [rbp-D0h] BYREF
  __m128i v62; // [rsp+1B0h] [rbp-C0h] BYREF
  __m128i v63; // [rsp+1C0h] [rbp-B0h] BYREF
  __m128i v64; // [rsp+1D0h] [rbp-A0h] BYREF
  __m128i v65; // [rsp+1E0h] [rbp-90h] BYREF
  __m128i v66; // [rsp+1F0h] [rbp-80h] BYREF
  __m128i v67; // [rsp+200h] [rbp-70h] BYREF
  __m128i v68; // [rsp+210h] [rbp-60h] BYREF
  __m128i v69; // [rsp+220h] [rbp-50h] BYREF
  __m128i v70[4]; // [rsp+230h] [rbp-40h] BYREF

  v45 = 0;
  v47 = *(_QWORD *)&dword_4F063F8;
  sub_7B8B50(a1, a2, a3, a4);
  sub_7BE280(27, 125, 0, 0);
  v5 = qword_4F061C8;
  v6 = qword_4D03C50;
  ++*(_BYTE *)(qword_4F061C8 + 36LL);
  ++*(_QWORD *)(v6 + 40);
  ++*(_BYTE *)(v5 + 75);
  sub_6E1E00(5, v48, 0, 0);
  v7 = sub_6AB980();
  v46 = v7;
  for ( i = *(_BYTE *)(v7 + 140); i == 12; i = *(_BYTE *)(v7 + 140) )
    v7 = *(_QWORD *)(v7 + 160);
  if ( !i )
    v45 = 1;
  sub_7BE280(67, 253, 0, 0);
  if ( word_4F06418[0] != 28 && word_4F06418[0] != 67 )
    sub_6ABAC0(&v46, &v45);
  v9 = 253;
  v10 = 67;
  sub_7BE280(67, 253, 0, 0);
  if ( word_4F06418[0] != 67 && word_4F06418[0] != 28 )
  {
    v9 = (__int64)&v45;
    v10 = (__int64)&v46;
    sub_6ABAC0(&v46, &v45);
  }
  sub_6E2B30(v10, v9);
  v11 = -1;
  v44 = v46;
  if ( !v45 )
  {
    v12 = *(_BYTE *)(v46 + 160);
    switch ( v12 )
    {
      case 4:
        v11 = 4;
        break;
      case 6:
        v11 = 6;
        break;
      case 2:
        v11 = 5;
        break;
      default:
        sub_721090(v10);
    }
    if ( *(_BYTE *)(v46 + 140) == 5 )
      v11 += 3;
  }
  v13 = 4;
  v43 = *(_BYTE *)(qword_4D03C50 + 17LL);
  do
  {
    sub_7BE280(67, 253, 0, 0);
    if ( word_4F06418[0] != 67 && word_4F06418[0] != 28 )
    {
      *(_BYTE *)(qword_4D03C50 + 17LL) = (v11 == v13) | (2 * (v11 == v13)) | *(_BYTE *)(qword_4D03C50 + 17LL) & 0xFC;
      sub_69ED20((__int64)&v49, 0, 0, 1);
      if ( v11 == v13 )
      {
        v22 = _mm_loadu_si128(&v50);
        v23 = _mm_loadu_si128(&v51);
        v24 = _mm_loadu_si128(&v52);
        v25 = _mm_loadu_si128(&v53);
        v26 = _mm_loadu_si128(&v54);
        *a1 = _mm_loadu_si128(&v49);
        v27 = _mm_loadu_si128(&v55);
        v28 = _mm_loadu_si128(&v56);
        a1[1] = v22;
        v29 = _mm_loadu_si128(&v57);
        v30 = v50.m128i_i8[0];
        a1[2] = v23;
        a1[3] = v24;
        a1[4] = v25;
        a1[5] = v26;
        a1[6] = v27;
        a1[7] = v28;
        a1[8] = v29;
        if ( v30 == 2 )
        {
          v31 = _mm_loadu_si128(&v59);
          v32 = _mm_loadu_si128(&v60);
          v33 = _mm_loadu_si128(&v61);
          v34 = _mm_loadu_si128(&v62);
          v35 = _mm_loadu_si128(&v63);
          a1[9] = _mm_loadu_si128(&v58);
          v36 = _mm_loadu_si128(&v64);
          v37 = _mm_loadu_si128(&v65);
          a1[10] = v31;
          a1[11] = v32;
          v38 = _mm_loadu_si128(&v66);
          v39 = _mm_loadu_si128(&v67);
          a1[12] = v33;
          v40 = _mm_loadu_si128(&v68);
          a1[13] = v34;
          v41 = _mm_loadu_si128(&v69);
          a1[14] = v35;
          v42 = _mm_loadu_si128(v70);
          a1[15] = v36;
          a1[16] = v37;
          a1[17] = v38;
          a1[18] = v39;
          a1[19] = v40;
          a1[20] = v41;
          a1[21] = v42;
          sub_6F69D0(a1, 0);
        }
        else if ( v30 == 5 || v30 == 1 )
        {
          a1[9].m128i_i64[0] = v58.m128i_i64[0];
          sub_6F69D0(a1, 0);
        }
        else
        {
          sub_6F69D0(a1, 0);
        }
      }
      goto LABEL_21;
    }
    if ( v11 == v13 || v11 > v13 && word_4F06418[0] == 28 )
    {
      LOBYTE(v14) = word_4F06418[0] == 28;
      if ( (unsigned int)sub_6E5430(67, 253, v14, v15, v16) )
        sub_685360(0x408u, &v47, v44);
      v45 = 1;
      if ( word_4F06418[0] == 28 )
        break;
    }
LABEL_21:
    ++v13;
  }
  while ( v13 != 10 );
  *(_BYTE *)(qword_4D03C50 + 17LL) = v43 & 3 | *(_BYTE *)(qword_4D03C50 + 17LL) & 0xFC;
  --*(_BYTE *)(qword_4F061C8 + 75LL);
  sub_7BE280(28, 18, 0, 0);
  --*(_BYTE *)(qword_4F061C8 + 36LL);
  --*(_QWORD *)(qword_4D03C50 + 40LL);
  if ( v45 )
    sub_6E6260(a1);
  a1[4].m128i_i32[1] = v47;
  v17 = qword_4F063F0;
  v18 = WORD2(qword_4F063F0);
  a1[4].m128i_i16[4] = WORD2(v47);
  v19 = *(__int64 *)((char *)a1[4].m128i_i64 + 4);
  a1[5].m128i_i16[0] = v18;
  a1[4].m128i_i32[3] = v17;
  v20 = *(__int64 *)((char *)&a1[4].m128i_i64[1] + 4);
  *(_QWORD *)dword_4F07508 = v19;
  *(_QWORD *)&dword_4F061D8 = v20;
  return sub_6E3280(a1, &v47);
}
