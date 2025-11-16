// Function: sub_6A8930
// Address: 0x6a8930
//
__int64 __fastcall sub_6A8930(__int64 a1, __m128i *a2, __int64 a3, __int64 a4)
{
  __int16 v4; // bx
  __int64 v5; // rdx
  char i; // al
  __int32 v7; // eax
  __int64 result; // rax
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rax
  __m128i v14; // xmm7
  __m128i v15; // xmm0
  __m128i v16; // xmm1
  __m128i v17; // xmm2
  __m128i v18; // xmm3
  __m128i v19; // xmm4
  __m128i v20; // xmm5
  __m128i v21; // xmm6
  __int8 v22; // al
  __int64 v23; // rax
  __m128i v24; // xmm1
  __m128i v25; // xmm2
  __m128i v26; // xmm3
  __m128i v27; // xmm4
  __m128i v28; // xmm5
  __m128i v29; // xmm6
  __m128i v30; // xmm7
  __m128i v31; // xmm0
  __m128i v32; // xmm1
  __m128i v33; // xmm2
  __m128i v34; // xmm3
  __m128i v35; // xmm4
  __m128i v36; // xmm5
  __m128i v37; // xmm6
  __m128i v38; // xmm7
  __m128i v39; // xmm1
  __m128i v40; // xmm2
  __m128i v41; // xmm3
  __m128i v42; // xmm4
  __m128i v43; // xmm5
  __m128i v44; // xmm0
  __m128i v45; // xmm1
  __m128i v46; // xmm2
  __m128i v47; // xmm7
  __m128i v48; // xmm3
  __m128i v49; // xmm4
  __m128i v50; // xmm5
  __m128i v51; // xmm7
  __m128i v52; // xmm6
  __m128i v53; // xmm0
  __m128i v54; // xmm7
  __m128i v55; // xmm7
  __int64 v56; // [rsp+0h] [rbp-390h]
  _BOOL4 v57; // [rsp+8h] [rbp-388h]
  __int32 v58; // [rsp+Ch] [rbp-384h]
  char v59; // [rsp+10h] [rbp-380h] BYREF
  int v60; // [rsp+14h] [rbp-37Ch] BYREF
  __int64 v61; // [rsp+18h] [rbp-378h] BYREF
  __m128i v62; // [rsp+20h] [rbp-370h] BYREF
  __m128i v63; // [rsp+30h] [rbp-360h] BYREF
  __m128i v64; // [rsp+40h] [rbp-350h] BYREF
  __m128i v65; // [rsp+50h] [rbp-340h] BYREF
  __m128i v66; // [rsp+60h] [rbp-330h] BYREF
  __m128i v67; // [rsp+70h] [rbp-320h] BYREF
  __m128i v68; // [rsp+80h] [rbp-310h] BYREF
  __m128i v69; // [rsp+90h] [rbp-300h] BYREF
  __m128i v70; // [rsp+A0h] [rbp-2F0h] BYREF
  __m128i v71; // [rsp+B0h] [rbp-2E0h] BYREF
  __m128i v72; // [rsp+C0h] [rbp-2D0h] BYREF
  __m128i v73; // [rsp+D0h] [rbp-2C0h] BYREF
  __m128i v74; // [rsp+E0h] [rbp-2B0h] BYREF
  __m128i v75; // [rsp+F0h] [rbp-2A0h] BYREF
  __m128i v76; // [rsp+100h] [rbp-290h] BYREF
  __m128i v77; // [rsp+110h] [rbp-280h] BYREF
  __m128i v78; // [rsp+120h] [rbp-270h] BYREF
  __m128i v79; // [rsp+130h] [rbp-260h] BYREF
  __m128i v80; // [rsp+140h] [rbp-250h] BYREF
  __m128i v81; // [rsp+150h] [rbp-240h] BYREF
  __m128i v82; // [rsp+160h] [rbp-230h] BYREF
  __m128i v83; // [rsp+170h] [rbp-220h] BYREF
  _QWORD v84[66]; // [rsp+180h] [rbp-210h] BYREF

  if ( a1 )
  {
    sub_6F8AB0(a1, (unsigned int)&v62, 0, 0, (unsigned int)&v61, (unsigned int)&v59, 0);
  }
  else
  {
    v61 = *(_QWORD *)&dword_4F063F8;
    sub_7B8B50(0, a2, a3, a4);
    sub_7BE280(27, 125, 0, 0);
    ++*(_BYTE *)(qword_4F061C8 + 36LL);
    memset(v84, 0, 0x1D8u);
    v84[19] = v84;
    v84[3] = *(_QWORD *)&dword_4F063F8;
    if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
      BYTE2(v84[22]) |= 1u;
    sub_69ED20((__int64)&v62, 0, 0, 1);
    v58 = qword_4F063F0;
    v4 = WORD2(qword_4F063F0);
    if ( dword_4F04C44 == -1 )
    {
      v23 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      if ( (*(_BYTE *)(v23 + 6) & 6) == 0
        && *(_BYTE *)(v23 + 4) != 12
        && (v63.m128i_i8[1] == 2 || (unsigned int)sub_6ED0A0(&v62)) )
      {
        sub_6E68E0(158, &v62);
      }
    }
  }
  if ( !v63.m128i_i8[0] )
    goto LABEL_7;
  v5 = v62.m128i_i64[0];
  for ( i = *(_BYTE *)(v62.m128i_i64[0] + 140); i == 12; i = *(_BYTE *)(v5 + 140) )
    v5 = *(_QWORD *)(v5 + 160);
  if ( i )
  {
    sub_6F69D0(&v62, 47);
    sub_6FED50(&v62, 0, 0, 0, 1, &v61);
    if ( v63.m128i_i8[0] == 2 )
    {
      v24 = _mm_loadu_si128(&v63);
      v25 = _mm_loadu_si128(&v64);
      v26 = _mm_loadu_si128(&v65);
      v27 = _mm_loadu_si128(&v66);
      v28 = _mm_loadu_si128(&v67);
      *a2 = _mm_loadu_si128(&v62);
      v29 = _mm_loadu_si128(&v68);
      v30 = _mm_loadu_si128(&v69);
      a2[1] = v24;
      v31 = _mm_loadu_si128(&v70);
      v32 = _mm_loadu_si128(&v71);
      a2[2] = v25;
      a2[3] = v26;
      v33 = _mm_loadu_si128(&v72);
      v34 = _mm_loadu_si128(&v73);
      a2[4] = v27;
      v35 = _mm_loadu_si128(&v74);
      a2[5] = v28;
      v36 = _mm_loadu_si128(&v75);
      a2[6] = v29;
      v37 = _mm_loadu_si128(&v76);
      a2[7] = v30;
      v38 = _mm_loadu_si128(&v77);
      a2[9] = v32;
      a2[10] = v33;
      a2[11] = v34;
      a2[12] = v35;
      a2[13] = v36;
      a2[8] = v31;
      a2[14] = v37;
      a2[15] = v38;
      v39 = _mm_loadu_si128(&v79);
      v40 = _mm_loadu_si128(&v80);
      v41 = _mm_loadu_si128(&v81);
      v42 = _mm_loadu_si128(&v82);
      v43 = _mm_loadu_si128(&v83);
      a2[16] = _mm_loadu_si128(&v78);
      a2[17] = v39;
      a2[18] = v40;
      a2[19] = v41;
      a2[20] = v42;
      a2[21] = v43;
    }
    else
    {
      v84[0] = sub_724DC0(&v62, 0, v9, v10, v11, v12);
      v56 = v84[0];
      v57 = *(_BYTE *)(qword_4D03C50 + 16LL) != 0;
      v13 = sub_6F6F40(&v62, 0);
      sub_7197C0(v13, v56, v57, &v61, &v60);
      if ( v60 )
      {
        v14 = _mm_loadu_si128(&v63);
        v15 = _mm_loadu_si128(&v64);
        v16 = _mm_loadu_si128(&v65);
        v17 = _mm_loadu_si128(&v66);
        v18 = _mm_loadu_si128(&v67);
        *a2 = _mm_loadu_si128(&v62);
        v19 = _mm_loadu_si128(&v68);
        v20 = _mm_loadu_si128(&v69);
        a2[1] = v14;
        v21 = _mm_loadu_si128(&v70);
        v22 = v63.m128i_i8[0];
        a2[2] = v15;
        a2[3] = v16;
        a2[4] = v17;
        a2[5] = v18;
        a2[6] = v19;
        a2[7] = v20;
        a2[8] = v21;
        if ( v22 == 2 )
        {
          v44 = _mm_loadu_si128(&v75);
          v45 = _mm_loadu_si128(&v76);
          v46 = _mm_loadu_si128(&v77);
          a2[9] = _mm_loadu_si128(&v71);
          v47 = _mm_loadu_si128(&v72);
          v48 = _mm_loadu_si128(&v78);
          v49 = _mm_loadu_si128(&v79);
          v50 = _mm_loadu_si128(&v80);
          a2[13] = v44;
          a2[10] = v47;
          v51 = _mm_loadu_si128(&v73);
          v52 = _mm_loadu_si128(&v81);
          v53 = _mm_loadu_si128(&v83);
          a2[14] = v45;
          a2[11] = v51;
          v54 = _mm_loadu_si128(&v74);
          a2[15] = v46;
          a2[12] = v54;
          v55 = _mm_loadu_si128(&v82);
          a2[16] = v48;
          a2[17] = v49;
          a2[18] = v50;
          a2[19] = v52;
          a2[20] = v55;
          a2[21] = v53;
        }
        else if ( v22 == 5 || v22 == 1 )
        {
          a2[9].m128i_i64[0] = v71.m128i_i64[0];
        }
      }
      else
      {
        sub_6E6A50(v84[0], a2);
        a2->m128i_i64[0] = a2[17].m128i_i64[0];
      }
      sub_724E30(v84);
    }
  }
  else
  {
LABEL_7:
    sub_6E6260(a2);
    sub_6E6450(&v62);
  }
  v7 = v61;
  a2[5].m128i_i16[0] = v4;
  a2[4].m128i_i32[1] = v7;
  a2[4].m128i_i16[4] = WORD2(v61);
  *(_QWORD *)dword_4F07508 = *(__int64 *)((char *)a2[4].m128i_i64 + 4);
  a2[4].m128i_i32[3] = v58;
  unk_4F061D8 = *(__int64 *)((char *)&a2[4].m128i_i64[1] + 4);
  sub_6E3280(a2, &v61);
  result = sub_6E3BA0(a2, &v61, 0, 0);
  if ( !a1 )
  {
    --*(_BYTE *)(qword_4F061C8 + 36LL);
    return sub_7BE280(28, 18, 0, 0);
  }
  return result;
}
