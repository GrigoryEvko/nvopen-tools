// Function: sub_837960
// Address: 0x837960
//
__int64 __fastcall sub_837960(
        __m128i *a1,
        const __m128i *a2,
        unsigned int a3,
        unsigned int a4,
        int a5,
        __int64 *a6,
        __m128i *a7)
{
  __int64 result; // rax
  __int64 v11; // rax
  char v12; // dl
  __m128i v13; // xmm1
  __m128i v14; // xmm2
  __m128i v15; // xmm3
  __m128i v16; // xmm4
  __m128i v17; // xmm5
  __m128i v18; // xmm6
  __int8 v19; // al
  __m128i v20; // xmm7
  __m128i v21; // xmm0
  __int64 v22; // rax
  __m128i v23; // xmm1
  __m128i v24; // xmm2
  __m128i v25; // xmm3
  __m128i v26; // xmm7
  __m128i v27; // xmm4
  __m128i v28; // xmm5
  __m128i v29; // xmm6
  __int8 v30; // dl
  __m128i v31; // xmm7
  __int64 v32; // rcx
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 i; // rax
  __int64 v37; // rax
  __m128i v38; // xmm2
  __m128i v39; // xmm3
  __m128i v40; // xmm4
  __m128i v41; // xmm5
  __m128i v42; // xmm6
  __m128i v43; // xmm1
  __m128i v44; // xmm2
  __int8 v45; // dl
  __m128i v46; // xmm3
  __m128i v47; // xmm2
  __m128i v48; // xmm3
  __m128i v49; // xmm4
  __m128i v50; // xmm5
  __m128i v51; // xmm6
  __m128i v52; // xmm7
  __m128i v53; // xmm1
  __m128i v54; // xmm2
  __m128i v55; // xmm3
  __m128i v56; // xmm4
  __m128i v57; // xmm5
  __m128i v58; // xmm6
  __m128i v59; // xmm5
  __m128i v60; // xmm6
  __m128i v61; // xmm7
  __m128i v62; // xmm1
  __m128i v63; // xmm2
  __m128i v64; // xmm4
  __m128i v65; // xmm3
  __m128i v66; // xmm5
  __m128i v67; // xmm6
  __m128i v68; // xmm4
  __m128i v69; // xmm5
  __m128i v70; // xmm6
  __m128i v71; // xmm4
  __m128i v72; // xmm5
  __m128i v73; // xmm6
  __m128i v74; // xmm1
  __m128i v75; // xmm2
  __m128i v76; // xmm3
  __m128i v77; // xmm0
  __m128i v78; // xmm7
  __m128i v79; // xmm4
  __m128i v80; // xmm5
  __m128i v81; // xmm6
  __m128i v82; // xmm1
  unsigned int v84; // [rsp+24h] [rbp-19Ch] BYREF
  __int64 v85; // [rsp+28h] [rbp-198h] BYREF
  __m128i v86; // [rsp+30h] [rbp-190h] BYREF
  __m128i v87; // [rsp+40h] [rbp-180h] BYREF
  __m128i v88; // [rsp+50h] [rbp-170h] BYREF
  __m128i v89; // [rsp+60h] [rbp-160h] BYREF
  __m128i v90; // [rsp+70h] [rbp-150h] BYREF
  __m128i v91; // [rsp+80h] [rbp-140h] BYREF
  __m128i v92; // [rsp+90h] [rbp-130h] BYREF
  __m128i v93; // [rsp+A0h] [rbp-120h] BYREF
  __m128i v94; // [rsp+B0h] [rbp-110h] BYREF
  __m128i v95; // [rsp+C0h] [rbp-100h] BYREF
  __m128i v96; // [rsp+D0h] [rbp-F0h] BYREF
  __m128i v97; // [rsp+E0h] [rbp-E0h] BYREF
  __m128i v98; // [rsp+F0h] [rbp-D0h] BYREF
  __m128i v99; // [rsp+100h] [rbp-C0h] BYREF
  __m128i v100; // [rsp+110h] [rbp-B0h] BYREF
  __m128i v101; // [rsp+120h] [rbp-A0h] BYREF
  __m128i v102; // [rsp+130h] [rbp-90h] BYREF
  __m128i v103; // [rsp+140h] [rbp-80h] BYREF
  __m128i v104; // [rsp+150h] [rbp-70h] BYREF
  __m128i v105; // [rsp+160h] [rbp-60h] BYREF
  __m128i v106; // [rsp+170h] [rbp-50h] BYREF
  __m128i v107[4]; // [rsp+180h] [rbp-40h] BYREF

  if ( !(unsigned int)sub_6E9790((__int64)a1, &v85)
    && (dword_4F077C4 != 2 || unk_4F07778 <= 202001 || !(unsigned int)sub_6EA170((__int64)a1, &v85))
    || !(unsigned int)sub_695430(v85, (a5 & 2) != 0, 1) )
  {
    return 0;
  }
  if ( (a5 & 2) == 0 )
  {
    v11 = 776LL * dword_4F04C64;
    v12 = *(_BYTE *)(qword_4F04C68[0] + v11 + 7);
    if ( v12 < 0 )
    {
      v32 = *(_QWORD *)(v85 + 40);
      if ( *(_QWORD *)(qword_4F04C68[0] + v11 + 184) != v32 )
      {
        v33 = qword_4F04C68[0] + v11 - 776;
        while ( (v12 & 0x40) == 0 )
        {
          v34 = v33;
          v33 -= 776;
          if ( *(_QWORD *)(v33 + 960) == v32 )
            goto LABEL_10;
          v12 = *(_BYTE *)(v34 + 7);
        }
        return 0;
      }
    }
  }
LABEL_10:
  v13 = _mm_loadu_si128(a1 + 1);
  v14 = _mm_loadu_si128(a1 + 2);
  v15 = _mm_loadu_si128(a1 + 3);
  v16 = _mm_loadu_si128(a1 + 4);
  v17 = _mm_loadu_si128(a1 + 5);
  v86 = _mm_loadu_si128(a1);
  v18 = _mm_loadu_si128(a1 + 6);
  v19 = a1[1].m128i_i8[0];
  v87 = v13;
  v20 = _mm_loadu_si128(a1 + 7);
  v88 = v14;
  v21 = _mm_loadu_si128(a1 + 8);
  v89 = v15;
  v90 = v16;
  v91 = v17;
  v92 = v18;
  v93 = v20;
  v94 = v21;
  if ( v19 == 2 )
  {
    v47 = _mm_loadu_si128(a1 + 10);
    v48 = _mm_loadu_si128(a1 + 11);
    v49 = _mm_loadu_si128(a1 + 12);
    v50 = _mm_loadu_si128(a1 + 13);
    v95 = _mm_loadu_si128(a1 + 9);
    v51 = _mm_loadu_si128(a1 + 14);
    v52 = _mm_loadu_si128(a1 + 15);
    v96 = v47;
    v53 = _mm_loadu_si128(a1 + 16);
    v54 = _mm_loadu_si128(a1 + 17);
    v97 = v48;
    v55 = _mm_loadu_si128(a1 + 18);
    v98 = v49;
    v56 = _mm_loadu_si128(a1 + 19);
    v99 = v50;
    v57 = _mm_loadu_si128(a1 + 20);
    v100 = v51;
    v58 = _mm_loadu_si128(a1 + 21);
    v101 = v52;
    v102 = v53;
    v103 = v54;
    v104 = v55;
    v105 = v56;
    v106 = v57;
    v107[0] = v58;
  }
  else if ( v19 == 5 || v19 == 1 )
  {
    v95.m128i_i64[0] = a1[9].m128i_i64[0];
  }
  v22 = sub_72D6A0(v86.m128i_i64[0]);
  sub_6FAB30(&v86, v22, 0, 1u, 0);
  result = sub_836C50(&v86, 0, a2, 1u, a3, a4, 0, 0, a5, (__int64)a6, a7, &v84, 0);
  if ( !(_DWORD)result )
  {
    if ( v84 )
    {
      v23 = _mm_loadu_si128(&v88);
      v24 = _mm_loadu_si128(&v89);
      v25 = _mm_loadu_si128(&v90);
      *a1 = _mm_loadu_si128(&v86);
      v26 = _mm_loadu_si128(&v87);
      v27 = _mm_loadu_si128(&v91);
      v28 = _mm_loadu_si128(&v92);
      v29 = _mm_loadu_si128(&v93);
      a1[2] = v23;
      a1[1] = v26;
      v30 = v87.m128i_i8[0];
      v31 = _mm_loadu_si128(&v94);
      a1[3] = v24;
      a1[4] = v25;
      a1[5] = v27;
      a1[6] = v28;
      a1[7] = v29;
      a1[8] = v31;
      if ( v30 == 2 )
      {
        v59 = _mm_loadu_si128(&v96);
        v60 = _mm_loadu_si128(&v97);
        v61 = _mm_loadu_si128(&v101);
        a1[9] = _mm_loadu_si128(&v95);
        v62 = _mm_loadu_si128(&v105);
        v63 = _mm_loadu_si128(&v106);
        v64 = _mm_loadu_si128(&v98);
        v65 = _mm_loadu_si128(v107);
        a1[10] = v59;
        a1[11] = v60;
        v66 = _mm_loadu_si128(&v99);
        v67 = _mm_loadu_si128(&v100);
        a1[12] = v64;
        v68 = _mm_loadu_si128(&v102);
        a1[13] = v66;
        v69 = _mm_loadu_si128(&v103);
        a1[14] = v67;
        v70 = _mm_loadu_si128(&v104);
        a1[15] = v61;
        a1[16] = v68;
        a1[17] = v69;
        a1[18] = v70;
        a1[19] = v62;
        a1[20] = v63;
        a1[21] = v65;
        return result;
      }
      if ( v30 == 5 || v30 == 1 )
        goto LABEL_18;
      return result;
    }
    return 0;
  }
  v35 = *a6;
  if ( !*a6 || *(_BYTE *)(v35 + 174) != 1 )
    return 0;
  for ( i = *(_QWORD *)(v35 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v37 = **(_QWORD **)(i + 168);
  if ( !v37 || !(unsigned int)sub_8D3110(*(_QWORD *)(v37 + 8)) )
    return 0;
  v38 = _mm_loadu_si128(&v87);
  v39 = _mm_loadu_si128(&v88);
  v40 = _mm_loadu_si128(&v89);
  v41 = _mm_loadu_si128(&v90);
  v42 = _mm_loadu_si128(&v91);
  *a1 = _mm_loadu_si128(&v86);
  a1[1] = v38;
  v43 = _mm_loadu_si128(&v92);
  v44 = _mm_loadu_si128(&v93);
  a1[2] = v39;
  v45 = v87.m128i_i8[0];
  v46 = _mm_loadu_si128(&v94);
  a1[3] = v40;
  a1[4] = v41;
  a1[5] = v42;
  a1[6] = v43;
  a1[7] = v44;
  a1[8] = v46;
  if ( v45 == 2 )
  {
    v71 = _mm_loadu_si128(&v96);
    v72 = _mm_loadu_si128(&v97);
    v73 = _mm_loadu_si128(&v98);
    v74 = _mm_loadu_si128(&v99);
    v75 = _mm_loadu_si128(&v100);
    a1[9] = _mm_loadu_si128(&v95);
    v76 = _mm_loadu_si128(&v101);
    v77 = _mm_loadu_si128(&v106);
    a1[10] = v71;
    v78 = _mm_loadu_si128(&v102);
    v79 = _mm_loadu_si128(&v103);
    a1[11] = v72;
    a1[12] = v73;
    v80 = _mm_loadu_si128(&v104);
    v81 = _mm_loadu_si128(&v105);
    a1[13] = v74;
    v82 = _mm_loadu_si128(v107);
    a1[14] = v75;
    a1[15] = v76;
    a1[16] = v78;
    a1[17] = v79;
    a1[18] = v80;
    a1[19] = v81;
    a1[20] = v77;
    a1[21] = v82;
    return 1;
  }
  if ( v45 == 5 )
  {
    a1[9].m128i_i64[0] = v95.m128i_i64[0];
    return 1;
  }
  result = 1;
  if ( v45 == 1 )
LABEL_18:
    a1[9].m128i_i64[0] = v95.m128i_i64[0];
  return result;
}
