// Function: sub_69BA40
// Address: 0x69ba40
//
__int64 __fastcall sub_69BA40(__int64 a1, __m128i *a2, __int64 a3, const __m128i *a4, __int64 a5, __int64 a6)
{
  int v6; // r15d
  int v7; // r14d
  unsigned __int8 v8; // r13
  unsigned __int64 v10; // rax
  __m128i *v11; // rbx
  __int64 v12; // rax
  unsigned int v13; // r14d
  __int64 result; // rax
  __m128i v15; // xmm1
  __m128i v16; // xmm2
  __m128i v17; // xmm3
  __m128i v18; // xmm4
  __m128i v19; // xmm5
  __m128i v20; // xmm6
  __int8 v21; // al
  __m128i v22; // xmm7
  __m128i v23; // xmm0
  __int64 v24; // rbx
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __m128i *v29; // rsi
  __m128i *v30; // r15
  unsigned __int16 v31; // ax
  __m128i v32; // xmm4
  __m128i v33; // xmm5
  __m128i v34; // xmm6
  __m128i v35; // xmm7
  __m128i v36; // xmm3
  __m128i v37; // xmm1
  __m128i v38; // xmm2
  __m128i v39; // xmm4
  __m128i v40; // xmm5
  __m128i v41; // xmm6
  __m128i v42; // xmm7
  __m128i v43; // xmm3
  __m128i v44; // xmm2
  __m128i v45; // xmm3
  __m128i v46; // xmm4
  __m128i v47; // xmm5
  __m128i v48; // xmm6
  __m128i v49; // xmm7
  __m128i v50; // xmm1
  __int8 v51; // al
  __m128i v52; // xmm2
  __int64 v53; // rax
  __m128i v54; // xmm5
  __m128i v55; // xmm6
  __m128i v56; // xmm7
  __m128i v57; // xmm1
  __m128i v58; // xmm2
  __m128i v59; // xmm4
  __m128i v60; // xmm5
  __m128i v61; // xmm6
  __m128i v62; // xmm7
  __m128i v63; // xmm1
  __m128i v64; // xmm2
  __m128i v65; // xmm4
  __int64 v66; // [rsp+18h] [rbp-2F8h] BYREF
  _BYTE v67[352]; // [rsp+20h] [rbp-2F0h] BYREF
  __m128i v68; // [rsp+180h] [rbp-190h] BYREF
  __m128i v69; // [rsp+190h] [rbp-180h]
  __m128i v70; // [rsp+1A0h] [rbp-170h]
  __m128i v71; // [rsp+1B0h] [rbp-160h]
  __m128i v72; // [rsp+1C0h] [rbp-150h] BYREF
  __m128i v73; // [rsp+1D0h] [rbp-140h]
  __m128i v74; // [rsp+1E0h] [rbp-130h]
  __m128i v75; // [rsp+1F0h] [rbp-120h]
  __m128i v76; // [rsp+200h] [rbp-110h]
  __m128i v77; // [rsp+210h] [rbp-100h]
  __m128i v78; // [rsp+220h] [rbp-F0h]
  __m128i v79; // [rsp+230h] [rbp-E0h]
  __m128i v80; // [rsp+240h] [rbp-D0h]
  __m128i v81; // [rsp+250h] [rbp-C0h]
  __m128i v82; // [rsp+260h] [rbp-B0h]
  __m128i v83; // [rsp+270h] [rbp-A0h]
  __m128i v84; // [rsp+280h] [rbp-90h]
  __m128i v85; // [rsp+290h] [rbp-80h]
  __m128i v86; // [rsp+2A0h] [rbp-70h]
  __m128i v87; // [rsp+2B0h] [rbp-60h]
  __m128i v88; // [rsp+2C0h] [rbp-50h]
  __m128i v89; // [rsp+2D0h] [rbp-40h]

  v6 = a5;
  v7 = a3;
  v8 = a1;
  v10 = ++qword_4D03A40;
  if ( a2 )
  {
    a1 = a2[4].m128i_i64[1];
    v11 = a2;
    a2 = &v68;
    v68.m128i_i64[0] = 0;
    v12 = sub_72B0F0(a1, &v68);
    if ( v12 && (*(_BYTE *)(v12 + 206) & 0x10) != 0 )
    {
LABEL_20:
      sub_6E6260(a4);
      v10 = qword_4D03A40;
      goto LABEL_10;
    }
    v11[1].m128i_i8[11] |= 2u;
    if ( v6 )
      v11[3].m128i_i8[12] |= 2u;
    v10 = qword_4D03A40;
  }
  if ( v10 > 0x64 )
  {
    if ( (unsigned int)sub_6E5430(a1, a2, a3, a4, a5, a6) )
    {
      sub_6851C0(0xBA6u, &a4[4].m128i_i32[1]);
      sub_6E6260(a4);
      v10 = qword_4D03A40;
      goto LABEL_10;
    }
    goto LABEL_20;
  }
  v13 = v7 + 1;
  if ( (unsigned __int8)(v8 - 30) > 1u )
  {
    v15 = _mm_loadu_si128(a4 + 1);
    v16 = _mm_loadu_si128(a4 + 2);
    v17 = _mm_loadu_si128(a4 + 3);
    v18 = _mm_loadu_si128(a4 + 4);
    v19 = _mm_loadu_si128(a4 + 5);
    v68 = _mm_loadu_si128(a4);
    v20 = _mm_loadu_si128(a4 + 6);
    v21 = a4[1].m128i_i8[0];
    v69 = v15;
    v22 = _mm_loadu_si128(a4 + 7);
    v70 = v16;
    v23 = _mm_loadu_si128(a4 + 8);
    v71 = v17;
    v72 = v18;
    v73 = v19;
    v74 = v20;
    v75 = v22;
    v76 = v23;
    if ( v21 == 2 )
    {
      v32 = _mm_loadu_si128(a4 + 10);
      v33 = _mm_loadu_si128(a4 + 11);
      v34 = _mm_loadu_si128(a4 + 12);
      v35 = _mm_loadu_si128(a4 + 13);
      v77 = _mm_loadu_si128(a4 + 9);
      v36 = _mm_loadu_si128(a4 + 14);
      v37 = _mm_loadu_si128(a4 + 19);
      v78 = v32;
      v38 = _mm_loadu_si128(a4 + 20);
      v39 = _mm_loadu_si128(a4 + 15);
      v79 = v33;
      v40 = _mm_loadu_si128(a4 + 16);
      v80 = v34;
      v41 = _mm_loadu_si128(a4 + 17);
      v81 = v35;
      v42 = _mm_loadu_si128(a4 + 18);
      v82 = v36;
      v43 = _mm_loadu_si128(a4 + 21);
      v83 = v39;
      v84 = v40;
      v85 = v41;
      v86 = v42;
      v87 = v37;
      v88 = v38;
      v89 = v43;
    }
    else if ( v21 == 5 || v21 == 1 )
    {
      v77.m128i_i64[0] = a4[9].m128i_i64[0];
    }
    v24 = sub_72BA30(5);
    v66 = sub_724DC0(5, a2, v25, v26, v27, v28);
    sub_72BB40(v24, v66);
    sub_6E6A50(v66, v67);
    sub_724E30(&v66);
    if ( v6 )
    {
      v30 = (__m128i *)v67;
      v29 = &v68;
    }
    else
    {
      v29 = (__m128i *)v67;
      v30 = &v68;
    }
    if ( v8 == 34 )
    {
      sub_68FEF0(v30, v29, &v72.m128i_i32[1], v13, 0, (__int64)a4);
    }
    else
    {
      v31 = sub_691DE0(v8);
      sub_69B310(v30->m128i_i64, v29->m128i_i64, v31, &v72.m128i_i32[1], v13, (__int64)a4);
    }
    sub_6E4BC0(&v68, a4);
    v10 = qword_4D03A40;
  }
  else if ( v8 == 31 )
  {
    v44 = _mm_loadu_si128(a4 + 1);
    v45 = _mm_loadu_si128(a4 + 2);
    v46 = _mm_loadu_si128(a4 + 3);
    v47 = _mm_loadu_si128(a4 + 4);
    v48 = _mm_loadu_si128(a4 + 5);
    v68 = _mm_loadu_si128(a4);
    v49 = _mm_loadu_si128(a4 + 6);
    v50 = _mm_loadu_si128(a4 + 7);
    v69 = v44;
    v51 = a4[1].m128i_i8[0];
    v70 = v45;
    v52 = _mm_loadu_si128(a4 + 8);
    v71 = v46;
    v72 = v47;
    v73 = v48;
    v74 = v49;
    v75 = v50;
    v76 = v52;
    if ( v51 == 2 )
    {
      v54 = _mm_loadu_si128(a4 + 10);
      v55 = _mm_loadu_si128(a4 + 11);
      v56 = _mm_loadu_si128(a4 + 12);
      v57 = _mm_loadu_si128(a4 + 13);
      v77 = _mm_loadu_si128(a4 + 9);
      v58 = _mm_loadu_si128(a4 + 14);
      v59 = _mm_loadu_si128(a4 + 15);
      v78 = v54;
      v60 = _mm_loadu_si128(a4 + 16);
      v79 = v55;
      v61 = _mm_loadu_si128(a4 + 17);
      v80 = v56;
      v62 = _mm_loadu_si128(a4 + 18);
      v81 = v57;
      v63 = _mm_loadu_si128(a4 + 19);
      v82 = v58;
      v64 = _mm_loadu_si128(a4 + 20);
      v83 = v59;
      v65 = _mm_loadu_si128(a4 + 21);
      v84 = v60;
      v85 = v61;
      v86 = v62;
      v87 = v63;
      v88 = v64;
      v89 = v65;
    }
    else if ( v51 == 5 || v51 == 1 )
    {
      v77.m128i_i64[0] = a4[9].m128i_i64[0];
    }
    v53 = sub_72C390();
    sub_6FEAC0(29, &v68, v53, a4, (char *)v72.m128i_i64 + 4, v13);
    sub_6E4BC0(&v68, a4);
    v10 = qword_4D03A40;
  }
LABEL_10:
  result = v10 - 1;
  qword_4D03A40 = result;
  return result;
}
