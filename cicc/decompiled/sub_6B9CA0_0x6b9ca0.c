// Function: sub_6B9CA0
// Address: 0x6b9ca0
//
void __fastcall sub_6B9CA0(int a1, int a2, __m128i *a3, __m128i *a4, _DWORD *a5)
{
  __m128i *v5; // r12
  int v7; // edx
  int v8; // edi
  int v9; // r15d
  __int64 v10; // rax
  __m128i *v11; // rsi
  __m128i *v12; // rdi
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
  __m128i v26; // xmm2
  __m128i v27; // xmm3
  __m128i v28; // xmm4
  __m128i v29; // xmm5
  __m128i v30; // xmm6
  __m128i v31; // xmm7
  __m128i v32; // xmm1
  __int8 v33; // al
  __m128i v34; // xmm2
  __m128i v35; // xmm4
  __m128i v36; // xmm5
  __m128i v37; // xmm6
  __m128i v38; // xmm7
  __m128i v39; // xmm3
  __m128i v40; // xmm4
  __m128i v41; // xmm0
  __m128i v42; // xmm5
  __m128i v43; // xmm6
  __m128i v44; // xmm7
  __m128i v45; // xmm3
  __m128i v46; // xmm4
  __m128i v47; // xmm6
  __m128i v48; // xmm7
  __m128i v49; // xmm1
  __m128i v50; // xmm2
  __m128i v51; // xmm0
  __m128i v52; // xmm3
  __m128i v53; // xmm4
  __m128i v54; // xmm5
  __m128i v55; // xmm6
  __m128i v56; // xmm1
  __m128i v57; // xmm7
  __m128i v58; // xmm2
  unsigned int v61; // [rsp+18h] [rbp-308h]
  int v62; // [rsp+1Ch] [rbp-304h]
  __int64 v63; // [rsp+28h] [rbp-2F8h] BYREF
  __m128i v64; // [rsp+30h] [rbp-2F0h] BYREF
  __m128i v65; // [rsp+40h] [rbp-2E0h] BYREF
  __m128i v66; // [rsp+50h] [rbp-2D0h] BYREF
  __m128i v67; // [rsp+60h] [rbp-2C0h] BYREF
  __m128i v68; // [rsp+70h] [rbp-2B0h] BYREF
  __m128i v69; // [rsp+80h] [rbp-2A0h] BYREF
  __m128i v70; // [rsp+90h] [rbp-290h] BYREF
  __m128i v71; // [rsp+A0h] [rbp-280h] BYREF
  __m128i v72; // [rsp+B0h] [rbp-270h] BYREF
  __m128i v73; // [rsp+C0h] [rbp-260h] BYREF
  __m128i v74; // [rsp+D0h] [rbp-250h] BYREF
  __m128i v75; // [rsp+E0h] [rbp-240h] BYREF
  __m128i v76; // [rsp+F0h] [rbp-230h] BYREF
  __m128i v77; // [rsp+100h] [rbp-220h] BYREF
  __m128i v78; // [rsp+110h] [rbp-210h] BYREF
  __m128i v79; // [rsp+120h] [rbp-200h] BYREF
  __m128i v80; // [rsp+130h] [rbp-1F0h] BYREF
  __m128i v81; // [rsp+140h] [rbp-1E0h] BYREF
  __m128i v82; // [rsp+150h] [rbp-1D0h] BYREF
  __m128i v83; // [rsp+160h] [rbp-1C0h] BYREF
  __m128i v84; // [rsp+170h] [rbp-1B0h] BYREF
  __m128i v85; // [rsp+180h] [rbp-1A0h] BYREF
  __m128i v86; // [rsp+190h] [rbp-190h] BYREF
  __m128i v87; // [rsp+1A0h] [rbp-180h] BYREF
  __m128i v88; // [rsp+1B0h] [rbp-170h] BYREF
  __m128i v89; // [rsp+1C0h] [rbp-160h] BYREF
  __m128i v90; // [rsp+1D0h] [rbp-150h] BYREF
  __m128i v91; // [rsp+1E0h] [rbp-140h] BYREF
  __m128i v92; // [rsp+1F0h] [rbp-130h] BYREF
  __m128i v93; // [rsp+200h] [rbp-120h] BYREF
  __m128i v94; // [rsp+210h] [rbp-110h] BYREF
  __m128i v95; // [rsp+220h] [rbp-100h] BYREF
  __m128i v96; // [rsp+230h] [rbp-F0h] BYREF
  __m128i v97; // [rsp+240h] [rbp-E0h] BYREF
  __m128i v98; // [rsp+250h] [rbp-D0h] BYREF
  __m128i v99; // [rsp+260h] [rbp-C0h] BYREF
  __m128i v100; // [rsp+270h] [rbp-B0h] BYREF
  __m128i v101; // [rsp+280h] [rbp-A0h] BYREF
  __m128i v102; // [rsp+290h] [rbp-90h] BYREF
  __m128i v103; // [rsp+2A0h] [rbp-80h] BYREF
  __m128i v104; // [rsp+2B0h] [rbp-70h] BYREF
  __m128i v105; // [rsp+2C0h] [rbp-60h] BYREF
  __m128i v106; // [rsp+2D0h] [rbp-50h] BYREF
  __m128i v107[4]; // [rsp+2E0h] [rbp-40h] BYREF

  v5 = a4;
  v7 = a1 | 1;
  *a5 = 1;
  v61 = a1 | 1;
  if ( dword_4F077C4 == 2 )
  {
    v8 = a1 | 0x11;
    if ( !a4 )
      v8 = v7;
    v61 = v8;
  }
  while ( !(unsigned int)sub_869470(&v63) )
  {
    if ( !(unsigned int)sub_7BE800(67) )
      goto LABEL_31;
  }
  v62 = 0;
  v9 = 1;
  do
  {
    if ( a2 && dword_4D04428 && word_4F06418[0] == 73 )
    {
      v5 = 0;
      v11 = &v64;
      v12 = (__m128i *)sub_6BA760(0, 0);
      sub_6E9FE0(v12, &v64);
    }
    else
    {
      v11 = &v86;
      v12 = &v64;
      sub_69ED20((__int64)&v64, &v86, 0, v61);
      if ( (v65.m128i_i8[2] & 1) == 0 )
        v5 = 0;
    }
    if ( v9 )
    {
      v17 = _mm_loadu_si128(&v65);
      v18 = _mm_loadu_si128(&v66);
      v19 = _mm_loadu_si128(&v67);
      v20 = _mm_loadu_si128(&v68);
      v21 = _mm_loadu_si128(&v69);
      *a3 = _mm_loadu_si128(&v64);
      v22 = _mm_loadu_si128(&v70);
      v23 = _mm_loadu_si128(&v71);
      a3[1] = v17;
      v24 = _mm_loadu_si128(&v72);
      v25 = v65.m128i_i8[0];
      a3[2] = v18;
      a3[3] = v19;
      a3[4] = v20;
      a3[5] = v21;
      a3[6] = v22;
      a3[7] = v23;
      a3[8] = v24;
      if ( v25 == 2 )
      {
        v35 = _mm_loadu_si128(&v74);
        v36 = _mm_loadu_si128(&v75);
        v37 = _mm_loadu_si128(&v76);
        v38 = _mm_loadu_si128(&v77);
        a3[9] = _mm_loadu_si128(&v73);
        v39 = _mm_loadu_si128(&v78);
        a3[10] = v35;
        v40 = _mm_loadu_si128(&v79);
        v41 = _mm_loadu_si128(&v83);
        a3[11] = v36;
        v42 = _mm_loadu_si128(&v80);
        a3[12] = v37;
        v43 = _mm_loadu_si128(&v81);
        a3[13] = v38;
        v44 = _mm_loadu_si128(&v82);
        a3[14] = v39;
        v45 = _mm_loadu_si128(&v84);
        a3[15] = v40;
        v46 = _mm_loadu_si128(&v85);
        a3[16] = v42;
        a3[17] = v43;
        a3[18] = v44;
        a3[19] = v41;
        a3[20] = v45;
        a3[21] = v46;
      }
      else if ( v25 == 5 || v25 == 1 )
      {
        a3[9].m128i_i64[0] = v73.m128i_i64[0];
      }
      if ( v5 )
      {
        v26 = _mm_loadu_si128(&v87);
        v27 = _mm_loadu_si128(&v88);
        v28 = _mm_loadu_si128(&v89);
        v29 = _mm_loadu_si128(&v90);
        v30 = _mm_loadu_si128(&v91);
        *v5 = _mm_loadu_si128(&v86);
        v31 = _mm_loadu_si128(&v92);
        v32 = _mm_loadu_si128(&v93);
        v5[1] = v26;
        v33 = v87.m128i_i8[0];
        v5[2] = v27;
        v34 = _mm_loadu_si128(&v94);
        v5[3] = v28;
        v5[4] = v29;
        v5[5] = v30;
        v5[6] = v31;
        v5[7] = v32;
        v5[8] = v34;
        if ( v33 == 2 )
        {
          v47 = _mm_loadu_si128(&v96);
          v48 = _mm_loadu_si128(&v97);
          v49 = _mm_loadu_si128(&v98);
          v50 = _mm_loadu_si128(&v99);
          v51 = _mm_loadu_si128(&v105);
          v5[9] = _mm_loadu_si128(&v95);
          v52 = _mm_loadu_si128(&v106);
          v53 = _mm_loadu_si128(v107);
          v5[10] = v47;
          v54 = _mm_loadu_si128(&v100);
          v55 = _mm_loadu_si128(&v101);
          v5[11] = v48;
          v5[12] = v49;
          v56 = _mm_loadu_si128(&v103);
          v57 = _mm_loadu_si128(&v102);
          v5[13] = v50;
          v58 = _mm_loadu_si128(&v104);
          v5[14] = v54;
          v5[15] = v55;
          v5[16] = v57;
          v5[17] = v56;
          v5[18] = v58;
          v5[19] = v51;
          v5[20] = v52;
          v5[21] = v53;
        }
        else if ( v33 == 5 || v33 == 1 )
        {
          v5[9].m128i_i64[0] = v95.m128i_i64[0];
        }
      }
      *a5 = 0;
    }
    else
    {
      if ( !v62 && (unsigned int)sub_6E5430(v12, v11, v13, v14, v15, v16) )
        sub_6851C0(0x832u, &v68.m128i_i32[1]);
      sub_6E6450(&v64);
      v62 = 1;
    }
    v10 = sub_867630(v63, 0);
    if ( v10 )
      a3[8].m128i_i64[0] = v10;
    v9 = 0;
  }
  while ( (unsigned int)sub_866C00(v63) );
LABEL_31:
  sub_690C20();
}
