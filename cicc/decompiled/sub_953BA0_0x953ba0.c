// Function: sub_953BA0
// Address: 0x953ba0
//
_QWORD *__fastcall sub_953BA0(__int64 a1, int a2, char a3, unsigned __int64 *a4, _QWORD *a5, _QWORD *a6, _QWORD *a7)
{
  __int64 v7; // r12
  int v8; // ebx
  __int64 v9; // rax
  __int64 v10; // rcx
  __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // r8
  __int64 v14; // rdi
  __int64 v15; // rsi
  __int64 v17; // r14
  __int64 v18; // r13
  __int64 v19; // r15
  __m128i *v20; // r14
  __int64 v21; // rax
  int *v22; // rdx
  int *v23; // rbx
  _BOOL4 v24; // r12d
  __int64 v25; // rax
  __m128i v26; // xmm1
  __m128i v27; // xmm2
  __m128i v28; // xmm3
  __m128i v29; // xmm4
  __int64 v30; // rax
  int v31; // edx
  __int64 v32; // [rsp+8h] [rbp-288h]
  __int64 v33; // [rsp+10h] [rbp-280h]
  __int64 v34; // [rsp+18h] [rbp-278h]
  __int64 v35; // [rsp+20h] [rbp-270h]
  __int64 v40; // [rsp+48h] [rbp-248h]
  __int64 v41; // [rsp+50h] [rbp-240h] BYREF
  int v42; // [rsp+58h] [rbp-238h] BYREF
  __int64 v43; // [rsp+60h] [rbp-230h]
  int *v44; // [rsp+68h] [rbp-228h]
  int *v45; // [rsp+70h] [rbp-220h]
  __int64 v46; // [rsp+78h] [rbp-218h]
  int v47; // [rsp+80h] [rbp-210h] BYREF
  __int64 v48; // [rsp+88h] [rbp-208h]
  __int64 v49; // [rsp+90h] [rbp-200h]
  __int64 v50; // [rsp+98h] [rbp-1F8h]
  __int64 v51; // [rsp+A0h] [rbp-1F0h]
  __int64 v52; // [rsp+A8h] [rbp-1E8h]
  int v53; // [rsp+B0h] [rbp-1E0h]
  __int64 v54; // [rsp+B8h] [rbp-1D8h]
  __int64 v55; // [rsp+C0h] [rbp-1D0h]
  __int64 v56; // [rsp+C8h] [rbp-1C8h]
  int v57; // [rsp+D0h] [rbp-1C0h]
  __int64 v58; // [rsp+D8h] [rbp-1B8h]
  __int64 v59; // [rsp+E0h] [rbp-1B0h]
  __int64 v60; // [rsp+E8h] [rbp-1A8h]
  __int64 v61; // [rsp+F0h] [rbp-1A0h]
  __int64 v62; // [rsp+F8h] [rbp-198h]
  int v63; // [rsp+100h] [rbp-190h]
  __int64 v64; // [rsp+108h] [rbp-188h]
  __int64 v65; // [rsp+110h] [rbp-180h]
  __int64 v66; // [rsp+118h] [rbp-178h]
  int v67; // [rsp+120h] [rbp-170h]
  __int64 v68; // [rsp+128h] [rbp-168h]
  __int64 v69; // [rsp+130h] [rbp-160h]
  __int64 v70; // [rsp+138h] [rbp-158h]
  __int64 v71; // [rsp+140h] [rbp-150h]
  __int64 v72; // [rsp+148h] [rbp-148h]
  int v73; // [rsp+150h] [rbp-140h]
  __int64 v74; // [rsp+158h] [rbp-138h]
  __int64 v75; // [rsp+160h] [rbp-130h]
  __int64 v76; // [rsp+168h] [rbp-128h]
  int v77; // [rsp+170h] [rbp-120h]
  __int64 v78; // [rsp+178h] [rbp-118h]
  __int64 v79; // [rsp+180h] [rbp-110h]
  __int64 v80; // [rsp+188h] [rbp-108h]
  __int64 v81; // [rsp+190h] [rbp-100h]
  __int64 v82; // [rsp+198h] [rbp-F8h]
  int v83; // [rsp+1A0h] [rbp-F0h]
  __int64 v84; // [rsp+1A8h] [rbp-E8h]
  __int64 v85; // [rsp+1B0h] [rbp-E0h]
  __int64 v86; // [rsp+1B8h] [rbp-D8h]
  int v87; // [rsp+1C0h] [rbp-D0h]
  __int64 v88; // [rsp+1C8h] [rbp-C8h]
  __int64 v89; // [rsp+1D0h] [rbp-C0h]
  __int64 v90; // [rsp+1D8h] [rbp-B8h]
  __int64 v91; // [rsp+1E0h] [rbp-B0h]
  __int64 v92; // [rsp+1E8h] [rbp-A8h]
  int v93; // [rsp+1F0h] [rbp-A0h]
  __int64 v94; // [rsp+1F8h] [rbp-98h]
  __int64 v95; // [rsp+200h] [rbp-90h]
  __int64 v96; // [rsp+208h] [rbp-88h]
  int v97; // [rsp+210h] [rbp-80h]
  __int64 v98; // [rsp+218h] [rbp-78h]
  __int64 v99; // [rsp+220h] [rbp-70h]
  __int64 v100; // [rsp+228h] [rbp-68h]
  __int64 v101; // [rsp+230h] [rbp-60h]
  __int64 v102; // [rsp+238h] [rbp-58h]
  int v103; // [rsp+240h] [rbp-50h]
  __int64 v104; // [rsp+248h] [rbp-48h]
  __int64 v105; // [rsp+250h] [rbp-40h]
  __int64 v106; // [rsp+258h] [rbp-38h]
  char v107; // [rsp+260h] [rbp-30h] BYREF

  v7 = a1;
  v8 = a2;
  v40 = a1 + 560;
  if ( !*(_QWORD *)(a1 + 592) )
  {
    v32 = sub_BCB160(*(_QWORD *)(a1 + 344));
    v17 = sub_BCB170(*(_QWORD *)(a1 + 344));
    sub_BCB2E0(*(_QWORD *)(a1 + 344));
    v18 = sub_BCB2D0(*(_QWORD *)(a1 + 344));
    v33 = sub_BCDA70(v17, 2);
    v34 = sub_BCDA70(v18, 2);
    v19 = sub_BCDA70(v18, 4);
    v35 = sub_BCDA70(v18, 8);
    v54 = v18;
    v55 = v18;
    v64 = v17;
    v65 = v17;
    v20 = (__m128i *)&v47;
    v74 = v19;
    v75 = v19;
    v47 = 745;
    v48 = 0;
    v49 = 1;
    v50 = 5;
    v51 = 1;
    v52 = 1;
    v53 = 0;
    v56 = v34;
    v57 = 746;
    v58 = 1;
    v59 = 0;
    v60 = 1;
    v61 = 9;
    v62 = 9;
    v63 = 0;
    v66 = v33;
    v67 = 747;
    v68 = 0;
    v69 = 0;
    v70 = 25;
    v71 = 8;
    v72 = 8;
    v73 = 0;
    v76 = sub_BCDA70(v32, 8);
    v77 = 748;
    v84 = v19;
    v85 = v19;
    v78 = 0;
    v79 = 0;
    v80 = 23;
    v81 = 7;
    v82 = 7;
    v83 = 0;
    v86 = v76;
    v87 = 749;
    v88 = 0;
    v89 = 0;
    v90 = 24;
    v91 = 7;
    v92 = 7;
    v93 = 0;
    v94 = v35;
    v95 = v34;
    v96 = v76;
    v97 = 750;
    v98 = 0;
    v99 = 0;
    v100 = 6;
    v101 = 7;
    v102 = 7;
    v103 = 0;
    v104 = v34;
    v105 = v35;
    v106 = v76;
    v42 = 0;
    v43 = 0;
    v44 = &v42;
    v45 = &v42;
    v46 = 0;
    do
    {
      v21 = sub_953AA0(&v41, (__int64)&v42, v20->m128i_i32);
      v23 = v22;
      if ( v22 )
      {
        v24 = v21 || v22 == &v42 || v20->m128i_i32[0] < v22[8];
        v25 = sub_22077B0(112);
        v26 = _mm_loadu_si128(v20 + 1);
        v27 = _mm_loadu_si128(v20 + 2);
        v28 = _mm_loadu_si128(v20 + 3);
        v29 = _mm_loadu_si128(v20 + 4);
        *(__m128i *)(v25 + 32) = _mm_loadu_si128(v20);
        *(__m128i *)(v25 + 48) = v26;
        *(__m128i *)(v25 + 64) = v27;
        *(__m128i *)(v25 + 80) = v28;
        *(__m128i *)(v25 + 96) = v29;
        sub_220F040(v24, v25, v23, &v42);
        ++v46;
      }
      v20 += 5;
    }
    while ( v20 != (__m128i *)&v107 );
    v7 = a1;
    v8 = a2;
    sub_948E80(*(_QWORD *)(a1 + 568));
    v30 = v43;
    *(_QWORD *)(a1 + 568) = 0;
    *(_QWORD *)(a1 + 592) = 0;
    *(_QWORD *)(a1 + 576) = v40;
    *(_QWORD *)(a1 + 584) = v40;
    if ( v30 )
    {
      v31 = v42;
      *(_QWORD *)(a1 + 568) = v30;
      *(_DWORD *)(a1 + 560) = v31;
      *(_QWORD *)(a1 + 576) = v44;
      *(_QWORD *)(a1 + 584) = v45;
      *(_QWORD *)(v30 + 8) = v40;
      v43 = 0;
      *(_QWORD *)(a1 + 592) = v46;
      v44 = &v42;
      v45 = &v42;
      v46 = 0;
    }
    sub_948E80(0);
  }
  v9 = *(_QWORD *)(v7 + 568);
  if ( !v9 )
    goto LABEL_9;
  v10 = a1 + 560;
  do
  {
    while ( 1 )
    {
      v11 = *(_QWORD *)(v9 + 16);
      v12 = *(_QWORD *)(v9 + 24);
      if ( *(_DWORD *)(v9 + 32) >= v8 )
        break;
      v9 = *(_QWORD *)(v9 + 24);
      if ( !v12 )
        goto LABEL_7;
    }
    v10 = v9;
    v9 = *(_QWORD *)(v9 + 16);
  }
  while ( v11 );
LABEL_7:
  if ( v10 == v40 || *(_DWORD *)(v10 + 32) > v8 )
LABEL_9:
    sub_91B980("unexpected overloaded mma intrinsic call!", 0);
  v13 = *(_QWORD *)(v10 + 88);
  v14 = *(_QWORD *)(v10 + 96);
  v15 = *(_QWORD *)(v10 + 104);
  *a4 = ((unsigned __int64)*(unsigned int *)(v10 + 80) << 28)
      | (*(_QWORD *)(v10 + 56) << 32)
      | (*(_QWORD *)(v10 + 72) << 16)
      | (*(_QWORD *)(v10 + 64) << 8)
      | (16LL * *(_QWORD *)(v10 + 48))
      | *(_QWORD *)(v10 + 40)
      | ((-(__int64)((a3 & 2) == 0) & 0xFFFFFFFFFF000000LL) + 0x2000000)
      | (((a3 & 1) + 1LL) << 26);
  *a5 = v13;
  *a6 = v14;
  *a7 = v15;
  return a7;
}
