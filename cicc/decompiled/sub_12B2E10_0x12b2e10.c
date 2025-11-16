// Function: sub_12B2E10
// Address: 0x12b2e10
//
_QWORD *__fastcall sub_12B2E10(__int64 a1, int a2, __int64 a3, __int64 *a4, _QWORD *a5)
{
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v12; // r15
  __int64 v13; // r14
  __int64 v14; // r13
  __int64 v15; // r14
  __m128i *v16; // r14
  __int64 v17; // rax
  int *v18; // rdx
  _BOOL4 v19; // r9d
  __int64 v20; // rax
  __m128i v21; // xmm1
  __int64 v22; // rax
  int v23; // edx
  __int64 v24; // [rsp+8h] [rbp-278h]
  __int64 v28; // [rsp+28h] [rbp-258h]
  int *v29; // [rsp+28h] [rbp-258h]
  __int64 v30; // [rsp+30h] [rbp-250h]
  _BOOL4 v31; // [rsp+30h] [rbp-250h]
  __int64 v32; // [rsp+38h] [rbp-248h]
  __int64 v33; // [rsp+40h] [rbp-240h] BYREF
  int v34; // [rsp+48h] [rbp-238h] BYREF
  __int64 v35; // [rsp+50h] [rbp-230h]
  int *v36; // [rsp+58h] [rbp-228h]
  int *v37; // [rsp+60h] [rbp-220h]
  __int64 v38; // [rsp+68h] [rbp-218h]
  int v39; // [rsp+70h] [rbp-210h] BYREF
  __int64 v40; // [rsp+78h] [rbp-208h]
  __int64 v41; // [rsp+80h] [rbp-200h]
  __int64 v42; // [rsp+88h] [rbp-1F8h]
  __int64 v43; // [rsp+90h] [rbp-1F0h]
  int v44; // [rsp+98h] [rbp-1E8h]
  __int64 v45; // [rsp+A0h] [rbp-1E0h]
  __int64 v46; // [rsp+A8h] [rbp-1D8h]
  __int64 v47; // [rsp+B0h] [rbp-1D0h]
  __int64 v48; // [rsp+B8h] [rbp-1C8h]
  int v49; // [rsp+C0h] [rbp-1C0h]
  __int64 v50; // [rsp+C8h] [rbp-1B8h]
  __int64 v51; // [rsp+D0h] [rbp-1B0h]
  __int64 v52; // [rsp+D8h] [rbp-1A8h]
  __int64 v53; // [rsp+E0h] [rbp-1A0h]
  int v54; // [rsp+E8h] [rbp-198h]
  __int64 v55; // [rsp+F0h] [rbp-190h]
  __int64 v56; // [rsp+F8h] [rbp-188h]
  __int64 v57; // [rsp+100h] [rbp-180h]
  __int64 v58; // [rsp+108h] [rbp-178h]
  int v59; // [rsp+110h] [rbp-170h]
  __int64 v60; // [rsp+118h] [rbp-168h]
  __int64 v61; // [rsp+120h] [rbp-160h]
  __int64 v62; // [rsp+128h] [rbp-158h]
  __int64 v63; // [rsp+130h] [rbp-150h]
  int v64; // [rsp+138h] [rbp-148h]
  __int64 v65; // [rsp+140h] [rbp-140h]
  __int64 v66; // [rsp+148h] [rbp-138h]
  __int64 v67; // [rsp+150h] [rbp-130h]
  __int64 v68; // [rsp+158h] [rbp-128h]
  int v69; // [rsp+160h] [rbp-120h]
  __int64 v70; // [rsp+168h] [rbp-118h]
  __int64 v71; // [rsp+170h] [rbp-110h]
  __int64 v72; // [rsp+178h] [rbp-108h]
  __int64 v73; // [rsp+180h] [rbp-100h]
  int v74; // [rsp+188h] [rbp-F8h]
  __int64 v75; // [rsp+190h] [rbp-F0h]
  __int64 v76; // [rsp+198h] [rbp-E8h]
  __int64 v77; // [rsp+1A0h] [rbp-E0h]
  __int64 v78; // [rsp+1A8h] [rbp-D8h]
  int v79; // [rsp+1B0h] [rbp-D0h]
  __int64 v80; // [rsp+1B8h] [rbp-C8h]
  __int64 v81; // [rsp+1C0h] [rbp-C0h]
  __int64 v82; // [rsp+1C8h] [rbp-B8h]
  __int64 v83; // [rsp+1D0h] [rbp-B0h]
  int v84; // [rsp+1D8h] [rbp-A8h]
  __int64 v85; // [rsp+1E0h] [rbp-A0h]
  __int64 v86; // [rsp+1E8h] [rbp-98h]
  __int64 v87; // [rsp+1F0h] [rbp-90h]
  __int64 v88; // [rsp+1F8h] [rbp-88h]
  int v89; // [rsp+200h] [rbp-80h]
  __int64 v90; // [rsp+208h] [rbp-78h]
  __int64 v91; // [rsp+210h] [rbp-70h]
  __int64 v92; // [rsp+218h] [rbp-68h]
  __int64 v93; // [rsp+220h] [rbp-60h]
  int v94; // [rsp+228h] [rbp-58h]
  __int64 v95; // [rsp+230h] [rbp-50h]
  __int64 v96; // [rsp+238h] [rbp-48h]
  __int64 v97; // [rsp+240h] [rbp-40h]
  __int64 v98; // [rsp+248h] [rbp-38h]
  char v99; // [rsp+250h] [rbp-30h] BYREF

  v32 = a1 + 688;
  if ( !*(_QWORD *)(a1 + 720) )
  {
    v24 = sub_16432A0(*(_QWORD *)(a1 + 360));
    v12 = sub_16432B0(*(_QWORD *)(a1 + 360));
    sub_1643360(*(_QWORD *)(a1 + 360));
    v13 = sub_1643350(*(_QWORD *)(a1 + 360));
    v28 = sub_16463B0(v12, 2);
    v30 = sub_16463B0(v13, 2);
    v14 = sub_16463B0(v13, 4);
    v15 = sub_16463B0(v13, 8);
    v43 = v12;
    v48 = v12;
    v58 = v14;
    v63 = v14;
    v39 = 753;
    v40 = 1;
    v41 = 9;
    v42 = 0;
    v44 = 754;
    v45 = 1;
    v46 = 9;
    v47 = 1;
    v49 = 755;
    v50 = 1;
    v51 = 9;
    v52 = 2;
    v53 = v28;
    v54 = 756;
    v55 = 25;
    v56 = 8;
    v57 = 0;
    v59 = 757;
    v60 = 25;
    v61 = 8;
    v62 = 1;
    v64 = 758;
    v65 = 25;
    v66 = 10;
    v67 = 2;
    v68 = sub_16463B0(v24, 8);
    v69 = 759;
    v70 = 23;
    v73 = v14;
    v78 = v14;
    v83 = v15;
    v98 = v15;
    v16 = (__m128i *)&v39;
    v71 = 7;
    v72 = 0;
    v74 = 760;
    v75 = 23;
    v76 = 7;
    v77 = 1;
    v79 = 761;
    v80 = 24;
    v81 = 7;
    v82 = 0;
    v84 = 762;
    v85 = 24;
    v86 = 7;
    v87 = 1;
    v88 = v30;
    v89 = 763;
    v90 = 6;
    v91 = 7;
    v92 = 0;
    v93 = v30;
    v94 = 764;
    v95 = 6;
    v96 = 7;
    v97 = 1;
    v34 = 0;
    v35 = 0;
    v36 = &v34;
    v37 = &v34;
    v38 = 0;
    do
    {
      v17 = sub_12B2D10(&v33, (__int64)&v34, v16->m128i_i32);
      if ( v18 )
      {
        v19 = v17 || v18 == &v34 || v16->m128i_i32[0] < v18[8];
        v29 = v18;
        v31 = v19;
        v20 = sub_22077B0(72);
        v21 = _mm_loadu_si128(v16 + 1);
        *(__m128i *)(v20 + 32) = _mm_loadu_si128(v16);
        *(__m128i *)(v20 + 48) = v21;
        *(_QWORD *)(v20 + 64) = v16[2].m128i_i64[0];
        sub_220F040(v31, v20, v29, &v34);
        ++v38;
      }
      v16 = (__m128i *)((char *)v16 + 40);
    }
    while ( v16 != (__m128i *)&v99 );
    sub_12A7830(*(_QWORD *)(a1 + 696));
    v22 = v35;
    *(_QWORD *)(a1 + 696) = 0;
    *(_QWORD *)(a1 + 720) = 0;
    *(_QWORD *)(a1 + 704) = v32;
    *(_QWORD *)(a1 + 712) = v32;
    if ( v22 )
    {
      v23 = v34;
      *(_QWORD *)(a1 + 696) = v22;
      *(_DWORD *)(a1 + 688) = v23;
      *(_QWORD *)(a1 + 704) = v36;
      *(_QWORD *)(a1 + 712) = v37;
      *(_QWORD *)(v22 + 8) = v32;
      v35 = 0;
      *(_QWORD *)(a1 + 720) = v38;
      v36 = &v34;
      v37 = &v34;
      v38 = 0;
    }
    sub_12A7830(0);
  }
  v6 = *(_QWORD *)(a1 + 696);
  if ( !v6 )
    goto LABEL_9;
  v7 = a1 + 688;
  do
  {
    while ( 1 )
    {
      v8 = *(_QWORD *)(v6 + 16);
      v9 = *(_QWORD *)(v6 + 24);
      if ( *(_DWORD *)(v6 + 32) >= a2 )
        break;
      v6 = *(_QWORD *)(v6 + 24);
      if ( !v9 )
        goto LABEL_7;
    }
    v7 = v6;
    v6 = *(_QWORD *)(v6 + 16);
  }
  while ( v8 );
LABEL_7:
  if ( v32 == v7 || a2 < *(_DWORD *)(v7 + 32) )
LABEL_9:
    sub_127B630("unexpected overloaded mma load intrinsic call!", 0);
  v10 = *(_QWORD *)(v7 + 64);
  *a4 = (*(_QWORD *)(v7 + 40) << 32) | a3 | (16LL * *(_QWORD *)(v7 + 48)) | (2LL * *(_QWORD *)(v7 + 56));
  *a5 = v10;
  return a5;
}
