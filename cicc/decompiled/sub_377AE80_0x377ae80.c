// Function: sub_377AE80
// Address: 0x377ae80
//
void __fastcall sub_377AE80(__int64 *a1, unsigned __int64 a2, __int64 a3, __int64 a4, __m128i a5)
{
  __int64 *v6; // rax
  unsigned int v7; // ebx
  __int64 v8; // rsi
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 v11; // rsi
  __int16 *v12; // rax
  __int16 v13; // dx
  __int64 v14; // rax
  __int64 v15; // r9
  unsigned __int64 v16; // rdx
  __m128i *v17; // rcx
  __int64 v18; // r10
  __m128i *v19; // rsi
  __int64 *v20; // rax
  __int64 *v21; // rdx
  __int64 *i; // r10
  __int64 v23; // rax
  unsigned __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rbx
  __int64 v27; // r15
  __int64 v28; // rax
  unsigned __int64 v29; // rax
  unsigned __int64 *v30; // rax
  unsigned __int64 v31; // r14
  __int64 v32; // r9
  __int64 v33; // rax
  __int64 v34; // rax
  __int16 v35; // dx
  __int64 v36; // r8
  bool v37; // al
  unsigned int v38; // eax
  __int64 v39; // rsi
  __int64 v40; // r11
  __int64 v41; // r10
  int v42; // eax
  unsigned int *v43; // r10
  __int64 v44; // rdx
  __int64 v45; // rdx
  __int16 v46; // cx
  __m128i v47; // xmm1
  __m128i *v48; // rax
  _QWORD *v49; // r15
  int v50; // r9d
  unsigned int *v51; // rax
  __int64 v52; // rdx
  unsigned __int8 *v53; // rax
  unsigned __int64 v54; // rcx
  __int64 v55; // rbx
  int v56; // edx
  _QWORD *v57; // r15
  unsigned int *v58; // rax
  __int64 v59; // rdx
  int v60; // edx
  unsigned int v61; // edx
  __int128 v62; // [rsp-20h] [rbp-2A0h]
  __int128 v63; // [rsp-10h] [rbp-290h]
  __int64 v65; // [rsp+10h] [rbp-270h]
  __int16 v66; // [rsp+1Ch] [rbp-264h]
  __int16 v67; // [rsp+1Eh] [rbp-262h]
  __int64 v68; // [rsp+20h] [rbp-260h]
  __int64 v69; // [rsp+30h] [rbp-250h]
  __int16 v71; // [rsp+5Ah] [rbp-226h]
  __int64 v72; // [rsp+68h] [rbp-218h]
  __int64 v73; // [rsp+70h] [rbp-210h]
  __int128 *v74; // [rsp+70h] [rbp-210h]
  __int64 v75; // [rsp+78h] [rbp-208h]
  __int64 v76; // [rsp+78h] [rbp-208h]
  _QWORD *v77; // [rsp+78h] [rbp-208h]
  int v78; // [rsp+78h] [rbp-208h]
  int v79; // [rsp+78h] [rbp-208h]
  __int64 v80; // [rsp+80h] [rbp-200h]
  __m128i *v81; // [rsp+80h] [rbp-200h]
  __m128i *v82; // [rsp+80h] [rbp-200h]
  __int64 v83; // [rsp+80h] [rbp-200h]
  __int64 v84; // [rsp+88h] [rbp-1F8h]
  unsigned __int8 *v85; // [rsp+B0h] [rbp-1D0h]
  unsigned __int8 *v86; // [rsp+C0h] [rbp-1C0h]
  __int64 v87; // [rsp+100h] [rbp-180h] BYREF
  int v88; // [rsp+108h] [rbp-178h]
  __int64 v89; // [rsp+110h] [rbp-170h] BYREF
  __int64 v90; // [rsp+118h] [rbp-168h]
  __int64 v91; // [rsp+120h] [rbp-160h] BYREF
  __int64 v92; // [rsp+128h] [rbp-158h]
  __int16 v93; // [rsp+130h] [rbp-150h] BYREF
  __int64 v94; // [rsp+138h] [rbp-148h]
  __int64 v95; // [rsp+140h] [rbp-140h] BYREF
  int v96; // [rsp+148h] [rbp-138h]
  __m128i v97; // [rsp+150h] [rbp-130h] BYREF
  __m128i v98; // [rsp+160h] [rbp-120h] BYREF
  __int64 v99; // [rsp+170h] [rbp-110h] BYREF
  __int64 v100; // [rsp+178h] [rbp-108h]
  __int16 v101; // [rsp+180h] [rbp-100h]
  __int64 v102; // [rsp+188h] [rbp-F8h]
  __m128i v103; // [rsp+190h] [rbp-F0h] BYREF
  __m128i v104; // [rsp+1A0h] [rbp-E0h] BYREF
  __m128i *v105; // [rsp+1B0h] [rbp-D0h] BYREF
  __int64 v106; // [rsp+1B8h] [rbp-C8h]
  __int64 v107; // [rsp+1C0h] [rbp-C0h] BYREF
  int v108; // [rsp+1C8h] [rbp-B8h]
  __int64 *v109; // [rsp+200h] [rbp-80h] BYREF
  __int64 v110; // [rsp+208h] [rbp-78h]
  __int64 v111; // [rsp+210h] [rbp-70h] BYREF
  __int64 v112; // [rsp+218h] [rbp-68h]

  v6 = *(__int64 **)(a2 + 40);
  v7 = *(_DWORD *)(a2 + 64);
  v8 = *(_QWORD *)(a2 + 80);
  v9 = *v6;
  v10 = v6[1];
  v87 = v8;
  v69 = v10;
  if ( v8 )
    sub_B96E90((__int64)&v87, v8, 1);
  v11 = a1[1];
  v88 = *(_DWORD *)(a2 + 72);
  v12 = *(__int16 **)(a2 + 48);
  v13 = *v12;
  v14 = *((_QWORD *)v12 + 1);
  LOWORD(v105) = v13;
  v106 = v14;
  sub_33D0340((__int64)&v109, v11, (__int64 *)&v105);
  v16 = v7;
  v67 = (__int16)v109;
  v105 = (__m128i *)&v107;
  v68 = v110;
  v66 = v111;
  v65 = v112;
  v106 = 0x400000000LL;
  if ( !v7 )
  {
    v110 = 0x400000000LL;
    v107 = v9;
    v108 = v69;
    v111 = v9;
    LODWORD(v112) = v69;
    v48 = (__m128i *)&v107;
    v109 = &v111;
    goto LABEL_28;
  }
  v17 = (__m128i *)&v107;
  v18 = 16LL * v7;
  v19 = (__m128i *)((char *)&v107 + v18);
  if ( v7 > 4uLL )
  {
    sub_C8D5F0((__int64)&v105, &v107, v7, 0x10u, (__int64)&v109, v15);
    v18 = 16LL * v7;
    v16 = v7;
    v17 = &v105[(unsigned int)v106];
    v19 = (__m128i *)((char *)v105 + v18);
    if ( v17 == (__m128i *)&v105->m128i_i8[v18] )
    {
      LODWORD(v106) = v7;
      v109 = &v111;
      v110 = 0x400000000LL;
LABEL_38:
      v83 = v18;
      sub_C8D5F0((__int64)&v109, &v111, v16, 0x10u, (__int64)&v109, v15);
      v21 = v109;
      v18 = v83;
      v20 = &v109[2 * (unsigned int)v110];
      goto LABEL_10;
    }
  }
  do
  {
    if ( v17 )
    {
      v17->m128i_i64[0] = 0;
      v17->m128i_i32[2] = 0;
    }
    ++v17;
  }
  while ( v17 != v19 );
  v20 = &v111;
  LODWORD(v106) = v7;
  v109 = &v111;
  v110 = 0x400000000LL;
  if ( v16 > 4 )
    goto LABEL_38;
  v21 = &v111;
LABEL_10:
  for ( i = (__int64 *)((char *)v21 + v18); i != v20; v20 += 2 )
  {
    if ( v20 )
    {
      *v20 = 0;
      *((_DWORD *)v20 + 2) = 0;
    }
  }
  v23 = (__int64)v105;
  LODWORD(v110) = v7;
  v105->m128i_i64[0] = v9;
  *(_DWORD *)(v23 + 8) = v69;
  v24 = (unsigned __int64)v109;
  *v109 = v9;
  *(_DWORD *)(v24 + 8) = v69;
  if ( v7 != 1 )
  {
    v25 = v7 - 2;
    v26 = 2;
    v27 = 40;
    v80 = 40 * v25 + 80;
    while ( 1 )
    {
      v30 = (unsigned __int64 *)(v27 + *(_QWORD *)(a2 + 40));
      v31 = *v30;
      v32 = v30[1];
      v33 = *((unsigned int *)v30 + 2);
      v89 = v31;
      v34 = *(_QWORD *)(v31 + 48) + 16 * v33;
      v90 = v32;
      v91 = v31;
      v92 = v32;
      v35 = *(_WORD *)v34;
      v36 = *(_QWORD *)(v34 + 8);
      v93 = v35;
      v94 = v36;
      if ( v35 )
      {
        if ( (unsigned __int16)(v35 - 17) <= 0xD3u )
          goto LABEL_20;
      }
      else
      {
        v72 = v32;
        v73 = v36;
        v37 = sub_30070B0((__int64)&v93);
        v35 = 0;
        v36 = v73;
        v32 = v72;
        if ( v37 )
        {
LABEL_20:
          v75 = v32;
          HIWORD(v38) = v71;
          LOWORD(v38) = v35;
          sub_2FE6CC0((__int64)&v103, *a1, *(_QWORD *)(a1[1] + 64), v38, v36);
          if ( v103.m128i_i8[0] == 6 )
          {
            sub_375E8D0((__int64)a1, v31, v75, (__int64)&v89, (__int64)&v91);
          }
          else
          {
            v39 = *(_QWORD *)(a2 + 80);
            v40 = a1[1];
            v95 = v39;
            if ( v39 )
            {
              v76 = v40;
              sub_B96E90((__int64)&v95, v39, 1);
              v40 = v76;
            }
            v41 = *(_QWORD *)(a2 + 40);
            v42 = *(_DWORD *)(a2 + 72);
            v97.m128i_i64[1] = 0;
            v43 = (unsigned int *)(v27 + v41);
            v96 = v42;
            v98.m128i_i64[1] = 0;
            v44 = v43[2];
            v97.m128i_i16[0] = 0;
            v98.m128i_i16[0] = 0;
            v74 = (__int128 *)v43;
            v45 = *(_QWORD *)(*(_QWORD *)v43 + 48LL) + 16 * v44;
            v77 = (_QWORD *)v40;
            v46 = *(_WORD *)v45;
            v100 = *(_QWORD *)(v45 + 8);
            LOWORD(v99) = v46;
            sub_33D0340((__int64)&v103, v40, &v99);
            a5 = _mm_loadu_si128(&v103);
            v47 = _mm_loadu_si128(&v104);
            v97 = a5;
            v98 = v47;
            sub_3408290((__int64)&v103, v77, v74, (__int64)&v95, (unsigned int *)&v97, (unsigned int *)&v98, a5);
            if ( v95 )
              sub_B91220((__int64)&v95, v95);
            v89 = v103.m128i_i64[0];
            LODWORD(v90) = v103.m128i_i32[2];
            v91 = v104.m128i_i64[0];
            LODWORD(v92) = v104.m128i_i32[2];
          }
        }
      }
      v28 = (__int64)v105;
      v27 += 40;
      v105[(unsigned __int64)v26 / 2].m128i_i64[0] = v89;
      *(_DWORD *)(v28 + v26 * 8 + 8) = v90;
      v29 = (unsigned __int64)v109;
      v109[v26] = v91;
      *(_DWORD *)(v29 + v26 * 8 + 8) = v92;
      v26 += 2;
      if ( v27 == v80 )
      {
        v48 = v105;
        v16 = (unsigned int)v106;
        goto LABEL_28;
      }
    }
  }
  v48 = v105;
  v16 = (unsigned int)v106;
LABEL_28:
  v49 = (_QWORD *)a1[1];
  v84 = v16;
  v50 = *(_DWORD *)(a2 + 28);
  v101 = 1;
  v103.m128i_i64[1] = v65;
  LOWORD(v99) = v67;
  v104.m128i_i16[0] = 1;
  v100 = v68;
  v103.m128i_i16[0] = v66;
  v78 = v50;
  v102 = 0;
  v104.m128i_i64[1] = 0;
  v81 = v48;
  v51 = (unsigned int *)sub_33E5830(v49, (unsigned __int16 *)&v99, 2);
  v53 = sub_3410740(v49, *(unsigned int *)(a2 + 24), (__int64)&v87, v51, v52, v78, a5, v81, v84);
  v54 = (unsigned __int64)v109;
  v55 = (unsigned int)v110;
  *(_QWORD *)a3 = v53;
  v82 = (__m128i *)v54;
  *(_DWORD *)(a3 + 8) = v56;
  v57 = (_QWORD *)a1[1];
  v79 = *(_DWORD *)(a2 + 28);
  v58 = (unsigned int *)sub_33E5830(v57, (unsigned __int16 *)&v103, 2);
  v86 = sub_3410740(v57, *(unsigned int *)(a2 + 24), (__int64)&v87, v58, v59, v79, a5, v82, v55);
  *(_QWORD *)a4 = v86;
  *(_DWORD *)(a4 + 8) = v60;
  *((_QWORD *)&v63 + 1) = 1;
  *(_QWORD *)&v63 = v86;
  *((_QWORD *)&v62 + 1) = 1;
  *(_QWORD *)&v62 = *(_QWORD *)a3;
  v85 = sub_3406EB0((_QWORD *)a1[1], 2u, (__int64)&v87, 1, 0, 1, v62, v63);
  sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v85, v69 & 0xFFFFFFFF00000000LL | v61);
  if ( v109 != &v111 )
    _libc_free((unsigned __int64)v109);
  if ( v105 != (__m128i *)&v107 )
    _libc_free((unsigned __int64)v105);
  if ( v87 )
    sub_B91220((__int64)&v87, v87);
}
