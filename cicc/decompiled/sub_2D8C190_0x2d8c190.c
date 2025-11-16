// Function: sub_2D8C190
// Address: 0x2d8c190
//
__int64 __fastcall sub_2D8C190(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  unsigned int v8; // eax
  __int64 (*v9)(); // rax
  __int64 v10; // rdi
  __int64 (*v11)(); // rdx
  __int64 v12; // rax
  _QWORD *v13; // rax
  _QWORD *v14; // r12
  __int64 v15; // r10
  _QWORD *v16; // rdx
  _QWORD *v17; // rax
  _QWORD *v18; // rax
  unsigned __int64 v19; // rdi
  __int64 *v20; // rax
  __int64 *v21; // r15
  __int64 *v22; // r12
  __int64 v23; // rax
  __int64 v24; // rsi
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rdi
  int v28; // r10d
  unsigned int i; // eax
  __int64 v30; // r8
  unsigned int v31; // eax
  __int64 v32; // r15
  __int64 *v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rsi
  int v36; // r10d
  unsigned int j; // eax
  __int64 v38; // rdi
  unsigned int v39; // eax
  __int64 v40; // rax
  char v41; // al
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // rdx
  __int64 v48; // rcx
  __int64 v49; // [rsp+20h] [rbp-530h] BYREF
  unsigned __int64 v50; // [rsp+28h] [rbp-528h]
  __int64 v51; // [rsp+30h] [rbp-520h] BYREF
  unsigned int v52; // [rsp+38h] [rbp-518h]
  char v53; // [rsp+3Ch] [rbp-514h]
  _BYTE v54[16]; // [rsp+40h] [rbp-510h] BYREF
  __int64 v55; // [rsp+50h] [rbp-500h] BYREF
  _BYTE *v56; // [rsp+58h] [rbp-4F8h]
  __int64 v57; // [rsp+60h] [rbp-4F0h]
  int v58; // [rsp+68h] [rbp-4E8h]
  char v59; // [rsp+6Ch] [rbp-4E4h]
  _BYTE v60[64]; // [rsp+70h] [rbp-4E0h] BYREF
  __int64 v61; // [rsp+B0h] [rbp-4A0h] BYREF
  __int64 v62; // [rsp+B8h] [rbp-498h]
  __int64 v63; // [rsp+C0h] [rbp-490h]
  __int64 v64; // [rsp+C8h] [rbp-488h]
  __int64 v65; // [rsp+D0h] [rbp-480h]
  __int64 v66; // [rsp+D8h] [rbp-478h]
  __int64 v67; // [rsp+E0h] [rbp-470h]
  __int64 v68; // [rsp+E8h] [rbp-468h]
  __int64 *v69; // [rsp+F0h] [rbp-460h]
  _QWORD *v70; // [rsp+F8h] [rbp-458h]
  __int64 v71; // [rsp+100h] [rbp-450h]
  __int64 v72; // [rsp+108h] [rbp-448h]
  __int16 v73; // [rsp+110h] [rbp-440h]
  __int64 v74; // [rsp+118h] [rbp-438h] BYREF
  __int64 v75; // [rsp+120h] [rbp-430h]
  __int64 v76; // [rsp+128h] [rbp-428h]
  unsigned int v77; // [rsp+130h] [rbp-420h]
  char v78; // [rsp+158h] [rbp-3F8h]
  __int64 v79; // [rsp+168h] [rbp-3E8h]
  char *v80; // [rsp+170h] [rbp-3E0h]
  __int64 v81; // [rsp+178h] [rbp-3D8h]
  int v82; // [rsp+180h] [rbp-3D0h]
  char v83; // [rsp+184h] [rbp-3CCh]
  char v84; // [rsp+188h] [rbp-3C8h] BYREF
  __int64 v85; // [rsp+208h] [rbp-348h]
  __int64 v86; // [rsp+210h] [rbp-340h]
  __int64 v87; // [rsp+218h] [rbp-338h]
  int v88; // [rsp+220h] [rbp-330h]
  __int64 v89; // [rsp+228h] [rbp-328h]
  char *v90; // [rsp+230h] [rbp-320h]
  __int64 v91; // [rsp+238h] [rbp-318h]
  int v92; // [rsp+240h] [rbp-310h]
  char v93; // [rsp+244h] [rbp-30Ch]
  char v94; // [rsp+248h] [rbp-308h] BYREF
  __int64 v95; // [rsp+2C8h] [rbp-288h]
  __int64 v96; // [rsp+2D0h] [rbp-280h]
  __int64 v97; // [rsp+2D8h] [rbp-278h]
  int v98; // [rsp+2E0h] [rbp-270h]
  __int64 v99; // [rsp+2E8h] [rbp-268h]
  __int64 v100; // [rsp+2F0h] [rbp-260h]
  __int64 v101; // [rsp+2F8h] [rbp-258h]
  int v102; // [rsp+300h] [rbp-250h]
  _QWORD *v103; // [rsp+308h] [rbp-248h]
  __int64 v104; // [rsp+310h] [rbp-240h]
  _QWORD v105[2]; // [rsp+318h] [rbp-238h] BYREF
  char v106; // [rsp+328h] [rbp-228h] BYREF
  int v107; // [rsp+360h] [rbp-1F0h] BYREF
  __int64 v108; // [rsp+368h] [rbp-1E8h]
  int *v109; // [rsp+370h] [rbp-1E0h]
  int *v110; // [rsp+378h] [rbp-1D8h]
  __int64 v111; // [rsp+380h] [rbp-1D0h]
  __int64 v112; // [rsp+388h] [rbp-1C8h]
  __int64 v113; // [rsp+390h] [rbp-1C0h]
  __int64 v114; // [rsp+398h] [rbp-1B8h]
  int v115; // [rsp+3A0h] [rbp-1B0h]
  __int64 v116; // [rsp+3A8h] [rbp-1A8h]
  __int64 v117; // [rsp+3B0h] [rbp-1A0h]
  __int64 v118; // [rsp+3B8h] [rbp-198h]
  int v119; // [rsp+3C0h] [rbp-190h]
  char *v120; // [rsp+3C8h] [rbp-188h]
  __int64 v121; // [rsp+3D0h] [rbp-180h]
  char v122; // [rsp+3D8h] [rbp-178h] BYREF
  __int64 v123; // [rsp+3E0h] [rbp-170h]
  __int64 v124; // [rsp+3E8h] [rbp-168h]
  char v125; // [rsp+3F0h] [rbp-160h]
  __int64 v126; // [rsp+3F8h] [rbp-158h]
  char *v127; // [rsp+400h] [rbp-150h]
  __int64 v128; // [rsp+408h] [rbp-148h]
  int v129; // [rsp+410h] [rbp-140h]
  char v130; // [rsp+414h] [rbp-13Ch]
  char v131; // [rsp+418h] [rbp-138h] BYREF

  v7 = *a2;
  v62 = 0;
  v63 = 0;
  v61 = v7;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  v67 = 0;
  v68 = 0;
  v69 = 0;
  v70 = 0;
  v71 = 0;
  v72 = 0;
  v73 = 0;
  v74 = 0;
  v8 = sub_AF1560(0x56u);
  v77 = v8;
  if ( v8 )
  {
    v75 = sub_C7D670((unsigned __int64)v8 << 6, 8);
    sub_2D69A40((__int64)&v74);
  }
  else
  {
    v75 = 0;
    v76 = 0;
  }
  v78 = 0;
  v80 = &v84;
  v90 = &v94;
  v103 = v105;
  v105[0] = &v106;
  v105[1] = 0x200000000LL;
  v109 = &v107;
  v110 = &v107;
  v79 = 0;
  v81 = 16;
  v82 = 0;
  v83 = 1;
  v85 = 0;
  v86 = 0;
  v87 = 0;
  v88 = 0;
  v89 = 0;
  v91 = 16;
  v92 = 0;
  v93 = 1;
  v95 = 0;
  v96 = 0;
  v97 = 0;
  v98 = 0;
  v99 = 0;
  v100 = 0;
  v101 = 0;
  v102 = 0;
  v104 = 0;
  v107 = 0;
  v108 = 0;
  v111 = 0;
  v112 = 0;
  v120 = &v122;
  v113 = 0;
  v114 = 0;
  v115 = 0;
  v116 = 0;
  v117 = 0;
  v118 = 0;
  v119 = 0;
  v121 = 0;
  v123 = 0;
  v124 = 0;
  v125 = 0;
  v126 = 0;
  v127 = &v131;
  v128 = 32;
  v129 = 0;
  v130 = 1;
  v123 = sub_B2BEC0(a3);
  v9 = *(__int64 (**)())(*(_QWORD *)v61 + 16LL);
  if ( v9 == sub_23CE270 )
  {
    v62 = 0;
    BUG();
  }
  v62 = ((__int64 (__fastcall *)(__int64, __int64))v9)(v61, a3);
  v10 = v62;
  v11 = *(__int64 (**)())(*(_QWORD *)v62 + 144LL);
  v12 = 0;
  if ( v11 != sub_2C8F680 )
  {
    v12 = ((__int64 (__fastcall *)(__int64))v11)(v62);
    v10 = v62;
  }
  v63 = v12;
  v64 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v10 + 200LL))(v10);
  v67 = sub_BC1CD0(a4, &unk_4F6D3F8, a3) + 8;
  v65 = sub_BC1CD0(a4, &unk_4F89C30, a3) + 8;
  v68 = sub_BC1CD0(a4, &unk_4F875F0, a3) + 8;
  v13 = (_QWORD *)sub_22077B0(0x118u);
  v14 = v13;
  if ( v13 )
  {
    v15 = v68;
    *v13 = 0;
    v16 = v13 + 21;
    v17 = v13 + 13;
    *(v17 - 12) = 0;
    *(v17 - 11) = 0;
    *((_DWORD *)v17 - 20) = 0;
    *(v17 - 9) = 0;
    *(v17 - 8) = 0;
    *(v17 - 7) = 0;
    *((_DWORD *)v17 - 12) = 0;
    *(v17 - 5) = 0;
    *(v17 - 4) = 0;
    *(v17 - 3) = 0;
    *(v17 - 2) = 0;
    *(v17 - 1) = 1;
    do
    {
      if ( v17 )
        *v17 = -4096;
      v17 += 2;
    }
    while ( v17 != v16 );
    v18 = v14 + 23;
    v14[21] = 0;
    v14[22] = 1;
    do
    {
      if ( v18 )
      {
        *v18 = -4096;
        *((_DWORD *)v18 + 2) = 0x7FFFFFFF;
      }
      v18 += 3;
    }
    while ( v18 != v14 + 35 );
    sub_FF9360(v14, a3, v15, 0, 0, 0);
  }
  v19 = (unsigned __int64)v70;
  v70 = v14;
  if ( v19 )
  {
    sub_2D59CD0(v19);
    v14 = v70;
  }
  v20 = (__int64 *)sub_22077B0(8u);
  v21 = v20;
  if ( v20 )
    sub_FE7FB0(v20, (const char *)a3, (__int64)v14, v68);
  v22 = v69;
  v69 = v21;
  if ( v22 )
  {
    sub_FDC110(v22);
    j_j___libc_free_0((unsigned __int64)v22);
  }
  v23 = sub_BC1CD0(a4, &unk_4F82410, a3);
  v24 = *(_QWORD *)(a3 + 40);
  v25 = *(_QWORD *)(v23 + 8);
  v26 = *(unsigned int *)(v25 + 88);
  v27 = *(_QWORD *)(v25 + 72);
  if ( !(_DWORD)v26 )
    goto LABEL_58;
  v28 = 1;
  for ( i = (v26 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F87C68 >> 9) ^ ((unsigned int)&unk_4F87C68 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4)))); ; i = (v26 - 1) & v31 )
  {
    v30 = v27 + 24LL * i;
    if ( *(_UNKNOWN **)v30 == &unk_4F87C68 && v24 == *(_QWORD *)(v30 + 8) )
      break;
    if ( *(_QWORD *)v30 == -4096 && *(_QWORD *)(v30 + 8) == -4096 )
      goto LABEL_58;
    v31 = v28 + i;
    ++v28;
  }
  if ( v30 == v27 + 24 * v26 )
  {
LABEL_58:
    v32 = 0;
  }
  else
  {
    v32 = *(_QWORD *)(*(_QWORD *)(v30 + 16) + 24LL);
    if ( v32 )
    {
      v50 = 1;
      v32 += 8;
      v33 = &v51;
      do
      {
        *v33 = -4096;
        v33 += 2;
      }
      while ( v33 != &v61 );
      if ( (v50 & 1) == 0 )
        sub_C7D6A0(v51, 16LL * v52, 8);
    }
  }
  v34 = *(unsigned int *)(a4 + 88);
  v35 = *(_QWORD *)(a4 + 72);
  v71 = v32;
  if ( !(_DWORD)v34 )
    goto LABEL_60;
  v36 = 1;
  for ( j = (v34 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_5016950 >> 9) ^ ((unsigned int)&unk_5016950 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; j = (v34 - 1) & v39 )
  {
    v38 = v35 + 24LL * j;
    if ( *(_UNKNOWN **)v38 == &unk_5016950 && a3 == *(_QWORD *)(v38 + 8) )
      break;
    if ( *(_QWORD *)v38 == -4096 && *(_QWORD *)(v38 + 8) == -4096 )
      goto LABEL_60;
    v39 = v36 + j;
    ++v36;
  }
  if ( v38 == v35 + 24 * v34 )
  {
LABEL_60:
    v40 = 0;
  }
  else
  {
    v40 = *(_QWORD *)(*(_QWORD *)(v38 + 16) + 24LL);
    if ( v40 )
      v40 += 8;
  }
  v66 = v40;
  v41 = sub_2D88660((__int64)&v61, a3);
  v43 = a1 + 80;
  if ( v41 )
  {
    v49 = 0;
    v50 = (unsigned __int64)v54;
    v51 = 2;
    v52 = 0;
    v53 = 1;
    v55 = 0;
    v56 = v60;
    v57 = 2;
    v58 = 0;
    v59 = 1;
    if ( !(unsigned __int8)sub_B19060((__int64)&v49, (__int64)&qword_4F82400, v42, v43) )
      sub_AE6EC0((__int64)&v49, (__int64)&unk_4F6D3F8);
    sub_25DDDB0((__int64)&v55, (__int64)&unk_4F89C30);
    if ( HIDWORD(v57) != v58 || !(unsigned __int8)sub_B19060((__int64)&v49, (__int64)&qword_4F82400, v45, v46) )
      sub_AE6EC0((__int64)&v49, (__int64)&unk_4F89C30);
    sub_25DDDB0((__int64)&v55, (__int64)&unk_4F875F0);
    if ( HIDWORD(v57) != v58 || !(unsigned __int8)sub_B19060((__int64)&v49, (__int64)&qword_4F82400, v47, v48) )
      sub_AE6EC0((__int64)&v49, (__int64)&unk_4F875F0);
    sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v54, (__int64)&v49);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v60, (__int64)&v55);
    if ( !v59 )
      _libc_free((unsigned __int64)v56);
    if ( !v53 )
      _libc_free(v50);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)a1 = 0;
    *(_QWORD *)(a1 + 16) = 2;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = v43;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    sub_AE6EC0(a1, (__int64)&qword_4F82400);
  }
  sub_2D5C240((__int64)&v61);
  return a1;
}
