// Function: sub_291F660
// Address: 0x291f660
//
__int64 __fastcall sub_291F660(__int64 a1, _QWORD *a2)
{
  unsigned int v2; // eax
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rdx
  unsigned __int8 *v10; // r13
  __int64 v11; // rdx
  __int64 v12; // rbx
  __int64 v13; // r12
  unsigned int v14; // rdx^4
  char v15; // al
  __m128i v16; // xmm0
  __m128i v17; // xmm1
  __m128i v18; // xmm2
  __int64 *v19; // rax
  __int64 *v20; // rax
  char v21; // r14
  unsigned __int64 v22; // rdi
  unsigned int v23; // r15d
  unsigned __int64 v25; // rax
  unsigned __int8 v26; // al
  __int64 v27; // rax
  __int64 v28; // rcx
  unsigned __int8 **v29; // rdi
  unsigned __int8 *v30; // rbx
  unsigned __int8 v31; // al
  unsigned __int64 v32; // rax
  unsigned __int8 v33; // al
  __int64 v34; // rax
  __m128i *v35; // rdi
  unsigned __int32 v36; // eax
  __int64 v37; // rdx
  __int64 v38; // r13
  unsigned __int8 v39; // dl
  unsigned int v40; // eax
  int v41; // eax
  unsigned __int32 v42; // r13d
  _QWORD *v43; // rax
  __m128i v44; // rax
  __int64 v45; // r15
  char v46; // al
  __m128i v47; // xmm3
  __m128i v48; // xmm4
  __m128i v49; // xmm5
  __int64 *v50; // rax
  __int64 *v51; // rax
  __int64 v52; // r14
  __int64 v53; // r15
  __int64 v54; // r14
  __int64 v55; // r15
  int v56; // edx
  __int64 v57; // rax
  __int64 v58; // rdi
  unsigned __int64 v59; // rax
  __int64 v60; // r14
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 v63; // r8
  __int64 v64; // r9
  __int64 v65; // rbx
  __int64 v66; // rcx
  __int64 v67; // rdx
  unsigned __int64 v68; // rax
  int v69; // edx
  unsigned __int8 *v70; // r12
  unsigned __int8 *v71; // rax
  unsigned __int8 *v72; // r14
  __int64 v73; // rax
  __int64 *v74; // r14
  __int64 v75; // r15
  _QWORD *v76; // rax
  __int64 v77; // rax
  unsigned __int64 v78; // rdx
  __int64 v79; // [rsp+10h] [rbp-410h]
  __int64 v81; // [rsp+20h] [rbp-400h]
  char v82; // [rsp+28h] [rbp-3F8h]
  unsigned __int8 **v83; // [rsp+30h] [rbp-3F0h]
  unsigned __int8 **v84; // [rsp+30h] [rbp-3F0h]
  unsigned __int64 v85; // [rsp+38h] [rbp-3E8h]
  char v86; // [rsp+38h] [rbp-3E8h]
  unsigned __int8 v87; // [rsp+43h] [rbp-3DDh]
  unsigned __int32 v88; // [rsp+44h] [rbp-3DCh]
  __int64 v89; // [rsp+48h] [rbp-3D8h]
  unsigned __int8 **v90; // [rsp+58h] [rbp-3C8h]
  unsigned __int8 **v91; // [rsp+60h] [rbp-3C0h]
  char v92; // [rsp+60h] [rbp-3C0h]
  __int64 v93; // [rsp+60h] [rbp-3C0h]
  __int64 v94; // [rsp+68h] [rbp-3B8h]
  unsigned __int64 v95; // [rsp+70h] [rbp-3B0h] BYREF
  unsigned __int32 v96; // [rsp+78h] [rbp-3A8h]
  _QWORD *v97; // [rsp+80h] [rbp-3A0h] BYREF
  unsigned __int32 v98; // [rsp+88h] [rbp-398h]
  const void *v99; // [rsp+90h] [rbp-390h] BYREF
  unsigned int v100; // [rsp+98h] [rbp-388h]
  __m128i v101; // [rsp+A0h] [rbp-380h] BYREF
  __m128i v102; // [rsp+B0h] [rbp-370h] BYREF
  __m128i v103; // [rsp+C0h] [rbp-360h] BYREF
  __m128i v104; // [rsp+D0h] [rbp-350h] BYREF
  __m128i v105; // [rsp+E0h] [rbp-340h] BYREF
  __m128i v106; // [rsp+F0h] [rbp-330h]
  char v107; // [rsp+100h] [rbp-320h]
  unsigned __int8 **v108; // [rsp+110h] [rbp-310h] BYREF
  unsigned int v109; // [rsp+118h] [rbp-308h]
  _BYTE v110[48]; // [rsp+120h] [rbp-300h] BYREF
  __m128i v111; // [rsp+150h] [rbp-2D0h] BYREF
  __int64 v112; // [rsp+160h] [rbp-2C0h] BYREF
  __int64 v113; // [rsp+168h] [rbp-2B8h] BYREF
  _QWORD v114[4]; // [rsp+170h] [rbp-2B0h] BYREF
  __int16 v115; // [rsp+190h] [rbp-290h]
  __int64 v116; // [rsp+198h] [rbp-288h]
  void **v117; // [rsp+1A0h] [rbp-280h]
  __int64 (__fastcall ***v118)(); // [rsp+1A8h] [rbp-278h]
  __int64 v119; // [rsp+1B0h] [rbp-270h]
  int v120; // [rsp+1B8h] [rbp-268h]
  __int16 v121; // [rsp+1BCh] [rbp-264h]
  char v122; // [rsp+1BEh] [rbp-262h]
  __int64 v123; // [rsp+1C0h] [rbp-260h]
  __int64 v124; // [rsp+1C8h] [rbp-258h]
  void *v125; // [rsp+1D0h] [rbp-250h] BYREF
  __int64 (__fastcall **v126)(); // [rsp+1D8h] [rbp-248h] BYREF
  _QWORD *v127; // [rsp+1E0h] [rbp-240h]
  __int64 v128; // [rsp+1E8h] [rbp-238h]
  _BYTE v129[184]; // [rsp+1F0h] [rbp-230h] BYREF
  void **v130; // [rsp+2A8h] [rbp-178h] BYREF
  __int64 v131; // [rsp+2B0h] [rbp-170h]
  char v132; // [rsp+2B8h] [rbp-168h]
  _BYTE *v133; // [rsp+2C0h] [rbp-160h]
  __int64 v134; // [rsp+2C8h] [rbp-158h]
  _BYTE v135[128]; // [rsp+2D0h] [rbp-150h] BYREF
  __int16 v136; // [rsp+350h] [rbp-D0h]
  void *v137; // [rsp+358h] [rbp-C8h] BYREF
  __int64 v138; // [rsp+360h] [rbp-C0h]
  __int64 v139; // [rsp+368h] [rbp-B8h]
  __int64 v140; // [rsp+370h] [rbp-B0h] BYREF
  unsigned int v141; // [rsp+378h] [rbp-A8h]
  _BYTE v142[48]; // [rsp+3F0h] [rbp-30h] BYREF

  v89 = sub_B43CC0(a1);
  v79 = *(_QWORD *)(a1 + 40);
  v2 = sub_AE43F0(v89, *(_QWORD *)(a1 + 8));
  v88 = v2;
  v96 = v2;
  if ( v2 > 0x40 )
  {
    sub_C43690((__int64)&v95, 0, 0);
    v98 = v88;
    sub_C43690((__int64)&v97, 0, 0);
  }
  else
  {
    v95 = 0;
    v98 = v2;
    v97 = 0;
  }
  sub_2914D90((__int64)&v108, a1, v3, v4, v5, v6);
  v9 = v109;
  v83 = &v108[v109];
  if ( v108 == v83 )
  {
    v87 = 0;
    goto LABEL_81;
  }
  v91 = v108;
  v87 = 0;
  while ( 2 )
  {
    v10 = *v91;
    if ( **v91 != 61 )
      goto LABEL_5;
    if ( sub_B46500(*v91) )
      goto LABEL_39;
    v82 = v10[2] & 1;
    if ( v82 || v79 != *((_QWORD *)v10 + 5) )
      goto LABEL_39;
    v81 = *((_QWORD *)v10 + 1);
    v111.m128i_i64[0] = sub_9208B0(v89, v81);
    v111.m128i_i64[1] = v11;
    v100 = v88;
    if ( v88 > 0x40 )
      sub_C43690((__int64)&v99, (unsigned __int64)(v111.m128i_i64[0] + 7) >> 3, 0);
    else
      v99 = (const void *)((unsigned __int64)(v111.m128i_i64[0] + 7) >> 3);
    v85 = 0;
    v12 = a1 + 24;
    while ( v10 != (unsigned __int8 *)(v12 - 24) )
    {
      v15 = *(_BYTE *)(v12 - 24);
      if ( v15 == 62 )
      {
        sub_D665A0(&v111, (__int64)v10);
        sub_D66630(&v104, v12 - 24);
        if ( (unsigned __int8)sub_CF4E00((__int64)a2, (__int64)&v104, (__int64)&v111) == 3 )
        {
          v13 = *(_QWORD *)(*(_QWORD *)(v12 - 88) + 8LL);
          v111.m128i_i64[0] = sub_9208B0(v89, v13);
          v111.m128i_i64[1] = __PAIR64__(v14, v88);
          if ( v88 > 0x40 )
            sub_C43690((__int64)&v111, (unsigned __int64)(v111.m128i_i64[0] + 7) >> 3, 0);
          else
            v111.m128i_i64[0] = (unsigned __int64)(v111.m128i_i64[0] + 7) >> 3;
          if ( v100 <= 0x40 )
          {
            if ( v99 == (const void *)v111.m128i_i64[0] )
              goto LABEL_56;
          }
          else
          {
            if ( !sub_C43C50((__int64)&v99, (const void **)&v111) )
              goto LABEL_18;
LABEL_56:
            if ( sub_29191E0(v89, v13, v81) )
            {
              v85 = *(_QWORD *)(v12 - 88);
              goto LABEL_19;
            }
          }
LABEL_18:
          v82 = 1;
          v85 = 0;
LABEL_19:
          if ( v111.m128i_i32[2] > 0x40u && v111.m128i_i64[0] )
            j_j___libc_free_0_0(v111.m128i_u64[0]);
          goto LABEL_22;
        }
        sub_D665A0(&v111, (__int64)v10);
        sub_D66630(&v104, v12 - 24);
        if ( (unsigned __int8)sub_CF4E00((__int64)a2, (__int64)&v104, (__int64)&v111) )
        {
          if ( v100 <= 0x40 )
            goto LABEL_39;
          goto LABEL_68;
        }
      }
      else if ( v15 == 85 )
      {
        sub_D665A0(&v101, (__int64)v10);
        v16 = _mm_loadu_si128(&v101);
        v107 = 1;
        v17 = _mm_loadu_si128(&v102);
        v18 = _mm_loadu_si128(&v103);
        v112 = 1;
        v111 = (__m128i)(unsigned __int64)a2;
        v19 = &v113;
        v104 = v16;
        v105 = v17;
        v106 = v18;
        do
        {
          *v19 = -4;
          v19 += 5;
          *(v19 - 4) = -3;
          *(v19 - 3) = -4;
          *(v19 - 2) = -3;
        }
        while ( v19 != (__int64 *)&v130 );
        v131 = 0;
        v136 = 256;
        v133 = v135;
        v130 = &v137;
        v132 = 0;
        v138 = 0;
        v139 = 1;
        v134 = 0x400000000LL;
        v137 = &unk_49DDBE8;
        v20 = &v140;
        do
        {
          *v20 = -4096;
          v20 += 2;
        }
        while ( v20 != (__int64 *)v142 );
        v21 = sub_CF63E0(a2, (unsigned __int8 *)(v12 - 24), &v104, (__int64)&v111);
        v137 = &unk_49DDBE8;
        if ( (v139 & 1) == 0 )
          sub_C7D6A0(v140, 16LL * v141, 8);
        nullsub_184();
        if ( v133 != v135 )
          _libc_free((unsigned __int64)v133);
        if ( (v112 & 1) == 0 )
          sub_C7D6A0(v113, 40LL * LODWORD(v114[0]), 8);
        if ( (v21 & 2) != 0 || !(unsigned __int8)sub_B46900((unsigned __int8 *)(v12 - 24)) )
          goto LABEL_38;
      }
      else if ( (unsigned __int8)sub_B46970((unsigned __int8 *)(v12 - 24)) )
      {
        if ( v100 > 0x40 )
          goto LABEL_68;
LABEL_39:
        v22 = (unsigned __int64)v108;
        if ( v108 == (unsigned __int8 **)v110 )
          goto LABEL_41;
        goto LABEL_40;
      }
LABEL_22:
      v12 = *(_QWORD *)(v12 + 8);
      if ( !v12 )
        BUG();
    }
    if ( v85 )
    {
      v116 = sub_BD5C60((__int64)v10);
      v121 = 512;
      v115 = 0;
      v125 = &unk_49DA100;
      v111.m128i_i64[1] = 0x200000000LL;
      v126 = off_49D3D08;
      v111.m128i_i64[0] = (__int64)&v112;
      v117 = &v125;
      v118 = &v126;
      v119 = 0;
      v120 = 0;
      v122 = 7;
      v123 = 0;
      v124 = 0;
      v114[2] = 0;
      v114[3] = 0;
      v127 = v129;
      v128 = 0;
      v129[0] = 0;
      sub_D5F1F0((__int64)&v111, (__int64)v10);
      v27 = sub_291C8F0(v89, (unsigned int **)&v111, v85, v81);
      sub_BD84D0((__int64)v10, v27);
      sub_B43D60(v10);
      v126 = off_49D3D08;
      if ( v127 != (_QWORD *)v129 )
        j_j___libc_free_0((unsigned __int64)v127);
      nullsub_61();
      v125 = &unk_49DA100;
      nullsub_63();
      if ( (__int64 *)v111.m128i_i64[0] != &v112 )
        _libc_free(v111.m128i_u64[0]);
      if ( v100 > 0x40 )
      {
LABEL_65:
        if ( v99 )
          j_j___libc_free_0_0((unsigned __int64)v99);
      }
    }
    else
    {
      if ( v82 )
      {
LABEL_38:
        if ( v100 <= 0x40 )
          goto LABEL_39;
LABEL_68:
        if ( v99 )
          j_j___libc_free_0_0((unsigned __int64)v99);
        goto LABEL_39;
      }
      if ( (int)sub_C49970((__int64)&v99, &v95) > 0 )
      {
        if ( v96 > 0x40 || v100 > 0x40 )
        {
          sub_C43990((__int64)&v95, (__int64)&v99);
          goto LABEL_62;
        }
        v96 = v100;
        v95 = (unsigned __int64)v99;
        _BitScanReverse64(&v32, 1LL << (*((_WORD *)v10 + 1) >> 1));
        v9 = 63 - ((unsigned int)v32 ^ 0x3F);
        v33 = v87;
        if ( v87 < (unsigned __int8)v9 )
          v33 = v9;
        v87 = v33;
      }
      else
      {
LABEL_62:
        _BitScanReverse64(&v25, 1LL << (*((_WORD *)v10 + 1) >> 1));
        v9 = 63 - ((unsigned int)v25 ^ 0x3F);
        v26 = v87;
        if ( v87 < (unsigned __int8)v9 )
          v26 = v9;
        v87 = v26;
        if ( v100 > 0x40 )
          goto LABEL_65;
      }
    }
LABEL_5:
    if ( v83 != ++v91 )
      continue;
    break;
  }
  v83 = v108;
LABEL_81:
  v28 = (__int64)v83;
  if ( v83 != (unsigned __int8 **)v110 )
    _libc_free((unsigned __int64)v83);
  sub_2914D90((__int64)&v108, a1, v9, v28, v7, v8);
  v29 = v108;
  v84 = &v108[v109];
  if ( v84 != v108 )
  {
    v90 = v108;
    while ( 1 )
    {
      v30 = *v90;
      v31 = **v90;
      if ( v31 <= 0x1Cu )
        goto LABEL_88;
      if ( v31 != 62 )
      {
        if ( v31 != 61 )
          goto LABEL_88;
        goto LABEL_103;
      }
      if ( sub_B46500(*v90) )
        goto LABEL_88;
      v86 = v30[2] & 1;
      if ( v86 || v79 != *((_QWORD *)v30 + 5) )
        goto LABEL_88;
      v42 = v98;
      if ( v98 > 0x40 )
      {
        if ( v42 - (unsigned int)sub_C444A0((__int64)&v97) > 0x40 )
          goto LABEL_88;
        v43 = (_QWORD *)*v97;
      }
      else
      {
        v43 = v97;
      }
      if ( v43 )
        goto LABEL_88;
      v44.m128i_i64[0] = sub_9C6480(v89, *(_QWORD *)(*((_QWORD *)v30 - 8) + 8LL));
      v104 = v44;
      v111.m128i_i32[2] = v88;
      if ( v88 > 0x40 )
        sub_C43690((__int64)&v111, v44.m128i_i64[0], 0);
      else
        v111.m128i_i64[0] = v44.m128i_i64[0];
      if ( v98 > 0x40 && v97 )
        j_j___libc_free_0_0((unsigned __int64)v97);
      v97 = (_QWORD *)v111.m128i_i64[0];
      v98 = v111.m128i_u32[2];
      v45 = a1 + 24;
      while ( v30 != (unsigned __int8 *)(v45 - 24) )
      {
        v46 = *(_BYTE *)(v45 - 24);
        switch ( v46 )
        {
          case '=':
            sub_D665A0(&v111, v45 - 24);
            goto LABEL_122;
          case '>':
            sub_D66630(&v111, v45 - 24);
LABEL_122:
            sub_D66630(&v104, (__int64)v30);
            if ( (unsigned __int8)sub_CF4E00((__int64)a2, (__int64)&v104, (__int64)&v111) )
              goto LABEL_88;
            break;
          case 'U':
            sub_D66630(&v101, (__int64)v30);
            v47 = _mm_loadu_si128(&v101);
            v48 = _mm_loadu_si128(&v102);
            v107 = 1;
            v49 = _mm_loadu_si128(&v103);
            v111 = (__m128i)(unsigned __int64)a2;
            v50 = &v113;
            v112 = 1;
            v104 = v47;
            v105 = v48;
            v106 = v49;
            do
            {
              *v50 = -4;
              v50 += 5;
              *(v50 - 4) = -3;
              *(v50 - 3) = -4;
              *(v50 - 2) = -3;
            }
            while ( v50 != (__int64 *)&v130 );
            v131 = 0;
            v130 = &v137;
            v133 = v135;
            v132 = 0;
            v136 = 256;
            v138 = 0;
            v139 = 1;
            v134 = 0x400000000LL;
            v137 = &unk_49DDBE8;
            v51 = &v140;
            do
            {
              *v51 = -4096;
              v51 += 2;
            }
            while ( v51 != (__int64 *)v142 );
            v92 = sub_CF63E0(a2, (unsigned __int8 *)(v45 - 24), &v104, (__int64)&v111);
            v137 = &unk_49DDBE8;
            if ( (v139 & 1) == 0 )
              sub_C7D6A0(v140, 16LL * v141, 8);
            nullsub_184();
            if ( v133 != v135 )
              _libc_free((unsigned __int64)v133);
            if ( (v112 & 1) == 0 )
              sub_C7D6A0(v113, 40LL * LODWORD(v114[0]), 8);
            if ( v92 || !(unsigned __int8)sub_B46900((unsigned __int8 *)(v45 - 24)) )
              goto LABEL_88;
            break;
          default:
            if ( (unsigned __int8)sub_B46970((unsigned __int8 *)(v45 - 24)) || (unsigned __int8)sub_B46420(v45 - 24) )
              goto LABEL_88;
            break;
        }
        v45 = *(_QWORD *)(v45 + 8);
        if ( !v45 )
          BUG();
      }
      v52 = 32LL * *(unsigned int *)(a1 + 72);
      v93 = *(_QWORD *)(a1 - 8);
      v53 = v52 + 8LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
      v54 = v93 + v52;
      v55 = v93 + v53;
      if ( v55 != v54 )
      {
        while ( 1 )
        {
          v59 = *(_QWORD *)(*(_QWORD *)v54 + 48LL) & 0xFFFFFFFFFFFFFFF8LL;
          if ( v59 == *(_QWORD *)v54 + 48LL )
          {
            v58 = 0;
          }
          else
          {
            if ( !v59 )
              BUG();
            v56 = *(unsigned __int8 *)(v59 - 24);
            v57 = v59 - 24;
            if ( (unsigned int)(v56 - 30) >= 0xB )
              v57 = 0;
            v58 = v57;
          }
          if ( (unsigned int)sub_B46E30(v58) != 1 )
            break;
          v54 += 8;
          if ( v55 == v54 )
            goto LABEL_93;
        }
LABEL_88:
        v22 = (unsigned __int64)v108;
        if ( v108 == (unsigned __int8 **)v110 )
        {
LABEL_41:
          v23 = 0;
          goto LABEL_42;
        }
LABEL_40:
        _libc_free(v22);
        goto LABEL_41;
      }
LABEL_93:
      v34 = *((_QWORD *)v30 - 8);
      v111.m128i_i64[1] = (__int64)v114;
      v35 = &v105;
      v114[0] = v34;
      v105.m128i_i64[0] = v34;
      v104.m128i_i64[1] = 0x400000001LL;
      v36 = 1;
      v104.m128i_i64[0] = (__int64)&v105;
      v112 = 0x100000004LL;
      LODWORD(v113) = 0;
      BYTE4(v113) = 1;
      v111.m128i_i64[0] = 1;
      while ( 2 )
      {
        v37 = v36--;
        v38 = v35->m128i_i64[v37 - 1];
        v104.m128i_i32[2] = v36;
        v39 = *(_BYTE *)v38;
        if ( *(_BYTE *)v38 > 0x1Cu )
        {
          if ( v79 != *(_QWORD *)(v38 + 40) )
            goto LABEL_95;
          if ( v39 == 85 )
          {
            v60 = *(_QWORD *)(v38 - 32);
            if ( v60 )
            {
              if ( !*(_BYTE *)v60 && *(_QWORD *)(v60 + 24) == *(_QWORD *)(v38 + 80) )
              {
                if ( (unsigned __int8)sub_B2D610(*(_QWORD *)(v38 - 32), 27) )
                  goto LABEL_167;
                if ( (unsigned __int8)sub_B2D610(v60, 6) )
                {
                  v86 = 0;
                  v35 = (__m128i *)v104.m128i_i64[0];
                  goto LABEL_98;
                }
              }
            }
          }
          if ( (unsigned __int8)sub_B46420(v38) || (unsigned __int8)sub_B46490(v38) )
          {
LABEL_167:
            v35 = (__m128i *)v104.m128i_i64[0];
            goto LABEL_98;
          }
          if ( *(_BYTE *)v38 == 84
            || ((v73 = 4LL * (*(_DWORD *)(v38 + 4) & 0x7FFFFFF), (*(_BYTE *)(v38 + 7) & 0x40) == 0)
              ? (v74 = (__int64 *)(v38 - v73 * 8))
              : (__int64 *)(v74 = *(__int64 **)(v38 - 8), v38 = (__int64)&v74[v73]),
                (__int64 *)v38 == v74) )
          {
LABEL_166:
            v36 = v104.m128i_u32[2];
            v35 = (__m128i *)v104.m128i_i64[0];
            goto LABEL_95;
          }
          while ( 2 )
          {
            v75 = *v74;
            if ( !BYTE4(v113) )
              goto LABEL_199;
            v76 = (_QWORD *)v111.m128i_i64[1];
            v61 = HIDWORD(v112);
            v62 = v111.m128i_i64[1] + 8LL * HIDWORD(v112);
            if ( v111.m128i_i64[1] == v62 )
            {
LABEL_194:
              if ( HIDWORD(v112) < (unsigned int)v112 )
              {
                ++HIDWORD(v112);
                *(_QWORD *)v62 = v75;
                ++v111.m128i_i64[0];
                goto LABEL_196;
              }
LABEL_199:
              sub_C8CC70((__int64)&v111, *v74, v61, v62, v63, v64);
              if ( (_BYTE)v61 )
              {
LABEL_196:
                v77 = v104.m128i_u32[2];
                v62 = v104.m128i_u32[3];
                v78 = v104.m128i_u32[2] + 1LL;
                if ( v78 > v104.m128i_u32[3] )
                {
                  sub_C8D5F0((__int64)&v104, &v105, v78, 8u, v63, v64);
                  v77 = v104.m128i_u32[2];
                }
                v61 = v104.m128i_i64[0];
                *(_QWORD *)(v104.m128i_i64[0] + 8 * v77) = v75;
                ++v104.m128i_i32[2];
              }
            }
            else
            {
              while ( v75 != *v76 )
              {
                if ( (_QWORD *)v62 == ++v76 )
                  goto LABEL_194;
              }
            }
            v74 += 4;
            if ( (__int64 *)v38 == v74 )
              goto LABEL_166;
            continue;
          }
        }
        if ( v39 > 0x16u )
          goto LABEL_98;
LABEL_95:
        if ( v36 )
          continue;
        break;
      }
      v86 = 1;
LABEL_98:
      if ( v35 != &v105 )
        _libc_free((unsigned __int64)v35);
      if ( !BYTE4(v113) )
        _libc_free(v111.m128i_u64[1]);
      if ( !v86 )
        goto LABEL_88;
LABEL_103:
      if ( v84 == ++v90 )
      {
        v29 = v108;
        break;
      }
    }
  }
  if ( v29 != (unsigned __int8 **)v110 )
    _libc_free((unsigned __int64)v29);
  LOBYTE(v40) = sub_D94970((__int64)&v95, 0);
  v23 = v40;
  if ( (_BYTE)v40 )
  {
    LOBYTE(v41) = sub_D94970((__int64)&v97, 0);
    v23 = v41 ^ 1;
  }
  else
  {
    v111.m128i_i32[2] = v96;
    if ( v96 > 0x40 )
      sub_C43780((__int64)&v111, (const void **)&v95);
    else
      v111.m128i_i64[0] = v95;
    v65 = 0;
    v94 = 8LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF);
    if ( (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) != 0 )
    {
      while ( 1 )
      {
        v66 = *(_QWORD *)(a1 - 8);
        v67 = *(_QWORD *)(v66 + 32LL * *(unsigned int *)(a1 + 72) + v65);
        v68 = *(_QWORD *)(v67 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v68 == v67 + 48 )
        {
          v70 = 0;
        }
        else
        {
          if ( !v68 )
            BUG();
          v69 = *(unsigned __int8 *)(v68 - 24);
          v70 = 0;
          v71 = (unsigned __int8 *)(v68 - 24);
          if ( (unsigned int)(v69 - 30) < 0xB )
            v70 = v71;
        }
        v72 = *(unsigned __int8 **)(v66 + 4 * v65);
        if ( v72 == v70
          || (unsigned __int8)sub_B46970(v70)
          || (unsigned int)sub_B46E30((__int64)v70) != 1
          && !(unsigned __int8)sub_D30F00(v72, v87, (__int64)&v111, v89, (__int64)v70, 0, 0, 0) )
        {
          break;
        }
        v65 += 8;
        if ( v65 == v94 )
          goto LABEL_203;
      }
    }
    else
    {
LABEL_203:
      v23 = 1;
    }
    sub_969240(v111.m128i_i64);
  }
LABEL_42:
  if ( v98 > 0x40 && v97 )
    j_j___libc_free_0_0((unsigned __int64)v97);
  if ( v96 > 0x40 && v95 )
    j_j___libc_free_0_0(v95);
  return v23;
}
