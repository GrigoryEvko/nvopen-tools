// Function: sub_37EC0D0
// Address: 0x37ec0d0
//
__int64 __fastcall sub_37EC0D0(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 (*v3)(void); // rax
  __int64 v4; // rdx
  __int64 *v5; // rbx
  int v6; // r12d
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // r15
  _QWORD *v11; // r12
  __int64 v12; // r15
  __int64 v13; // rax
  __int64 v14; // r9
  __int64 (*v15)(); // rax
  __int64 v16; // rdi
  void (*v17)(); // rax
  __int64 v18; // rax
  int v19; // r14d
  int v20; // eax
  __int64 v21; // rcx
  unsigned __int64 v22; // r8
  int v23; // eax
  __int64 v24; // rax
  __int64 v25; // rdx
  int v26; // r10d
  __int64 v27; // rax
  unsigned int v28; // esi
  int v30; // eax
  int v31; // r10d
  __int64 v32; // rcx
  unsigned __int64 v33; // r8
  int v34; // eax
  __int64 v35; // rax
  __int64 v36; // rdx
  int v37; // eax
  int v38; // ecx
  unsigned int v39; // esi
  __int64 v40; // rax
  __int32 v43; // ebx
  __int64 v44; // rax
  __int64 v45; // r15
  _QWORD *v46; // r12
  __int64 v47; // r15
  __int64 v48; // rdx
  __int64 v49; // rax
  unsigned __int32 v50; // eax
  unsigned int v51; // esi
  int v52; // ecx
  unsigned __int64 v53; // r8
  unsigned __int64 v54; // r8
  __int32 v58; // ebx
  char v59; // cl
  __int64 v60; // rdi
  int v61; // edx
  unsigned int v62; // esi
  int *v63; // rax
  int v64; // r8d
  unsigned int v65; // ecx
  __int64 v66; // rdx
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // r15
  __int64 v70; // r12
  _QWORD *v71; // r14
  __int64 v72; // r15
  __int64 v73; // rax
  unsigned __int32 v74; // ebx
  unsigned __int32 v75; // r8d
  unsigned int v76; // esi
  int v77; // ebx
  unsigned __int64 v78; // rdi
  __int64 v79; // rdx
  unsigned __int64 v80; // rdi
  __int64 v83; // rax
  __int64 v84; // rax
  __int64 v85; // rax
  __int64 v86; // rax
  __int64 v87; // r15
  __int64 v88; // r14
  _QWORD *v89; // r12
  __int64 v90; // r15
  __int64 v91; // rax
  unsigned __int8 *v92; // rsi
  int v93; // eax
  unsigned int v94; // r15d
  unsigned __int64 v96; // r14
  unsigned __int64 v97; // r14
  __int64 v98; // rax
  __int64 v99; // r14
  __int64 v100; // r15
  _QWORD *v101; // r12
  __int64 v102; // r14
  __int64 v103; // rax
  int v104; // r9d
  __int64 *v105; // [rsp+8h] [rbp-188h]
  __int64 v106; // [rsp+18h] [rbp-178h]
  int v107; // [rsp+30h] [rbp-160h]
  int v108; // [rsp+30h] [rbp-160h]
  int v109; // [rsp+30h] [rbp-160h]
  __int64 v110; // [rsp+38h] [rbp-158h]
  __int64 v111; // [rsp+38h] [rbp-158h]
  int v112; // [rsp+38h] [rbp-158h]
  int v113; // [rsp+38h] [rbp-158h]
  __int64 *v114; // [rsp+40h] [rbp-150h]
  __int64 v115; // [rsp+48h] [rbp-148h]
  __int64 v117; // [rsp+58h] [rbp-138h]
  __int64 *v119; // [rsp+68h] [rbp-128h]
  unsigned __int8 *v120; // [rsp+70h] [rbp-120h] BYREF
  unsigned __int8 *v121; // [rsp+78h] [rbp-118h] BYREF
  unsigned __int8 *v122; // [rsp+80h] [rbp-110h] BYREF
  __int64 v123; // [rsp+88h] [rbp-108h]
  __int64 v124; // [rsp+90h] [rbp-100h]
  _BYTE *v125; // [rsp+A0h] [rbp-F0h] BYREF
  __int64 v126; // [rsp+A8h] [rbp-E8h]
  _BYTE v127[48]; // [rsp+B0h] [rbp-E0h] BYREF
  int v128; // [rsp+E0h] [rbp-B0h]
  __m128i v129; // [rsp+F0h] [rbp-A0h] BYREF
  __int64 v130; // [rsp+100h] [rbp-90h]
  int v131; // [rsp+108h] [rbp-88h]
  char v132; // [rsp+110h] [rbp-80h]
  __int64 v133; // [rsp+118h] [rbp-78h]
  unsigned __int64 v134; // [rsp+120h] [rbp-70h]
  __int64 v135; // [rsp+128h] [rbp-68h]
  __int64 v136; // [rsp+130h] [rbp-60h]
  _QWORD *v137; // [rsp+138h] [rbp-58h]
  __int64 v138; // [rsp+140h] [rbp-50h]
  _BYTE v139[72]; // [rsp+148h] [rbp-48h] BYREF

  v117 = *(_QWORD *)(a2 + 328);
  v115 = 0;
  v2 = *(_QWORD *)(a1 + 200) + 184LL * *(int *)(v117 + 24);
  v3 = *(__int64 (**)(void))(**(_QWORD **)(a2 + 16) + 128LL);
  if ( v3 != sub_2DAC790 )
  {
    v115 = v3();
    v117 = *(_QWORD *)(a2 + 328);
  }
  v128 = 0;
  v125 = v127;
  v126 = 0x600000000LL;
  v106 = a2 + 320;
  if ( a2 + 320 == v117 )
    return 0;
  v4 = v117;
  v5 = (__int64 *)v2;
  v6 = 0;
  while ( 1 )
  {
    v18 = *(int *)(v117 + 24);
    if ( (_DWORD)v18 == *(_DWORD *)(v4 + 24) )
      goto LABEL_32;
    v119 = (__int64 *)(*(_QWORD *)(a1 + 200) + 184 * v18);
    v114 = *(__int64 **)(*v119 + 56);
    sub_2E32810((__int64 *)&v120, *v119, (__int64)v114);
    v19 = *(unsigned __int8 *)(v117 + 260);
    if ( v5[2] != v119[1] )
    {
      v7 = *((unsigned int *)v119 + 6);
      if ( *((_DWORD *)v5 + 7) != (_DWORD)v7 || (_BYTE)v19 )
      {
LABEL_7:
        v8 = *(_QWORD *)(*(_QWORD *)(a1 + 200) + 184LL * *(int *)(v117 + 24) + 8);
        v132 = 7;
        v129.m128i_i64[0] = 0;
        v133 = 0;
        v134 = 0;
        v135 = 0;
        v136 = 0;
        v137 = v139;
        v138 = 0;
        v139[0] = 0;
        v129.m128i_i32[2] = v7;
        v130 = v8;
        v107 = sub_2E7A450((_QWORD *)a2, (__int64)&v129, v7);
        if ( v137 != (_QWORD *)v139 )
          j_j___libc_free_0((unsigned __int64)v137);
        if ( v134 )
          j_j___libc_free_0(v134);
        v9 = *(_QWORD *)(v115 + 8);
        v121 = v120;
        v10 = v9 - 120;
        if ( v120 )
        {
          sub_B96E90((__int64)&v121, (__int64)v120, 1);
          v122 = v121;
          if ( v121 )
          {
            sub_B976B0((__int64)&v121, v121, (__int64)&v122);
            v121 = 0;
            v123 = 0;
            v124 = 0;
            v11 = *(_QWORD **)(*v119 + 32);
            v110 = *v119;
            v129.m128i_i64[0] = (__int64)v122;
            if ( v122 )
              sub_B96E90((__int64)&v129, (__int64)v122, 1);
LABEL_15:
            v12 = (__int64)sub_2E7B380(v11, v10, (unsigned __int8 **)&v129, 0);
            if ( v129.m128i_i64[0] )
              sub_B91220((__int64)&v129, v129.m128i_i64[0]);
            sub_2E31040((__int64 *)(v110 + 40), v12);
            v13 = *v114;
            *(_QWORD *)(v12 + 8) = v114;
            *(_QWORD *)v12 = v13 & 0xFFFFFFFFFFFFFFF8LL | *(_QWORD *)v12 & 7LL;
            *(_QWORD *)((v13 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v12;
            *v114 = v12 | *v114 & 7;
            if ( v123 )
              sub_2E882B0(v12, (__int64)v11, v123);
            if ( v124 )
              sub_2E88680(v12, (__int64)v11, v124);
            v129.m128i_i64[0] = 16;
            v130 = 0;
            v131 = v107;
            sub_2E8EAD0(v12, (__int64)v11, &v129);
            if ( v122 )
              sub_B91220((__int64)&v122, (__int64)v122);
            if ( v121 )
              sub_B91220((__int64)&v121, (__int64)v121);
            if ( (_BYTE)v19 )
            {
              v15 = *(__int64 (**)())(**(_QWORD **)(a2 + 16) + 136LL);
              if ( v15 == sub_2DD19D0 )
                BUG();
              v16 = v15();
              v17 = *(void (**)())(*(_QWORD *)v16 + 120LL);
              if ( v17 != nullsub_1671 )
                ((void (__fastcall *)(__int64, __int64, __int64 *))v17)(v16, *v119, v114);
              if ( v120 )
                sub_B91220((__int64)&v120, (__int64)v120);
              v5 = v119;
              v6 = v19;
              goto LABEL_32;
            }
            goto LABEL_177;
          }
        }
        else
        {
          v122 = 0;
        }
        v123 = 0;
        v124 = 0;
        v11 = *(_QWORD **)(*v119 + 32);
        v110 = *v119;
        v129.m128i_i64[0] = 0;
        goto LABEL_15;
      }
      v85 = *(_QWORD *)(*(_QWORD *)(a1 + 200) + 184LL * *(int *)(v117 + 24) + 8);
      v132 = 6;
      v129.m128i_i64[0] = 0;
      v133 = 0;
      v134 = 0;
      v135 = 0;
      v136 = 0;
      v137 = v139;
      v138 = 0;
      v139[0] = 0;
      v129.m128i_i32[2] = 0;
      v130 = v85;
      v112 = sub_2E7A450((_QWORD *)a2, (__int64)&v129, v7);
      if ( v137 != (_QWORD *)v139 )
        j_j___libc_free_0((unsigned __int64)v137);
      if ( v134 )
        j_j___libc_free_0(v134);
      v86 = *(_QWORD *)(v115 + 8);
      v121 = v120;
      v87 = v86 - 120;
      if ( v120 )
      {
        sub_B96E90((__int64)&v121, (__int64)v120, 1);
        v122 = v121;
        if ( v121 )
        {
          sub_B976B0((__int64)&v121, v121, (__int64)&v122);
          v121 = 0;
          v123 = 0;
          v124 = 0;
          v88 = *v119;
          v89 = *(_QWORD **)(*v119 + 32);
          v129.m128i_i64[0] = (__int64)v122;
          if ( v122 )
            sub_B96E90((__int64)&v129, (__int64)v122, 1);
LABEL_167:
          v90 = (__int64)sub_2E7B380(v89, v87, (unsigned __int8 **)&v129, 0);
          if ( v129.m128i_i64[0] )
            sub_B91220((__int64)&v129, v129.m128i_i64[0]);
          sub_2E31040((__int64 *)(v88 + 40), v90);
          v91 = *v114;
          *(_QWORD *)(v90 + 8) = v114;
          *(_QWORD *)v90 = v91 & 0xFFFFFFFFFFFFFFF8LL | *(_QWORD *)v90 & 7LL;
          *(_QWORD *)((v91 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v90;
          *v114 = v90 | *v114 & 7;
          if ( v123 )
            sub_2E882B0(v90, (__int64)v89, v123);
          if ( v124 )
            sub_2E88680(v90, (__int64)v89, v124);
          v129.m128i_i64[0] = 16;
          v130 = 0;
          v131 = v112;
          sub_2E8EAD0(v90, (__int64)v89, &v129);
          v92 = v122;
          if ( !v122 )
            goto LABEL_175;
          goto LABEL_174;
        }
      }
      else
      {
        v122 = 0;
      }
      v123 = 0;
      v124 = 0;
      v88 = *v119;
      v89 = *(_QWORD **)(*v119 + 32);
      v129.m128i_i64[0] = 0;
      goto LABEL_167;
    }
    v7 = *((unsigned int *)v119 + 6);
    if ( (_BYTE)v19 )
      goto LABEL_7;
    if ( *((_DWORD *)v5 + 7) != (_DWORD)v7 )
    {
      v129.m128i_i64[0] = 0;
      v132 = 5;
      v133 = 0;
      v134 = 0;
      v135 = 0;
      v136 = 0;
      v137 = v139;
      v138 = 0;
      v139[0] = 0;
      v129.m128i_i32[2] = v7;
      v130 = 0;
      v113 = sub_2E7A450((_QWORD *)a2, (__int64)&v129, v7);
      if ( v137 != (_QWORD *)v139 )
        j_j___libc_free_0((unsigned __int64)v137);
      if ( v134 )
        j_j___libc_free_0(v134);
      v98 = *(_QWORD *)(v115 + 8);
      v121 = v120;
      v99 = v98 - 120;
      if ( v120 )
      {
        sub_B96E90((__int64)&v121, (__int64)v120, 1);
        v122 = v121;
        if ( v121 )
        {
          sub_B976B0((__int64)&v121, v121, (__int64)&v122);
          v121 = 0;
          v123 = 0;
          v124 = 0;
          v100 = *v119;
          v101 = *(_QWORD **)(*v119 + 32);
          v129.m128i_i64[0] = (__int64)v122;
          if ( v122 )
            sub_B96E90((__int64)&v129, (__int64)v122, 1);
LABEL_203:
          v102 = (__int64)sub_2E7B380(v101, v99, (unsigned __int8 **)&v129, 0);
          if ( v129.m128i_i64[0] )
            sub_B91220((__int64)&v129, v129.m128i_i64[0]);
          sub_2E31040((__int64 *)(v100 + 40), v102);
          v103 = *v114;
          *(_QWORD *)(v102 + 8) = v114;
          *(_QWORD *)v102 = v103 & 0xFFFFFFFFFFFFFFF8LL | *(_QWORD *)v102 & 7LL;
          *(_QWORD *)((v103 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v102;
          *v114 = v102 | *v114 & 7;
          if ( v123 )
            sub_2E882B0(v102, (__int64)v101, v123);
          if ( v124 )
            sub_2E88680(v102, (__int64)v101, v124);
          v129.m128i_i64[0] = 16;
          v130 = 0;
          v131 = v113;
          sub_2E8EAD0(v102, (__int64)v101, &v129);
          v92 = v122;
          if ( !v122 )
          {
LABEL_175:
            if ( v121 )
              sub_B91220((__int64)&v121, (__int64)v121);
LABEL_177:
            v6 = 1;
            goto LABEL_38;
          }
LABEL_174:
          sub_B91220((__int64)&v122, (__int64)v92);
          goto LABEL_175;
        }
      }
      else
      {
        v122 = 0;
      }
      v123 = 0;
      v124 = 0;
      v100 = *v119;
      v101 = *(_QWORD **)(*v119 + 32);
      v129.m128i_i64[0] = 0;
      goto LABEL_203;
    }
LABEL_38:
    v20 = *((_DWORD *)v5 + 42);
    if ( (v128 & 0x3F) != 0 )
      *(_QWORD *)&v125[8 * (unsigned int)v126 - 8] &= ~(-1LL << (v128 & 0x3F));
    v21 = (unsigned int)v126;
    v128 = v20;
    v22 = (unsigned int)(v20 + 63) >> 6;
    if ( v22 != (unsigned int)v126 )
    {
      if ( v22 >= (unsigned int)v126 )
      {
        v96 = v22 - (unsigned int)v126;
        if ( v22 > HIDWORD(v126) )
        {
          sub_C8D5F0((__int64)&v125, v127, v22, 8u, v22, v14);
          v21 = (unsigned int)v126;
        }
        if ( 8 * v96 )
        {
          memset(&v125[8 * v21], 0, 8 * v96);
          LODWORD(v21) = v126;
        }
        LOBYTE(v20) = v128;
        LODWORD(v126) = v96 + v21;
      }
      else
      {
        LODWORD(v126) = (unsigned int)(v20 + 63) >> 6;
      }
    }
    v23 = v20 & 0x3F;
    if ( v23 )
      *(_QWORD *)&v125[8 * (unsigned int)v126 - 8] &= ~(-1LL << v23);
    v24 = 0;
    v25 = *((unsigned int *)v5 + 28);
    if ( (_DWORD)v25 )
    {
      do
      {
        *(_QWORD *)&v125[v24] = *(_QWORD *)(v5[13] + v24) & ~*(_QWORD *)(v119[4] + v24);
        v24 += 8;
      }
      while ( 8 * v25 != v24 );
    }
    v26 = v128;
    if ( (v128 & 0x3F) != 0 )
    {
      *(_QWORD *)&v125[8 * (unsigned int)v126 - 8] &= ~(-1LL << (v128 & 0x3F));
      v26 = v128;
    }
    if ( v26 )
    {
      v27 = 0;
      v28 = (unsigned int)(v26 - 1) >> 6;
      v14 = v28 + 1;
      while ( 1 )
      {
        _RDX = *(_QWORD *)&v125[8 * v27];
        if ( v28 == (_DWORD)v27 )
          _RDX = (0xFFFFFFFFFFFFFFFFLL >> -(char)v26) & *(_QWORD *)&v125[8 * v27];
        if ( _RDX )
          break;
        if ( v14 == ++v27 )
          goto LABEL_55;
      }
      __asm { tzcnt   rdx, rdx }
      if ( (_DWORD)_RDX + ((_DWORD)v27 << 6) != -1 )
      {
        v105 = v5;
        v43 = _RDX + ((_DWORD)v27 << 6);
        while ( 1 )
        {
          v129.m128i_i64[0] = 0;
          v132 = 11;
          v133 = 0;
          v134 = 0;
          v135 = 0;
          v136 = 0;
          v137 = v139;
          v138 = 0;
          v139[0] = 0;
          v129.m128i_i32[2] = v43;
          v130 = 0;
          v108 = sub_2E7A450((_QWORD *)a2, (__int64)&v129, _RDX);
          if ( v137 != (_QWORD *)v139 )
            j_j___libc_free_0((unsigned __int64)v137);
          if ( v134 )
            j_j___libc_free_0(v134);
          v44 = *(_QWORD *)(v115 + 8);
          v121 = v120;
          v45 = v44 - 120;
          if ( !v120 )
            break;
          sub_B96E90((__int64)&v121, (__int64)v120, 1);
          v122 = v121;
          if ( !v121 )
            goto LABEL_111;
          sub_B976B0((__int64)&v121, v121, (__int64)&v122);
          v121 = 0;
          v123 = 0;
          v124 = 0;
          v46 = *(_QWORD **)(*v119 + 32);
          v111 = *v119;
          v129.m128i_i64[0] = (__int64)v122;
          if ( v122 )
            sub_B96E90((__int64)&v129, (__int64)v122, 1);
LABEL_87:
          v47 = (__int64)sub_2E7B380(v46, v45, (unsigned __int8 **)&v129, 0);
          if ( v129.m128i_i64[0] )
            sub_B91220((__int64)&v129, v129.m128i_i64[0]);
          sub_2E31040((__int64 *)(v111 + 40), v47);
          v48 = *v114;
          v49 = *(_QWORD *)v47;
          *(_QWORD *)(v47 + 8) = v114;
          v48 &= 0xFFFFFFFFFFFFFFF8LL;
          *(_QWORD *)v47 = v48 | v49 & 7;
          *(_QWORD *)(v48 + 8) = v47;
          *v114 = v47 | *v114 & 7;
          if ( v123 )
            sub_2E882B0(v47, (__int64)v46, v123);
          if ( v124 )
            sub_2E88680(v47, (__int64)v46, v124);
          v129.m128i_i64[0] = 16;
          v130 = 0;
          v131 = v108;
          sub_2E8EAD0(v47, (__int64)v46, &v129);
          if ( v122 )
            sub_B91220((__int64)&v122, (__int64)v122);
          if ( v121 )
            sub_B91220((__int64)&v121, (__int64)v121);
          LOBYTE(v26) = v128;
          v50 = v43 + 1;
          if ( v128 != v43 + 1 )
          {
            v14 = v50 >> 6;
            v51 = (unsigned int)(v128 - 1) >> 6;
            if ( (unsigned int)v14 <= v51 )
            {
              _RDX = (unsigned int)v14;
              v52 = 64 - (v50 & 0x3F);
              v53 = 0xFFFFFFFFFFFFFFFFLL >> v52;
              if ( v52 == 64 )
                v53 = 0;
              v54 = ~v53;
              while ( 1 )
              {
                _RAX = *(_QWORD *)&v125[8 * _RDX];
                if ( (_DWORD)v14 == (_DWORD)_RDX )
                  _RAX = v54 & *(_QWORD *)&v125[8 * _RDX];
                if ( v51 == (_DWORD)_RDX )
                  _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v128;
                if ( _RAX )
                  break;
                if ( v51 < (unsigned int)++_RDX )
                  goto LABEL_109;
              }
              __asm { tzcnt   rax, rax }
              v43 = ((_DWORD)_RDX << 6) + _RAX;
              if ( v43 != -1 )
                continue;
            }
          }
LABEL_109:
          v5 = v105;
          v6 = 1;
          goto LABEL_55;
        }
        v122 = 0;
LABEL_111:
        v123 = 0;
        v124 = 0;
        v46 = *(_QWORD **)(*v119 + 32);
        v111 = *v119;
        v129.m128i_i64[0] = 0;
        goto LABEL_87;
      }
LABEL_55:
      v30 = *((_DWORD *)v119 + 24);
      v31 = v26 & 0x3F;
      if ( v31 )
        *(_QWORD *)&v125[8 * (unsigned int)v126 - 8] &= ~(-1LL << v31);
    }
    else
    {
      v30 = *((_DWORD *)v119 + 24);
    }
    v32 = (unsigned int)v126;
    v128 = v30;
    v33 = (unsigned int)(v30 + 63) >> 6;
    if ( v33 != (unsigned int)v126 )
    {
      if ( v33 >= (unsigned int)v126 )
      {
        v97 = v33 - (unsigned int)v126;
        if ( v33 > HIDWORD(v126) )
        {
          sub_C8D5F0((__int64)&v125, v127, v33, 8u, v33, v14);
          v32 = (unsigned int)v126;
        }
        if ( 8 * v97 )
        {
          memset(&v125[8 * v32], 0, 8 * v97);
          LODWORD(v32) = v126;
        }
        LOBYTE(v30) = v128;
        LODWORD(v126) = v97 + v32;
      }
      else
      {
        LODWORD(v126) = (unsigned int)(v30 + 63) >> 6;
      }
    }
    v34 = v30 & 0x3F;
    if ( v34 )
      *(_QWORD *)&v125[8 * (unsigned int)v126 - 8] &= ~(-1LL << v34);
    v35 = 0;
    v36 = *((unsigned int *)v119 + 10);
    if ( (_DWORD)v36 )
    {
      do
      {
        *(_QWORD *)&v125[v35] = *(_QWORD *)(v119[4] + v35) & ~*(_QWORD *)(v5[13] + v35);
        v35 += 8;
      }
      while ( 8 * v36 != v35 );
    }
    v37 = v128;
    if ( (v128 & 0x3F) != 0 )
    {
      *(_QWORD *)&v125[8 * (unsigned int)v126 - 8] &= ~(-1LL << (v128 & 0x3F));
      v37 = v128;
    }
    if ( v37 )
    {
      v38 = -v37;
      v39 = (unsigned int)(v37 - 1) >> 6;
      v40 = 0;
      while ( 1 )
      {
        _RDX = *(_QWORD *)&v125[8 * v40];
        if ( v39 == (_DWORD)v40 )
          _RDX = (0xFFFFFFFFFFFFFFFFLL >> v38) & *(_QWORD *)&v125[8 * v40];
        if ( _RDX )
          break;
        if ( v39 + 1 == ++v40 )
          goto LABEL_72;
      }
      __asm { tzcnt   rdx, rdx }
      v58 = _RDX + ((_DWORD)v40 << 6);
      if ( v58 != -1 )
      {
        while ( 1 )
        {
          v59 = *(_BYTE *)(a1 + 232) & 1;
          if ( v59 )
          {
            v60 = a1 + 240;
            v61 = 15;
          }
          else
          {
            v60 = *(_QWORD *)(a1 + 240);
            v83 = *(unsigned int *)(a1 + 248);
            if ( !(_DWORD)v83 )
              goto LABEL_157;
            v61 = v83 - 1;
          }
          v62 = v61 & (37 * v58);
          v63 = (int *)(v60 + 20LL * v62);
          v64 = *v63;
          if ( v58 == *v63 )
            goto LABEL_116;
          v93 = 1;
          while ( v64 != -1 )
          {
            v104 = v93 + 1;
            v62 = v61 & (v93 + v62);
            v63 = (int *)(v60 + 20LL * v62);
            v64 = *v63;
            if ( v58 == *v63 )
              goto LABEL_116;
            v93 = v104;
          }
          if ( v59 )
          {
            v84 = 320;
            goto LABEL_158;
          }
          v83 = *(unsigned int *)(a1 + 248);
LABEL_157:
          v84 = 20 * v83;
LABEL_158:
          v63 = (int *)(v60 + v84);
LABEL_116:
          v65 = v63[1];
          v66 = *((unsigned __int8 *)v63 + 16);
          if ( *((_BYTE *)v63 + 8) )
          {
            if ( (_BYTE)v66 )
LABEL_219:
              BUG();
            v132 = 13;
            v129.m128i_i64[0] = 0;
            v133 = 0;
            v134 = 0;
            v135 = 0;
            v136 = 0;
            v137 = v139;
            v138 = 0;
            v139[0] = 0;
            v129.m128i_i64[1] = __PAIR64__(v65, v58);
          }
          else
          {
            if ( !(_BYTE)v66 )
              goto LABEL_219;
            v67 = v63[3];
            v132 = 3;
            v129.m128i_i64[0] = 0;
            v133 = 0;
            v134 = 0;
            v135 = 0;
            v136 = 0;
            v137 = v139;
            v138 = 0;
            v139[0] = 0;
            v129.m128i_i32[2] = v58;
            v130 = v67;
          }
          v109 = sub_2E7A450((_QWORD *)a2, (__int64)&v129, v66);
          if ( v137 != (_QWORD *)v139 )
            j_j___libc_free_0((unsigned __int64)v137);
          if ( v134 )
            j_j___libc_free_0(v134);
          v68 = *(_QWORD *)(v115 + 8);
          v121 = v120;
          v69 = v68 - 120;
          if ( !v120 )
          {
            v122 = 0;
LABEL_155:
            v123 = 0;
            v124 = 0;
            v70 = *v119;
            v71 = *(_QWORD **)(*v119 + 32);
            v129.m128i_i64[0] = 0;
            goto LABEL_127;
          }
          sub_B96E90((__int64)&v121, (__int64)v120, 1);
          v122 = v121;
          if ( !v121 )
            goto LABEL_155;
          sub_B976B0((__int64)&v121, v121, (__int64)&v122);
          v121 = 0;
          v123 = 0;
          v124 = 0;
          v70 = *v119;
          v71 = *(_QWORD **)(*v119 + 32);
          v129.m128i_i64[0] = (__int64)v122;
          if ( v122 )
            sub_B96E90((__int64)&v129, (__int64)v122, 1);
LABEL_127:
          v72 = (__int64)sub_2E7B380(v71, v69, (unsigned __int8 **)&v129, 0);
          if ( v129.m128i_i64[0] )
            sub_B91220((__int64)&v129, v129.m128i_i64[0]);
          sub_2E31040((__int64 *)(v70 + 40), v72);
          v73 = *v114;
          *(_QWORD *)(v72 + 8) = v114;
          *(_QWORD *)v72 = v73 & 0xFFFFFFFFFFFFFFF8LL | *(_QWORD *)v72 & 7LL;
          *(_QWORD *)((v73 & 0xFFFFFFFFFFFFFFF8LL) + 8) = v72;
          *v114 = v72 | *v114 & 7;
          if ( v123 )
            sub_2E882B0(v72, (__int64)v71, v123);
          if ( v124 )
            sub_2E88680(v72, (__int64)v71, v124);
          v129.m128i_i64[0] = 16;
          v130 = 0;
          v131 = v109;
          sub_2E8EAD0(v72, (__int64)v71, &v129);
          if ( v122 )
            sub_B91220((__int64)&v122, (__int64)v122);
          if ( v121 )
            sub_B91220((__int64)&v121, (__int64)v121);
          v74 = v58 + 1;
          if ( v128 != v74 )
          {
            v75 = v74 >> 6;
            v76 = (unsigned int)(v128 - 1) >> 6;
            if ( v74 >> 6 <= v76 )
            {
              v77 = v74 & 0x3F;
              v78 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v77);
              if ( v77 == 0 )
                v78 = 0;
              v79 = v75;
              v80 = ~v78;
              while ( 1 )
              {
                _RAX = *(_QWORD *)&v125[8 * v79];
                if ( v75 == (_DWORD)v79 )
                  _RAX = v80 & *(_QWORD *)&v125[8 * v79];
                if ( v76 == (_DWORD)v79 )
                  _RAX &= 0xFFFFFFFFFFFFFFFFLL >> -(char)v128;
                if ( _RAX )
                  break;
                if ( v76 < (unsigned int)++v79 )
                  goto LABEL_149;
              }
              __asm { tzcnt   rax, rax }
              v58 = _RAX + ((_DWORD)v79 << 6);
              if ( v58 != -1 )
                continue;
            }
          }
LABEL_149:
          v6 = 1;
          break;
        }
      }
    }
LABEL_72:
    if ( v120 )
      sub_B91220((__int64)&v120, (__int64)v120);
    v5 = v119;
LABEL_32:
    v117 = *(_QWORD *)(v117 + 8);
    if ( v106 == v117 )
      break;
    v4 = *(_QWORD *)(a2 + 328);
  }
  v94 = v6;
  if ( v125 != v127 )
    _libc_free((unsigned __int64)v125);
  return v94;
}
