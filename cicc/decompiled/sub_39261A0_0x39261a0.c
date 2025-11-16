// Function: sub_39261A0
// Address: 0x39261a0
//
__int64 __fastcall sub_39261A0(__int64 a1, _QWORD *a2, __int64 **a3)
{
  __int64 v4; // rbx
  __int64 *v5; // r12
  __int64 v6; // rdx
  int v7; // ecx
  unsigned int v8; // eax
  int *v9; // rdx
  unsigned int v10; // esi
  __int64 v11; // rdi
  unsigned int v12; // ecx
  _QWORD *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r13
  const void *v16; // r14
  __int64 v17; // rax
  unsigned __int64 v18; // rbx
  const void *v19; // r8
  void *v20; // rdi
  unsigned __int64 *v21; // r14
  __int64 v22; // rbx
  __int64 v23; // rax
  int v24; // r8d
  int v25; // r9d
  __int64 v26; // r14
  _BYTE *v27; // rsi
  __int64 v28; // rax
  unsigned __int64 v29; // rax
  __int64 v30; // rax
  unsigned __int64 v31; // rdi
  unsigned __int64 v32; // rdi
  __int64 v33; // rax
  __int64 v34; // rsi
  __int64 v35; // rax
  __int64 v36; // rdx
  bool v37; // cf
  unsigned __int64 v38; // rax
  __int64 v39; // rdx
  unsigned __int64 *v40; // rsi
  unsigned __int64 *v41; // rbx
  _QWORD *v42; // r12
  __int64 v43; // rcx
  unsigned __int64 v44; // r13
  unsigned __int64 v45; // rdi
  unsigned __int64 v46; // rdi
  _QWORD *v47; // rdx
  unsigned __int64 v48; // rdi
  __int64 result; // rax
  __int64 *v50; // r13
  __int64 *v51; // r14
  __int64 v52; // r12
  __int64 v53; // r15
  __int64 *v54; // rax
  __int64 *v55; // rcx
  unsigned __int64 v56; // rax
  void *v57; // r9
  char v58; // al
  bool v59; // zf
  int v60; // eax
  char v61; // al
  __int64 v62; // rax
  __int64 v63; // rsi
  __int64 v64; // rax
  int v65; // r8d
  unsigned __int64 v66; // rax
  __int64 v67; // rdx
  unsigned __int64 v68; // rax
  __int64 v69; // rdx
  _QWORD *v70; // rax
  __int64 v71; // rax
  int v72; // r8d
  _QWORD *v73; // r11
  int v74; // edi
  int v75; // edx
  unsigned __int64 v76; // rdx
  __int64 v77; // rax
  unsigned int v78; // esi
  __int64 v79; // rdx
  __int64 v80; // r11
  __int64 v81; // rdi
  char *v82; // rax
  __int64 v83; // r9
  __int64 *v84; // rax
  __int64 v85; // rdi
  unsigned __int64 v86; // rax
  int v87; // esi
  int v88; // esi
  __int64 v89; // r9
  __int64 v90; // rcx
  __int64 v91; // rdi
  int v92; // r14d
  _QWORD *v93; // r10
  int v94; // ecx
  int v95; // ecx
  __int64 v96; // rdi
  _QWORD *v97; // r9
  __int64 v98; // r14
  int v99; // r11d
  __int64 v100; // rsi
  __int64 *v101; // rax
  __int64 v102; // rax
  char *v103; // rcx
  int v104; // eax
  int v105; // eax
  int v106; // eax
  int v107; // r11d
  __int64 v108; // r8
  unsigned int v109; // esi
  __int64 v110; // r10
  int v111; // r9d
  char *v112; // rdi
  int v113; // eax
  int v114; // r11d
  int v115; // r9d
  unsigned int v116; // esi
  __int64 v117; // r10
  unsigned __int64 v118; // rdx
  __int64 *v120; // [rsp+10h] [rbp-E0h]
  unsigned __int64 v121; // [rsp+18h] [rbp-D8h]
  __int64 v122; // [rsp+20h] [rbp-D0h]
  char *v123; // [rsp+28h] [rbp-C8h]
  unsigned __int64 v124; // [rsp+28h] [rbp-C8h]
  unsigned __int64 *v125; // [rsp+30h] [rbp-C0h]
  __int64 v126; // [rsp+30h] [rbp-C0h]
  __int64 *v127; // [rsp+30h] [rbp-C0h]
  unsigned int v128; // [rsp+30h] [rbp-C0h]
  __int64 *v130; // [rsp+40h] [rbp-B0h]
  __int64 *v131; // [rsp+40h] [rbp-B0h]
  int v132; // [rsp+40h] [rbp-B0h]
  __int64 v133; // [rsp+40h] [rbp-B0h]
  __int64 v134; // [rsp+40h] [rbp-B0h]
  _BYTE *src; // [rsp+48h] [rbp-A8h]
  _QWORD *srca; // [rsp+48h] [rbp-A8h]
  void *srcf; // [rsp+48h] [rbp-A8h]
  __int64 *srcg; // [rsp+48h] [rbp-A8h]
  __int64 *srcb; // [rsp+48h] [rbp-A8h]
  void *srch; // [rsp+48h] [rbp-A8h]
  __int64 *srcc; // [rsp+48h] [rbp-A8h]
  void *srcd; // [rsp+48h] [rbp-A8h]
  void *srci; // [rsp+48h] [rbp-A8h]
  void *srcj; // [rsp+48h] [rbp-A8h]
  char *srce; // [rsp+48h] [rbp-A8h]
  _QWORD v146[2]; // [rsp+50h] [rbp-A0h] BYREF
  _QWORD v147[2]; // [rsp+60h] [rbp-90h] BYREF
  __int16 v148; // [rsp+70h] [rbp-80h]
  _QWORD v149[2]; // [rsp+80h] [rbp-70h] BYREF
  __int16 v150; // [rsp+90h] [rbp-60h]
  const void *v151[2]; // [rsp+A0h] [rbp-50h] BYREF
  __int64 v152; // [rsp+B0h] [rbp-40h] BYREF

  v4 = a1;
  v5 = (__int64 *)a2[4];
  v130 = (__int64 *)a2[5];
  if ( v5 != v130 )
  {
    while ( 1 )
    {
      v15 = *v5;
      v16 = *(const void **)(*v5 + 160);
      src = *(_BYTE **)(*v5 + 152);
      v17 = sub_22077B0(0x78u);
      v18 = v17;
      if ( !v17 )
        goto LABEL_21;
      v19 = src;
      v20 = (void *)(v17 + 56);
      *(_QWORD *)(v17 + 32) = 0;
      *(_QWORD *)(v17 + 40) = v17 + 56;
      *(_OWORD *)v17 = 0;
      *(_OWORD *)(v17 + 16) = 0;
      if ( !src )
      {
        *(_QWORD *)(v17 + 48) = 0;
        *(_BYTE *)(v17 + 56) = 0;
        goto LABEL_20;
      }
      v151[0] = v16;
      if ( (unsigned __int64)v16 > 0xF )
        break;
      if ( v16 == (const void *)1 )
      {
        *(_BYTE *)(v17 + 56) = *src;
      }
      else if ( v16 )
      {
        goto LABEL_60;
      }
LABEL_19:
      *(_QWORD *)(v18 + 48) = v16;
      *((_BYTE *)v16 + (_QWORD)v20) = 0;
LABEL_20:
      *(_QWORD *)(v18 + 80) = 0;
      *(_QWORD *)(v18 + 88) = 0;
      *(_QWORD *)(v18 + 96) = 0;
      *(_QWORD *)(v18 + 104) = 0;
      *(_QWORD *)(v18 + 112) = 0;
LABEL_21:
      v21 = *(unsigned __int64 **)(a1 + 64);
      if ( v21 == *(unsigned __int64 **)(a1 + 72) )
      {
        v34 = (__int64)v21 - *(_QWORD *)(a1 + 56);
        v125 = *(unsigned __int64 **)(a1 + 56);
        v35 = v34 >> 3;
        if ( v34 >> 3 == 0xFFFFFFFFFFFFFFFLL )
          sub_4262D8((__int64)"vector::_M_realloc_insert");
        v36 = 1;
        if ( v35 )
          v36 = v34 >> 3;
        v37 = __CFADD__(v36, v35);
        v38 = v36 + v35;
        if ( v37 )
        {
          v118 = 0x7FFFFFFFFFFFFFF8LL;
        }
        else
        {
          if ( !v38 )
          {
            v123 = 0;
            v39 = 8;
            srca = 0;
            goto LABEL_67;
          }
          if ( v38 > 0xFFFFFFFFFFFFFFFLL )
            v38 = 0xFFFFFFFFFFFFFFFLL;
          v118 = 8 * v38;
        }
        v124 = v118;
        srca = (_QWORD *)sub_22077B0(v118);
        v123 = (char *)srca + v124;
        v39 = (__int64)(srca + 1);
LABEL_67:
        v40 = (_QWORD *)((char *)srca + v34);
        if ( v40 )
        {
          *v40 = v18;
          v18 = 0;
        }
        if ( v21 == v125 )
        {
LABEL_82:
          v48 = (unsigned __int64)v125;
          if ( v125 )
          {
            v126 = v39;
            j_j___libc_free_0(v48);
            v39 = v126;
          }
          *(_QWORD *)(a1 + 64) = v39;
          *(_QWORD *)(a1 + 56) = srca;
          *(_QWORD *)(a1 + 72) = v123;
          goto LABEL_51;
        }
        v121 = v18;
        v41 = v125;
        v120 = v5;
        v122 = v15;
        v42 = srca;
        while ( 2 )
        {
          v44 = *v41;
          if ( v42 )
          {
            *v42 = v44;
            *v41 = 0;
          }
          else if ( v44 )
          {
            v45 = *(_QWORD *)(v44 + 96);
            if ( v45 )
              j_j___libc_free_0(v45);
            v46 = *(_QWORD *)(v44 + 40);
            if ( v46 != v44 + 56 )
              j_j___libc_free_0(v46);
            ++v41;
            j_j___libc_free_0(v44);
            v43 = 8;
            if ( v21 == v41 )
            {
LABEL_81:
              v47 = v42;
              v15 = v122;
              v18 = v121;
              v5 = v120;
              v39 = (__int64)(v47 + 2);
              goto LABEL_82;
            }
            goto LABEL_73;
          }
          ++v41;
          v43 = (__int64)(v42 + 1);
          if ( v21 == v41 )
            goto LABEL_81;
LABEL_73:
          v42 = (_QWORD *)v43;
          continue;
        }
      }
      if ( v21 )
      {
        *v21 = v18;
        *(_QWORD *)(a1 + 64) += 8LL;
        goto LABEL_24;
      }
      *(_QWORD *)(a1 + 64) = 8;
LABEL_51:
      if ( v18 )
      {
        v31 = *(_QWORD *)(v18 + 96);
        if ( v31 )
          j_j___libc_free_0(v31);
        v32 = *(_QWORD *)(v18 + 40);
        if ( v32 != v18 + 56 )
          j_j___libc_free_0(v32);
        j_j___libc_free_0(v18);
      }
LABEL_24:
      v22 = *(_QWORD *)(*(_QWORD *)(a1 + 64) - 8LL);
      v23 = sub_3925260((_QWORD *)a1, *(const void **)(v15 + 152), *(_QWORD *)(v15 + 160));
      *(_QWORD *)(v22 + 88) = v23;
      v26 = v23;
      *(_QWORD *)(v23 + 104) = v22;
      *(_BYTE *)(v23 + 18) = 3;
      if ( *(_DWORD *)(v15 + 184) != 5 )
      {
        v27 = *(_BYTE **)(v15 + 176);
        if ( v27 )
        {
          v28 = sub_3925EE0(a1, v27);
          if ( *(_QWORD *)(v28 + 104) )
            sub_16BD130("two sections have the same comdat", 1u);
          *(_QWORD *)(v28 + 104) = v22;
        }
      }
      v29 = *(unsigned int *)(v26 + 64);
      if ( v29 > 1 )
      {
        *(_DWORD *)(v26 + 64) = 1;
        v6 = *(_QWORD *)(v26 + 56);
      }
      else if ( *(_DWORD *)(v26 + 64) )
      {
        v6 = *(_QWORD *)(v26 + 56);
      }
      else
      {
        if ( !*(_DWORD *)(v26 + 68) )
        {
          sub_16CD150(v26 + 56, (const void *)(v26 + 72), 1u, 24, v24, v25);
          v29 = *(unsigned int *)(v26 + 64);
        }
        v6 = *(_QWORD *)(v26 + 56);
        v30 = v6 + 24 * v29;
        if ( v30 != v6 + 24 )
        {
          do
          {
            if ( v30 )
            {
              *(_QWORD *)(v30 + 16) = 0;
              *(_OWORD *)v30 = 0;
            }
            v30 += 24;
          }
          while ( v6 + 24 != v30 );
          v6 = *(_QWORD *)(v26 + 56);
        }
        *(_DWORD *)(v26 + 64) = 1;
      }
      *(_QWORD *)(v6 + 16) = 0;
      *(_OWORD *)v6 = 0;
      **(_DWORD **)(v26 + 56) = 4;
      *(_BYTE *)(*(_QWORD *)(v26 + 56) + 20LL) = *(_DWORD *)(v15 + 184);
      v7 = *(_DWORD *)(v15 + 168);
      *(_DWORD *)(v22 + 36) = v7;
      v8 = *(_DWORD *)(v15 + 24);
      if ( v8 == 64 )
      {
        LODWORD(v9) = 7340032;
      }
      else if ( v8 <= 0x40 )
      {
        switch ( *(_DWORD *)(v15 + 24) )
        {
          case 0:
          case 3:
          case 5:
          case 6:
          case 7:
          case 9:
          case 0xA:
          case 0xB:
          case 0xC:
          case 0xD:
          case 0xE:
          case 0xF:
          case 0x11:
          case 0x12:
          case 0x13:
          case 0x14:
          case 0x15:
          case 0x16:
          case 0x17:
          case 0x18:
          case 0x19:
          case 0x1A:
          case 0x1B:
          case 0x1C:
          case 0x1D:
          case 0x1E:
          case 0x1F:
          case 0x20:
            LODWORD(v9) = 6291456;
            break;
          case 1:
            LODWORD(v9) = 0x100000;
            break;
          case 2:
            LODWORD(v9) = 0x200000;
            break;
          case 4:
            LODWORD(v9) = 3145728;
            break;
          case 8:
            v9 = &dword_400000;
            break;
          case 0x10:
            LODWORD(v9) = 5242880;
            break;
        }
      }
      else if ( v8 == 1024 )
      {
        LODWORD(v9) = 11534336;
      }
      else if ( v8 <= 0x400 )
      {
        LODWORD(v9) = 9437184;
        if ( v8 != 256 )
        {
          LODWORD(v9) = 0x800000;
          if ( v8 == 512 )
            LODWORD(v9) = (unsigned int)&loc_A00000;
        }
      }
      else
      {
        LODWORD(v9) = 13631488;
        if ( v8 != 4096 )
        {
          LODWORD(v9) = 12582912;
          if ( v8 == 0x2000 )
            LODWORD(v9) = (unsigned int)&loc_E00000;
        }
      }
      *(_QWORD *)(v22 + 80) = v15;
      *(_DWORD *)(v22 + 36) = v7 | (unsigned int)v9;
      v10 = *(_DWORD *)(a1 + 184);
      if ( !v10 )
      {
        ++*(_QWORD *)(a1 + 160);
        goto LABEL_155;
      }
      v11 = *(_QWORD *)(a1 + 168);
      v12 = (v10 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v13 = (_QWORD *)(v11 + 16LL * v12);
      v14 = *v13;
      if ( v15 != *v13 )
      {
        v72 = 1;
        v73 = 0;
        while ( v14 != -8 )
        {
          if ( !v73 && v14 == -16 )
            v73 = v13;
          v12 = (v10 - 1) & (v72 + v12);
          v13 = (_QWORD *)(v11 + 16LL * v12);
          v14 = *v13;
          if ( v15 == *v13 )
            goto LABEL_13;
          ++v72;
        }
        v74 = *(_DWORD *)(a1 + 176);
        if ( v73 )
          v13 = v73;
        ++*(_QWORD *)(a1 + 160);
        v75 = v74 + 1;
        if ( 4 * (v74 + 1) < 3 * v10 )
        {
          if ( v10 - *(_DWORD *)(a1 + 180) - v75 <= v10 >> 3 )
          {
            sub_3925B80(a1 + 160, v10);
            v94 = *(_DWORD *)(a1 + 184);
            if ( !v94 )
            {
LABEL_239:
              ++*(_DWORD *)(a1 + 176);
              BUG();
            }
            v95 = v94 - 1;
            v96 = *(_QWORD *)(a1 + 168);
            v97 = 0;
            LODWORD(v98) = v95 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
            v99 = 1;
            v75 = *(_DWORD *)(a1 + 176) + 1;
            v13 = (_QWORD *)(v96 + 16LL * (unsigned int)v98);
            v100 = *v13;
            if ( v15 != *v13 )
            {
              while ( v100 != -8 )
              {
                if ( !v97 && v100 == -16 )
                  v97 = v13;
                v98 = v95 & (unsigned int)(v98 + v99);
                v13 = (_QWORD *)(v96 + 16 * v98);
                v100 = *v13;
                if ( v15 == *v13 )
                  goto LABEL_135;
                ++v99;
              }
              if ( v97 )
                v13 = v97;
            }
          }
          goto LABEL_135;
        }
LABEL_155:
        sub_3925B80(a1 + 160, 2 * v10);
        v87 = *(_DWORD *)(a1 + 184);
        if ( !v87 )
          goto LABEL_239;
        v88 = v87 - 1;
        v89 = *(_QWORD *)(a1 + 168);
        LODWORD(v90) = v88 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
        v75 = *(_DWORD *)(a1 + 176) + 1;
        v13 = (_QWORD *)(v89 + 16LL * (unsigned int)v90);
        v91 = *v13;
        if ( v15 != *v13 )
        {
          v92 = 1;
          v93 = 0;
          while ( v91 != -8 )
          {
            if ( v91 == -16 && !v93 )
              v93 = v13;
            v90 = v88 & (unsigned int)(v90 + v92);
            v13 = (_QWORD *)(v89 + 16 * v90);
            v91 = *v13;
            if ( v15 == *v13 )
              goto LABEL_135;
            ++v92;
          }
          if ( v93 )
            v13 = v93;
        }
LABEL_135:
        *(_DWORD *)(a1 + 176) = v75;
        if ( *v13 != -8 )
          --*(_DWORD *)(a1 + 180);
        *v13 = v15;
        v13[1] = 0;
      }
LABEL_13:
      v13[1] = v22;
      if ( v130 == ++v5 )
      {
        v4 = a1;
        goto LABEL_87;
      }
    }
    v33 = sub_22409D0(v17 + 40, (unsigned __int64 *)v151, 0);
    v19 = src;
    *(_QWORD *)(v18 + 40) = v33;
    v20 = (void *)v33;
    *(const void **)(v18 + 56) = v151[0];
LABEL_60:
    memcpy(v20, v19, (size_t)v16);
    v16 = v151[0];
    v20 = *(void **)(v18 + 40);
    goto LABEL_19;
  }
LABEL_87:
  result = (__int64)a2;
  v50 = (__int64 *)a2[7];
  v51 = (__int64 *)a2[8];
  if ( v51 != v50 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v52 = *v50;
        if ( (*(_BYTE *)(*v50 + 8) & 1) == 0 )
          break;
        if ( v51 == ++v50 )
          return result;
      }
      v53 = sub_3925EE0(v4, (_BYTE *)*v50);
      v54 = (__int64 *)sub_38CF550(a3, v52);
      v55 = v54;
      if ( v54 )
        break;
      if ( (*(_BYTE *)(v52 + 13) & 1) == 0 )
      {
        *(_DWORD *)(v53 + 12) = -1;
        v57 = (void *)v53;
LABEL_97:
        if ( (*(_BYTE *)(v52 + 9) & 0xC) == 0xC && (*(_BYTE *)(v52 + 8) & 0x10) != 0 )
        {
          v60 = *(_DWORD *)(v52 + 24);
        }
        else
        {
          srcf = v57;
          v58 = sub_38D0480(a3, v52, v151);
          v57 = srcf;
          v59 = v58 == 0;
          v60 = 0;
          if ( !v59 )
            v60 = (int)v151[0];
        }
        *((_DWORD *)v57 + 2) = v60;
        *((_WORD *)v57 + 8) = *(_WORD *)(v52 + 32);
        result = *(unsigned __int16 *)(v52 + 12);
        *((_BYTE *)v57 + 18) = result;
        if ( !(_BYTE)result )
        {
          v61 = *(_BYTE *)(v52 + 8);
          if ( (v61 & 0x10) == 0
            && ((*(_QWORD *)v52 & 0xFFFFFFFFFFFFFFF8LL) != 0
             || (*(_BYTE *)(v52 + 9) & 0xC) == 8
             && ((v85 = *(_QWORD *)(v52 + 24),
                  srci = v57,
                  *(_BYTE *)(v52 + 8) = v61 | 4,
                  v86 = (unsigned __int64)sub_38CE440(v85),
                  v57 = srci,
                  *(_QWORD *)v52 = v86 | *(_QWORD *)v52 & 7LL,
                  v86)
              || (*(_BYTE *)(v52 + 9) & 0xC) == 8)) )
          {
            result = 3;
          }
          else
          {
            result = 2;
          }
          *((_BYTE *)v57 + 18) = result;
        }
        goto LABEL_105;
      }
LABEL_108:
      *(_BYTE *)(v53 + 18) = 105;
      if ( (*(_BYTE *)(v52 + 9) & 0xC) != 8 )
        goto LABEL_118;
      v62 = *(_QWORD *)(v52 + 24);
      *(_BYTE *)(v52 + 8) |= 4u;
      if ( *(_DWORD *)v62 != 2 )
        goto LABEL_118;
      v63 = *(_QWORD *)(v62 + 24);
      if ( (*(_QWORD *)v63 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        goto LABEL_118;
      if ( (*(_BYTE *)(v63 + 9) & 0xC) == 8 )
      {
        *(_BYTE *)(v63 + 8) |= 4u;
        v131 = v55;
        v68 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v63 + 24));
        v55 = v131;
        *(_QWORD *)v63 = v68 | *(_QWORD *)v63 & 7LL;
        if ( v68 )
          goto LABEL_118;
      }
      srcg = v55;
      v64 = sub_3925EE0(v4, (_BYTE *)v63);
      v55 = srcg;
      if ( !v64 )
      {
LABEL_118:
        if ( (*(_BYTE *)v52 & 4) != 0 )
        {
          v101 = *(__int64 **)(v52 - 8);
          v69 = *v101;
          v70 = v101 + 2;
        }
        else
        {
          v69 = 0;
          v70 = 0;
        }
        v146[0] = v70;
        v149[0] = ".weak.";
        v149[1] = v146;
        srcb = v55;
        v147[0] = v149;
        v148 = 770;
        v150 = 1283;
        v146[1] = v69;
        v147[1] = ".default";
        sub_16E2FC0((__int64 *)v151, (__int64)v147);
        v71 = sub_3925260((_QWORD *)v4, v151[0], (unsigned __int64)v151[1]);
        v57 = (void *)v71;
        if ( srcb )
          *(_QWORD *)(v71 + 104) = srcb;
        else
          *(_DWORD *)(v71 + 12) = -1;
        if ( v151[0] != &v152 )
        {
          srch = (void *)v71;
          j_j___libc_free_0((unsigned __int64)v151[0]);
          v57 = srch;
        }
        *(_QWORD *)(v53 + 96) = v57;
        v66 = *(unsigned int *)(v53 + 64);
        if ( v66 > 1 )
        {
LABEL_114:
          *(_DWORD *)(v53 + 64) = 1;
          v67 = *(_QWORD *)(v53 + 56);
          goto LABEL_115;
        }
      }
      else
      {
        *(_QWORD *)(v53 + 96) = v64;
        v66 = *(unsigned int *)(v53 + 64);
        v57 = 0;
        if ( v66 > 1 )
          goto LABEL_114;
      }
      if ( v66 )
      {
        v67 = *(_QWORD *)(v53 + 56);
      }
      else
      {
        if ( !*(_DWORD *)(v53 + 68) )
        {
          srcj = v57;
          sub_16CD150(v53 + 56, (const void *)(v53 + 72), 1u, 24, v65, (int)v57);
          v66 = *(unsigned int *)(v53 + 64);
          v57 = srcj;
        }
        v67 = *(_QWORD *)(v53 + 56);
        v102 = v67 + 24 * v66;
        if ( v102 != v67 + 24 )
        {
          do
          {
            if ( v102 )
            {
              *(_QWORD *)(v102 + 16) = 0;
              *(_OWORD *)v102 = 0;
            }
            v102 += 24;
          }
          while ( v67 + 24 != v102 );
          v67 = *(_QWORD *)(v53 + 56);
        }
        *(_DWORD *)(v53 + 64) = 1;
      }
LABEL_115:
      *(_QWORD *)(v67 + 16) = 0;
      *(_OWORD *)v67 = 0;
      **(_DWORD **)(v53 + 56) = 2;
      *(_DWORD *)(*(_QWORD *)(v53 + 56) + 4LL) = 0;
      result = *(_QWORD *)(v53 + 56);
      *(_DWORD *)(result + 8) = 2;
      if ( v57 )
        goto LABEL_97;
LABEL_105:
      ++v50;
      *(_QWORD *)(v53 + 120) = v52;
      if ( v51 == v50 )
        return result;
    }
    v56 = *v54 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v56 )
    {
      if ( (*((_BYTE *)v55 + 9) & 0xC) != 8
        || (*((_BYTE *)v55 + 8) |= 4u,
            srcc = v55,
            v76 = (unsigned __int64)sub_38CE440(v55[3]),
            v77 = v76 | *srcc & 7,
            *srcc = v77,
            !v76) )
      {
        v55 = 0;
        goto LABEL_95;
      }
      v56 = v77 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v56 )
      {
        if ( (*((_BYTE *)srcc + 9) & 0xC) != 8 )
          BUG();
        *((_BYTE *)srcc + 8) |= 4u;
        v56 = (unsigned __int64)sub_38CE440(srcc[3]);
        *srcc = v56 | *srcc & 7;
      }
    }
    v78 = *(_DWORD *)(v4 + 184);
    v79 = *(_QWORD *)(v56 + 24);
    srcd = (void *)(v4 + 160);
    v80 = *(_QWORD *)(v4 + 168);
    if ( v78 )
    {
      v81 = (v78 - 1) & (((unsigned int)v79 >> 9) ^ ((unsigned int)v79 >> 4));
      v82 = (char *)(v80 + 16 * v81);
      v83 = *(_QWORD *)v82;
      if ( v79 == *(_QWORD *)v82 )
      {
        v55 = (__int64 *)*((_QWORD *)v82 + 1);
        goto LABEL_147;
      }
      v132 = 1;
      v103 = 0;
      while ( v83 != -8 )
      {
        if ( v83 != -16 || v103 )
          v82 = v103;
        LODWORD(v81) = (v78 - 1) & (v132 + v81);
        v127 = (__int64 *)(v80 + 16LL * (unsigned int)v81);
        v83 = *v127;
        if ( v79 == *v127 )
        {
          v55 = (__int64 *)v127[1];
          goto LABEL_147;
        }
        ++v132;
        v103 = v82;
        v82 = (char *)(v80 + 16LL * (unsigned int)v81);
      }
      if ( !v103 )
        v103 = v82;
      v104 = *(_DWORD *)(v4 + 176);
      ++*(_QWORD *)(v4 + 160);
      v105 = v104 + 1;
      if ( 4 * v105 < 3 * v78 )
      {
        if ( v78 - *(_DWORD *)(v4 + 180) - v105 > v78 >> 3 )
          goto LABEL_188;
        v128 = ((unsigned int)v79 >> 9) ^ ((unsigned int)v79 >> 4);
        v134 = v79;
        sub_3925B80((__int64)srcd, v78);
        v113 = *(_DWORD *)(v4 + 184);
        if ( !v113 )
        {
LABEL_238:
          ++*(_DWORD *)(v4 + 176);
          BUG();
        }
        v114 = v113 - 1;
        v115 = 1;
        v79 = v134;
        srce = *(char **)(v4 + 168);
        v116 = v114 & v128;
        v105 = *(_DWORD *)(v4 + 176) + 1;
        v103 = &srce[16 * (v114 & v128)];
        v112 = 0;
        v117 = *(_QWORD *)v103;
        if ( v134 == *(_QWORD *)v103 )
          goto LABEL_188;
        while ( v117 != -8 )
        {
          if ( !v112 && v117 == -16 )
            v112 = v103;
          v116 = v114 & (v115 + v116);
          v103 = &srce[16 * v116];
          v117 = *(_QWORD *)v103;
          if ( v134 == *(_QWORD *)v103 )
            goto LABEL_188;
          ++v115;
        }
        goto LABEL_201;
      }
    }
    else
    {
      ++*(_QWORD *)(v4 + 160);
    }
    v133 = v79;
    sub_3925B80((__int64)srcd, 2 * v78);
    v106 = *(_DWORD *)(v4 + 184);
    if ( !v106 )
      goto LABEL_238;
    v79 = v133;
    v107 = v106 - 1;
    v108 = *(_QWORD *)(v4 + 168);
    v109 = (v106 - 1) & (((unsigned int)v133 >> 9) ^ ((unsigned int)v133 >> 4));
    v105 = *(_DWORD *)(v4 + 176) + 1;
    v103 = (char *)(v108 + 16LL * v109);
    v110 = *(_QWORD *)v103;
    if ( v133 == *(_QWORD *)v103 )
      goto LABEL_188;
    v111 = 1;
    v112 = 0;
    while ( v110 != -8 )
    {
      if ( !v112 && v110 == -16 )
        v112 = v103;
      v109 = v107 & (v111 + v109);
      v103 = (char *)(v108 + 16LL * v109);
      v110 = *(_QWORD *)v103;
      if ( v133 == *(_QWORD *)v103 )
        goto LABEL_188;
      ++v111;
    }
LABEL_201:
    if ( v112 )
      v103 = v112;
LABEL_188:
    *(_DWORD *)(v4 + 176) = v105;
    if ( *(_QWORD *)v103 != -8 )
      --*(_DWORD *)(v4 + 180);
    *(_QWORD *)v103 = v79;
    *((_QWORD *)v103 + 1) = 0;
    v55 = 0;
LABEL_147:
    v84 = *(__int64 **)(v53 + 104);
    if ( v84 && v84 != v55 )
      sub_16BD130("conflicting sections for symbol", 1u);
LABEL_95:
    if ( (*(_BYTE *)(v52 + 13) & 1) == 0 )
    {
      *(_QWORD *)(v53 + 104) = v55;
      v57 = (void *)v53;
      goto LABEL_97;
    }
    goto LABEL_108;
  }
  return result;
}
