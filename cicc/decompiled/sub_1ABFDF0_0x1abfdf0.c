// Function: sub_1ABFDF0
// Address: 0x1abfdf0
//
__int64 __fastcall sub_1ABFDF0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 *v6; // r12
  __int64 *v7; // rbx
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rdi
  unsigned int v12; // esi
  __int64 *v13; // rdx
  __int64 v14; // r12
  __int64 v15; // rbx
  unsigned int v16; // edx
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 v19; // r14
  char v20; // dl
  unsigned __int8 v21; // dl
  __int64 v22; // r15
  __int64 *v23; // rax
  __int64 *v24; // rsi
  __int64 *v25; // rcx
  __int64 v26; // r12
  char v27; // al
  __int64 v28; // rax
  unsigned int v29; // ecx
  __int64 v30; // rdi
  char v31; // bl
  __int64 v32; // rdx
  __int64 v33; // r14
  __int64 v34; // rbx
  __int64 *v35; // r13
  unsigned __int64 v36; // rax
  __int64 v37; // rdx
  __int64 *v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rdi
  __int64 v41; // rdx
  __int64 v42; // rax
  __int64 v43; // r15
  _QWORD *v44; // rax
  int v45; // ebx
  __int64 v46; // r14
  int v47; // r12d
  __int64 v48; // rdx
  int v49; // esi
  unsigned int v50; // eax
  __int64 v51; // rcx
  __int64 v52; // rsi
  __int64 v53; // rsi
  __int64 *v55; // r14
  char v56; // di
  char v57; // cl
  __int64 v58; // rax
  __int64 *v59; // r15
  __int64 v60; // r10
  unsigned int v61; // eax
  int v62; // eax
  __int64 *v63; // r15
  int v64; // r13d
  __int64 v65; // rbx
  int v66; // r12d
  __int64 v67; // rax
  __int64 v68; // rdx
  unsigned int v69; // eax
  __int64 v70; // rdi
  __int64 v71; // r13
  int v72; // r15d
  __int64 v73; // r14
  _QWORD *v74; // rax
  __int64 v75; // rdx
  int v76; // esi
  unsigned int v77; // eax
  __int64 v78; // rcx
  int v79; // edx
  int v80; // r10d
  __int64 v81; // r13
  int v82; // r15d
  __int64 v83; // r14
  _QWORD *v84; // rax
  __int64 v85; // rdx
  int v86; // esi
  unsigned int v87; // eax
  __int64 v88; // rcx
  __int64 v89; // rdi
  unsigned int v90; // eax
  __int64 v91; // rcx
  int v92; // esi
  __int64 v93; // rax
  __int64 v94; // [rsp+0h] [rbp-1F0h]
  __int64 v95; // [rsp+8h] [rbp-1E8h]
  int v96; // [rsp+8h] [rbp-1E8h]
  int v97; // [rsp+8h] [rbp-1E8h]
  char v98; // [rsp+16h] [rbp-1DAh]
  char v99; // [rsp+17h] [rbp-1D9h]
  __int64 *v100; // [rsp+18h] [rbp-1D8h]
  __int64 v101; // [rsp+20h] [rbp-1D0h]
  __int64 *v102; // [rsp+30h] [rbp-1C0h]
  __int64 v104; // [rsp+40h] [rbp-1B0h] BYREF
  __int64 v105; // [rsp+48h] [rbp-1A8h]
  __int64 v106; // [rsp+50h] [rbp-1A0h]
  __int64 v107; // [rsp+58h] [rbp-198h]
  __int64 *v108; // [rsp+60h] [rbp-190h]
  __int64 *v109; // [rsp+68h] [rbp-188h]
  __int64 v110; // [rsp+70h] [rbp-180h]
  _BYTE *v111; // [rsp+80h] [rbp-170h] BYREF
  __int64 v112; // [rsp+88h] [rbp-168h]
  _BYTE v113[128]; // [rsp+90h] [rbp-160h] BYREF
  __int64 v114; // [rsp+110h] [rbp-E0h] BYREF
  __int64 *v115; // [rsp+118h] [rbp-D8h]
  __int64 *v116; // [rsp+120h] [rbp-D0h]
  __int64 v117; // [rsp+128h] [rbp-C8h]
  int v118; // [rsp+130h] [rbp-C0h]
  _BYTE v119[184]; // [rsp+138h] [rbp-B8h] BYREF

  v6 = &a2[a3];
  v98 = a5;
  v99 = a6;
  v104 = 0;
  v105 = 0;
  v106 = 0;
  v107 = 0;
  v108 = 0;
  v109 = 0;
  v110 = 0;
  if ( v6 == a2 )
  {
    v100 = 0;
    v38 = 0;
LABEL_69:
    v39 = v105;
    v40 = 0;
    v105 = 0;
    ++v104;
    *(_QWORD *)(a1 + 8) = v39;
    v41 = v106;
    *(_QWORD *)(a1 + 40) = v38;
    v42 = v110;
    *(_QWORD *)(a1 + 16) = v41;
    LODWORD(v41) = v107;
    *(_QWORD *)a1 = 1;
    *(_DWORD *)(a1 + 24) = v41;
    v106 = 0;
    LODWORD(v107) = 0;
    *(_QWORD *)(a1 + 32) = v100;
    *(_QWORD *)(a1 + 48) = v42;
    goto LABEL_82;
  }
  v7 = a2;
  do
  {
    v9 = *v7;
    v114 = *v7;
    if ( !a4 )
    {
LABEL_8:
      sub_1ABFB80((__int64)&v104, &v114);
      goto LABEL_9;
    }
    v10 = *(unsigned int *)(a4 + 48);
    if ( !(_DWORD)v10 )
      goto LABEL_9;
    LODWORD(a5) = v10 - 1;
    v11 = *(_QWORD *)(a4 + 32);
    v12 = (v10 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
    v13 = (__int64 *)(v11 + 16LL * v12);
    a6 = *v13;
    if ( v9 == *v13 )
    {
LABEL_6:
      if ( v13 != (__int64 *)(v11 + 16 * v10) && v13[1] )
        goto LABEL_8;
    }
    else
    {
      v79 = 1;
      while ( a6 != -8 )
      {
        v80 = v79 + 1;
        v12 = a5 & (v79 + v12);
        v13 = (__int64 *)(v11 + 16LL * v12);
        a6 = *v13;
        if ( v9 == *v13 )
          goto LABEL_6;
        v79 = v80;
      }
    }
LABEL_9:
    ++v7;
  }
  while ( v6 != v7 );
  v100 = v109;
  if ( v108 == v109 )
  {
    v38 = v109;
    goto LABEL_69;
  }
  v102 = v108;
  while ( 2 )
  {
    v14 = *v102;
    if ( *(_WORD *)(*v102 + 18) )
      break;
    v15 = v14 + 40;
    v114 = 0;
    v115 = (__int64 *)v119;
    v16 = 16;
    v116 = (__int64 *)v119;
    v117 = 16;
    v118 = 0;
    v17 = *(_QWORD *)(v14 + 48);
    v111 = v113;
    v112 = 0x1000000000LL;
    v18 = 0;
    if ( v17 == v14 + 40 )
    {
LABEL_134:
      v31 = 1;
      goto LABEL_59;
    }
    while ( 1 )
    {
      v19 = v17 - 24;
      if ( !v17 )
        v19 = 0;
      if ( v16 <= (unsigned int)v18 )
      {
        sub_16CD150((__int64)&v111, v113, 0, 8, (unsigned __int8)a5, (unsigned __int8)a6);
        v18 = (unsigned int)v112;
      }
      *(_QWORD *)&v111[8 * v18] = v19;
      v18 = (unsigned int)(v112 + 1);
      LODWORD(v112) = v112 + 1;
      v17 = *(_QWORD *)(v17 + 8);
      if ( v15 == v17 )
        break;
      v16 = HIDWORD(v112);
    }
    if ( (_DWORD)v18 )
    {
      while ( 2 )
      {
        while ( 1 )
        {
          v22 = *(_QWORD *)&v111[8 * (unsigned int)v18 - 8];
          LODWORD(v112) = v18 - 1;
          v23 = v115;
          if ( v116 == v115 )
          {
            v24 = &v115[HIDWORD(v117)];
            if ( v115 != v24 )
            {
              v25 = 0;
              while ( v22 != *v23 )
              {
                if ( *v23 == -2 )
                  v25 = v23;
                if ( v24 == ++v23 )
                {
                  if ( !v25 )
                    goto LABEL_70;
                  *v25 = v22;
                  --v118;
                  ++v114;
                  v21 = *(_BYTE *)(v22 + 16);
                  if ( v21 == 4 )
                    goto LABEL_58;
                  goto LABEL_25;
                }
              }
              goto LABEL_35;
            }
LABEL_70:
            if ( HIDWORD(v117) < (unsigned int)v117 )
              break;
          }
          sub_16CCBA0((__int64)&v114, v22);
          if ( v20 )
            goto LABEL_24;
LABEL_35:
          LODWORD(v18) = v112;
          if ( !(_DWORD)v112 )
            goto LABEL_36;
        }
        ++HIDWORD(v117);
        *v24 = v22;
        ++v114;
LABEL_24:
        v21 = *(_BYTE *)(v22 + 16);
        if ( v21 == 4 )
          goto LABEL_58;
LABEL_25:
        v18 = (unsigned int)v112;
        if ( v21 <= 0x17u || v14 == *(_QWORD *)(v22 + 40) )
        {
          v32 = 24LL * (*(_DWORD *)(v22 + 20) & 0xFFFFFFF);
          a5 = v22 - v32;
          if ( (*(_BYTE *)(v22 + 23) & 0x40) != 0 )
          {
            a5 = *(_QWORD *)(v22 - 8);
            v22 = a5 + v32;
          }
          if ( a5 != v22 )
          {
            v33 = a5;
            do
            {
              v34 = *(_QWORD *)v33;
              if ( (unsigned __int8)(*(_BYTE *)(*(_QWORD *)v33 + 16LL) - 17) > 6u )
              {
                if ( (unsigned int)v18 >= HIDWORD(v112) )
                {
                  sub_16CD150((__int64)&v111, v113, 0, 8, a5, (unsigned __int8)a6);
                  v18 = (unsigned int)v112;
                }
                *(_QWORD *)&v111[8 * v18] = v34;
                v18 = (unsigned int)(v112 + 1);
                LODWORD(v112) = v112 + 1;
              }
              v33 += 24;
            }
            while ( v22 != v33 );
          }
        }
        if ( !(_DWORD)v18 )
        {
LABEL_36:
          v15 = v14 + 40;
          break;
        }
        continue;
      }
    }
    if ( v15 == *(_QWORD *)(v14 + 48) )
      goto LABEL_134;
    v101 = v14;
    v26 = *(_QWORD *)(v14 + 48);
LABEL_39:
    while ( 2 )
    {
      if ( !v26 )
        BUG();
      v27 = *(_BYTE *)(v26 - 8);
      switch ( v27 )
      {
        case 53:
          if ( !v99 )
            goto LABEL_86;
          goto LABEL_45;
        case 29:
          v28 = *(_QWORD *)(v26 - 48);
          if ( v28 )
          {
            if ( !(_DWORD)v107 )
              goto LABEL_86;
            LOBYTE(a5) = v105;
            v29 = (v107 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
            v30 = *(_QWORD *)(v105 + 8LL * v29);
            if ( v28 != v30 )
            {
              LODWORD(a6) = 1;
              while ( v30 != -8 )
              {
                v29 = (v107 - 1) & (a6 + v29);
                v30 = *(_QWORD *)(v105 + 8LL * v29);
                if ( v28 == v30 )
                  goto LABEL_45;
                LODWORD(a6) = a6 + 1;
              }
              goto LABEL_86;
            }
          }
          goto LABEL_45;
        case 34:
          v55 = (__int64 *)(v26 - 24);
          v56 = *(_BYTE *)(v26 - 1) & 0x40;
          v57 = *(_BYTE *)(v26 - 6) & 1;
          if ( !v57 )
            goto LABEL_93;
          if ( v56 )
          {
            v59 = *(__int64 **)(v26 - 32);
            v60 = v59[3];
            if ( v60 )
              goto LABEL_91;
            v62 = *(_DWORD *)(v26 - 4);
          }
          else
          {
            v58 = 24LL * (*(_DWORD *)(v26 - 4) & 0xFFFFFFF);
            v59 = &v55[v58 / 0xFFFFFFFFFFFFFFF8LL];
            v60 = *(_QWORD *)(v26 - v58);
            if ( !v60 )
              goto LABEL_125;
LABEL_91:
            if ( !(_DWORD)v107 )
              goto LABEL_86;
            v61 = (v107 - 1) & (((unsigned int)v60 >> 9) ^ ((unsigned int)v60 >> 4));
            a6 = *(_QWORD *)(v105 + 8LL * v61);
            LODWORD(a5) = 1;
            if ( a6 != v60 )
            {
              while ( a6 != -8 )
              {
                v61 = (v107 - 1) & (a5 + v61);
                a6 = *(_QWORD *)(v105 + 8LL * v61);
                if ( v60 == a6 )
                  goto LABEL_93;
                LODWORD(a5) = a5 + 1;
              }
              goto LABEL_86;
            }
LABEL_93:
            v62 = *(_DWORD *)(v26 - 4);
            LODWORD(a5) = v62 & 0xFFFFFFF;
            if ( !v56 )
            {
              v59 = &v55[-3 * (unsigned int)a5];
              if ( !v57 )
                goto LABEL_95;
LABEL_125:
              v63 = v59 + 6;
              goto LABEL_96;
            }
            v59 = *(__int64 **)(v26 - 32);
          }
          v55 = &v59[3 * (v62 & 0xFFFFFFF)];
          if ( v57 )
            goto LABEL_125;
LABEL_95:
          v63 = v59 + 3;
LABEL_96:
          if ( v55 != v63 )
          {
            v64 = v107;
            v95 = v26;
            v94 = v15;
            v65 = v105;
            v66 = v107 - 1;
            while ( v64 )
            {
              v67 = sub_1523720(*v63);
              LODWORD(a5) = 1;
              v68 = v67;
              v69 = v66 & (((unsigned int)v67 >> 9) ^ ((unsigned int)v67 >> 4));
              v70 = *(_QWORD *)(v65 + 8LL * v69);
              if ( v68 != v70 )
              {
                while ( v70 != -8 )
                {
                  v69 = v66 & (a5 + v69);
                  v70 = *(_QWORD *)(v65 + 8LL * v69);
                  if ( v68 == v70 )
                    goto LABEL_100;
                  LODWORD(a5) = a5 + 1;
                }
                goto LABEL_86;
              }
LABEL_100:
              v63 += 3;
              if ( v55 == v63 )
              {
                v15 = v94;
                v26 = *(_QWORD *)(v95 + 8);
                if ( v94 != v26 )
                  goto LABEL_39;
                goto LABEL_46;
              }
            }
            goto LABEL_86;
          }
          goto LABEL_45;
        case 74:
          v71 = *(_QWORD *)(v26 - 16);
          if ( v71 )
          {
            v72 = v107;
            v73 = v105;
            v96 = v107 - 1;
            v74 = sub_1648700(*(_QWORD *)(v26 - 16));
            if ( *((_BYTE *)v74 + 16) == 33 )
              goto LABEL_110;
LABEL_108:
            while ( 1 )
            {
              v71 = *(_QWORD *)(v71 + 8);
              if ( !v71 )
                break;
              v74 = sub_1648700(v71);
              if ( *((_BYTE *)v74 + 16) == 33 )
              {
LABEL_110:
                if ( !v72 )
                  goto LABEL_86;
                v75 = v74[5];
                v76 = 1;
                v77 = v96 & (((unsigned int)v75 >> 9) ^ ((unsigned int)v75 >> 4));
                v78 = *(_QWORD *)(v73 + 8LL * v77);
                if ( v75 != v78 )
                {
                  while ( v78 != -8 )
                  {
                    v77 = v96 & (v76 + v77);
                    v78 = *(_QWORD *)(v73 + 8LL * v77);
                    if ( v75 == v78 )
                      goto LABEL_108;
                    ++v76;
                  }
                  goto LABEL_86;
                }
              }
            }
          }
          goto LABEL_45;
        case 73:
          v81 = *(_QWORD *)(v26 - 16);
          if ( v81 )
          {
            v82 = v107;
            v83 = v105;
            v97 = v107 - 1;
            while ( 1 )
            {
              v84 = sub_1648700(v81);
              if ( *((_BYTE *)v84 + 16) == 32 )
              {
                if ( !v82 )
                  goto LABEL_86;
                v85 = v84[5];
                v86 = 1;
                v87 = v97 & (((unsigned int)v85 >> 9) ^ ((unsigned int)v85 >> 4));
                v88 = *(_QWORD *)(v83 + 8LL * v87);
                if ( v85 != v88 )
                  break;
              }
LABEL_130:
              v81 = *(_QWORD *)(v81 + 8);
              if ( !v81 )
                goto LABEL_45;
            }
            while ( v88 != -8 )
            {
              v87 = v97 & (v86 + v87);
              v88 = *(_QWORD *)(v83 + 8LL * v87);
              if ( v85 == v88 )
                goto LABEL_130;
              ++v86;
            }
            goto LABEL_86;
          }
          goto LABEL_45;
      }
      if ( v27 != 32 )
      {
        if ( v27 == 78 )
        {
          v93 = *(_QWORD *)(v26 - 48);
          if ( !*(_BYTE *)(v93 + 16) && *(_DWORD *)(v93 + 36) == 214 && !v98 )
            goto LABEL_86;
        }
        goto LABEL_45;
      }
      if ( (*(_BYTE *)(v26 - 6) & 1) == 0 )
        goto LABEL_45;
      v89 = *(_QWORD *)(v26 + 24 * (1LL - (*(_DWORD *)(v26 - 4) & 0xFFFFFFF)) - 24);
      if ( !v89 )
        goto LABEL_45;
      LOBYTE(a5) = v107;
      if ( !(_DWORD)v107 )
        goto LABEL_86;
      LODWORD(a5) = v107 - 1;
      LOBYTE(a6) = v105;
      v90 = (v107 - 1) & (((unsigned int)v89 >> 9) ^ ((unsigned int)v89 >> 4));
      v91 = *(_QWORD *)(v105 + 8LL * v90);
      if ( v89 == v91 )
      {
LABEL_45:
        v26 = *(_QWORD *)(v26 + 8);
        if ( v15 == v26 )
        {
LABEL_46:
          v14 = v101;
          v31 = 1;
          goto LABEL_59;
        }
        continue;
      }
      break;
    }
    v92 = 1;
    while ( v91 != -8 )
    {
      v90 = a5 & (v92 + v90);
      v91 = *(_QWORD *)(v105 + 8LL * v90);
      if ( v89 == v91 )
        goto LABEL_45;
      ++v92;
    }
LABEL_86:
    v14 = v101;
LABEL_58:
    v31 = 0;
LABEL_59:
    if ( v111 != v113 )
      _libc_free((unsigned __int64)v111);
    if ( v116 != v115 )
      _libc_free((unsigned __int64)v116);
    v35 = v108;
    if ( v31 )
    {
      if ( *v108 == v14 )
      {
        v36 = (unsigned int)*(unsigned __int8 *)(sub_157ED20(v14) + 16) - 34;
        if ( (unsigned int)v36 <= 0x36 )
        {
          v37 = 0x40018000000001LL;
          if ( _bittest64(&v37, v36) )
            break;
        }
      }
      else
      {
        v43 = *(_QWORD *)(v14 + 8);
        if ( v43 )
        {
          while ( 1 )
          {
            v44 = sub_1648700(v43);
            if ( (unsigned __int8)(*((_BYTE *)v44 + 16) - 25) <= 9u )
              break;
            v43 = *(_QWORD *)(v43 + 8);
            if ( !v43 )
              goto LABEL_67;
          }
          v45 = v107;
          v46 = v105;
          v47 = v107 - 1;
LABEL_78:
          if ( v45 )
          {
            v48 = v44[5];
            v49 = 1;
            v50 = v47 & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
            v51 = *(_QWORD *)(v46 + 8LL * v50);
            if ( v48 == v51 )
              goto LABEL_76;
            while ( v51 != -8 )
            {
              v50 = v47 & (v49 + v50);
              v51 = *(_QWORD *)(v46 + 8LL * v50);
              if ( v48 == v51 )
              {
LABEL_76:
                while ( 1 )
                {
                  v43 = *(_QWORD *)(v43 + 8);
                  if ( !v43 )
                    goto LABEL_67;
                  v44 = sub_1648700(v43);
                  if ( (unsigned __int8)(*((_BYTE *)v44 + 16) - 25) <= 9u )
                    goto LABEL_78;
                }
              }
              ++v49;
            }
          }
          v52 = v110;
          *(_QWORD *)(a1 + 48) = 0;
          v53 = v52 - (_QWORD)v35;
          *(_OWORD *)a1 = 0;
          *(_OWORD *)(a1 + 16) = 0;
          *(_OWORD *)(a1 + 32) = 0;
LABEL_80:
          j_j___libc_free_0(v35, v53);
          goto LABEL_81;
        }
      }
LABEL_67:
      if ( v100 == ++v102 )
      {
        v100 = v108;
        v38 = v109;
        goto LABEL_69;
      }
      continue;
    }
    break;
  }
  v35 = v108;
  *(_QWORD *)(a1 + 48) = 0;
  *(_OWORD *)a1 = 0;
  *(_OWORD *)(a1 + 16) = 0;
  *(_OWORD *)(a1 + 32) = 0;
  if ( v35 )
  {
    v53 = v110 - (_QWORD)v35;
    goto LABEL_80;
  }
LABEL_81:
  v40 = v105;
LABEL_82:
  j___libc_free_0(v40);
  return a1;
}
