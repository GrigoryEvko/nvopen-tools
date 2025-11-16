// Function: sub_2B2BBE0
// Address: 0x2b2bbe0
//
__int64 __fastcall sub_2B2BBE0(__int64 a1, char *a2, __int64 a3)
{
  __int64 v3; // r8
  char *v4; // r14
  char *v5; // rbx
  unsigned __int8 *v6; // r12
  unsigned __int8 *v7; // r13
  unsigned __int8 *i; // rdi
  __int64 v9; // rcx
  __int64 v10; // rdx
  char *j; // rax
  char *v12; // rcx
  unsigned __int8 *v13; // rax
  __int64 v14; // r12
  __int64 v15; // rax
  __int64 v16; // r15
  unsigned int v17; // edx
  __int64 v19; // rsi
  int v20; // ecx
  unsigned __int64 v21; // rcx
  __int64 v23; // rdi
  char *v24; // rdx
  unsigned int v25; // esi
  __int64 v27; // r8
  int v28; // ecx
  unsigned __int64 v29; // rcx
  __int64 v31; // rdi
  unsigned int v32; // esi
  __int64 v34; // r8
  int v35; // ecx
  unsigned __int64 v36; // rcx
  __int64 v38; // rdi
  unsigned int v39; // esi
  __int64 v41; // r8
  int v42; // ecx
  unsigned __int64 v43; // rcx
  unsigned __int8 v45; // al
  __int64 v46; // rdx
  unsigned __int8 *v48; // rsi
  __int64 v49; // rcx
  __int64 v50; // r10
  unsigned __int8 *v51; // rsi
  __int64 v52; // rcx
  unsigned __int8 *v53; // rsi
  __int64 v54; // rcx
  unsigned __int8 **v55; // r9
  unsigned __int8 **v56; // rcx
  __int64 v57; // r13
  __int64 v58; // rdi
  char *v59; // r15
  __int64 v60; // rax
  __int64 v61; // rdi
  __int64 v62; // rax
  __int64 v63; // rdi
  __int64 v64; // rax
  signed __int64 v65; // rax
  unsigned __int8 *v66; // rax
  __int64 v67; // rax
  unsigned __int8 *v68; // rax
  __int64 v69; // rax
  __int64 v70; // rax
  signed __int64 v71; // rdx
  unsigned __int8 *v72; // rdx
  unsigned int v73; // r12d
  __int64 v74; // rdi
  int v76; // r13d
  unsigned __int64 v77; // r13
  unsigned __int8 *v79; // rax
  unsigned int v80; // r13d
  __int64 v81; // rdi
  int v83; // r15d
  unsigned __int64 v84; // rax
  unsigned __int8 *v86; // rax
  unsigned int v87; // r13d
  __int64 v88; // rdi
  __int64 v90; // r12
  int v91; // r15d
  unsigned __int8 *v93; // rdi
  __int64 v94; // rcx
  unsigned __int8 *v95; // rdi
  unsigned __int8 *v96; // rdi
  unsigned __int8 *v97; // rax
  __int64 v98; // rax
  unsigned __int64 v99; // rax
  __int64 v100; // r12
  __int64 v101; // r15
  char v102; // [rsp+Dh] [rbp-53h]
  char v103; // [rsp+Eh] [rbp-52h]
  bool v104; // [rsp+Fh] [rbp-51h]
  unsigned __int8 *v105; // [rsp+10h] [rbp-50h]
  __int64 v106; // [rsp+18h] [rbp-48h]
  unsigned int v107; // [rsp+18h] [rbp-48h]
  __int64 v108; // [rsp+20h] [rbp-40h]
  int v109; // [rsp+20h] [rbp-40h]
  int v110; // [rsp+20h] [rbp-40h]
  int v111; // [rsp+20h] [rbp-40h]
  int v112; // [rsp+20h] [rbp-40h]
  char *v113; // [rsp+28h] [rbp-38h]

  v3 = 8 * a3;
  v4 = a2;
  v5 = a2;
  v113 = &a2[8 * a3];
  v6 = *(unsigned __int8 **)a2;
  v106 = (8 * a3) >> 3;
  v7 = *(unsigned __int8 **)a2;
  v108 = (8 * a3) >> 5;
  if ( v108 <= 0 )
  {
    v70 = (8 * a3) >> 3;
    v56 = (unsigned __int8 **)a2;
LABEL_127:
    if ( v70 != 2 )
    {
      if ( v70 != 3 )
      {
        if ( v70 != 1 )
        {
          v103 = 1;
          goto LABEL_131;
        }
        goto LABEL_181;
      }
      v93 = *v56;
      if ( !(unsigned __int8)sub_2B0D8B0(*v56) || (unsigned int)*v93 - 12 <= 1 )
        goto LABEL_183;
      v56 = (unsigned __int8 **)(v94 + 8);
    }
    v95 = *v56;
    if ( !(unsigned __int8)sub_2B0D8B0(*v56) || (unsigned int)*v95 - 12 <= 1 )
      goto LABEL_183;
    v56 = (unsigned __int8 **)(v94 + 8);
LABEL_181:
    v96 = *v56;
    v103 = sub_2B0D8B0(*v56);
    if ( v103 && (unsigned int)*v96 - 12 > 1 )
    {
LABEL_131:
      if ( v108 <= 0 )
      {
        if ( v3 != 24 )
        {
          if ( v3 == 8 )
          {
            v104 = 1;
            v102 = v103;
            goto LABEL_217;
          }
          if ( v3 != 16 )
          {
            v104 = 1;
            v102 = v103;
            goto LABEL_223;
          }
          v12 = v4;
LABEL_195:
          v12 += 8;
LABEL_196:
          if ( v6 == *(unsigned __int8 **)v12 )
          {
            v104 = 1;
            v102 = v103;
LABEL_198:
            if ( v108 > 0 )
            {
              v7 = v6;
              goto LABEL_13;
            }
            switch ( v3 )
            {
              case 16LL:
                v68 = v6;
                if ( *v6 != 17 )
                  goto LABEL_157;
                break;
              case 24LL:
                v66 = v6;
                if ( *v6 != 17 )
                {
                  v106 = 3;
                  goto LABEL_117;
                }
LABEL_102:
                if ( *((_DWORD *)v66 + 8) > 0x40u )
                {
                  if ( (unsigned int)sub_C44630((__int64)(v66 + 24)) != 1 )
                    goto LABEL_109;
                }
                else
                {
                  v67 = *((_QWORD *)v66 + 3);
                  if ( !v67 || (v67 & (v67 - 1)) != 0 )
                    goto LABEL_109;
                }
                v4 += 8;
LABEL_106:
                v68 = *(unsigned __int8 **)v4;
                if ( **(_BYTE **)v4 != 17 )
                {
LABEL_109:
                  if ( v108 > 0 )
                  {
                    v7 = v6;
                    goto LABEL_17;
                  }
LABEL_116:
                  if ( v106 == 2 )
                    goto LABEL_157;
                  goto LABEL_117;
                }
                break;
              case 8LL:
LABEL_217:
                v97 = v6;
                if ( *v6 != 17 )
                {
                  v106 = 1;
LABEL_117:
                  if ( v106 != 3 )
                  {
                    if ( v106 != 1 )
                      goto LABEL_119;
LABEL_167:
                    v86 = *(unsigned __int8 **)v5;
                    if ( **(_BYTE **)v5 != 17 )
                      goto LABEL_56;
                    v87 = *((_DWORD *)v86 + 8);
                    v88 = 1LL << ((unsigned __int8)v87 - 1);
                    _RDX = *((_QWORD *)v86 + 3);
                    if ( v87 <= 0x40 )
                    {
                      if ( (v88 & _RDX) == 0 )
                        goto LABEL_56;
                      if ( v87 )
                      {
                        v91 = 64;
                        if ( _RDX << (64 - (unsigned __int8)v87) != -1 )
                        {
                          _BitScanReverse64(&v99, ~(_RDX << (64 - (unsigned __int8)v87)));
                          v91 = v99 ^ 0x3F;
                        }
                      }
                      else
                      {
                        v91 = 0;
                      }
                      __asm { tzcnt   rax, rdx }
                      if ( (unsigned int)_RAX > v87 )
                        LODWORD(_RAX) = v87;
                    }
                    else
                    {
                      if ( (*(_QWORD *)(_RDX + 8LL * ((v87 - 1) >> 6)) & v88) == 0 )
                        goto LABEL_56;
                      v90 = (__int64)(v86 + 24);
                      v91 = sub_C44500((__int64)(v86 + 24));
                      LODWORD(_RAX) = sub_C44590(v90);
                    }
                    if ( v87 != v91 + (_DWORD)_RAX )
                      goto LABEL_56;
                    if ( !v102 )
                    {
                      if ( v103 )
                      {
                        v45 = 3;
                        v46 = 2;
                        return (v46 << 32) | v45;
                      }
                      v5 = v113;
LABEL_122:
                      v45 = v104;
                      goto LABEL_57;
                    }
                    goto LABEL_192;
                  }
                  v72 = *(unsigned __int8 **)v5;
                  if ( **(_BYTE **)v5 != 17 )
                    goto LABEL_56;
                  v73 = *((_DWORD *)v72 + 8);
                  v74 = 1LL << ((unsigned __int8)v73 - 1);
                  _RAX = *((_QWORD *)v72 + 3);
                  if ( v73 > 0x40 )
                  {
                    if ( (*(_QWORD *)(_RAX + 8LL * ((v73 - 1) >> 6)) & v74) == 0 )
                      goto LABEL_56;
                    v101 = (__int64)(v72 + 24);
                    v76 = sub_C44500((__int64)(v72 + 24));
                    LODWORD(_RAX) = sub_C44590(v101);
                  }
                  else
                  {
                    if ( (v74 & _RAX) == 0 )
                      goto LABEL_56;
                    if ( v73 )
                    {
                      v76 = 64;
                      if ( _RAX << (64 - (unsigned __int8)v73) != -1 )
                      {
                        _BitScanReverse64(&v77, ~(_RAX << (64 - (unsigned __int8)v73)));
                        v76 = v77 ^ 0x3F;
                      }
                    }
                    else
                    {
                      v76 = 0;
                    }
                    __asm { tzcnt   rax, rax }
                    if ( (unsigned int)_RAX > v73 )
                      LODWORD(_RAX) = *((_DWORD *)v72 + 8);
                  }
                  if ( v73 != v76 + (_DWORD)_RAX )
                    goto LABEL_56;
                  v5 += 8;
LABEL_157:
                  v79 = *(unsigned __int8 **)v5;
                  if ( **(_BYTE **)v5 != 17 )
                    goto LABEL_56;
                  v80 = *((_DWORD *)v79 + 8);
                  v81 = 1LL << ((unsigned __int8)v80 - 1);
                  _RDX = *((_QWORD *)v79 + 3);
                  if ( v80 > 0x40 )
                  {
                    if ( (*(_QWORD *)(_RDX + 8LL * ((v80 - 1) >> 6)) & v81) == 0 )
                      goto LABEL_56;
                    v100 = (__int64)(v79 + 24);
                    v83 = sub_C44500((__int64)(v79 + 24));
                    LODWORD(_RAX) = sub_C44590(v100);
                  }
                  else
                  {
                    if ( (v81 & _RDX) == 0 )
                      goto LABEL_56;
                    if ( v80 )
                    {
                      v83 = 64;
                      if ( _RDX << (64 - (unsigned __int8)v80) != -1 )
                      {
                        _BitScanReverse64(&v84, ~(_RDX << (64 - (unsigned __int8)v80)));
                        v83 = v84 ^ 0x3F;
                      }
                    }
                    else
                    {
                      v83 = 0;
                    }
                    __asm { tzcnt   rax, rdx }
                    if ( (unsigned int)_RAX > v80 )
                      LODWORD(_RAX) = v80;
                  }
                  if ( v80 != v83 + (_DWORD)_RAX )
                    goto LABEL_56;
                  v5 += 8;
                  goto LABEL_167;
                }
LABEL_187:
                if ( *((_DWORD *)v97 + 8) > 0x40u )
                {
                  if ( (unsigned int)sub_C44630((__int64)(v97 + 24)) == 1 )
                    v4 = v113;
                }
                else
                {
                  v98 = *((_QWORD *)v97 + 3);
                  if ( v98 && (v98 & (v98 - 1)) == 0 )
                    v4 = v113;
                }
                goto LABEL_109;
              default:
LABEL_223:
                v4 = v113;
LABEL_119:
                if ( !v102 )
                {
                  v5 = v113;
                  goto LABEL_121;
                }
LABEL_192:
                v45 = 2;
                v46 = 2;
                return (v46 << 32) | v45;
            }
            if ( *((_DWORD *)v68 + 8) > 0x40u )
            {
              if ( (unsigned int)sub_C44630((__int64)(v68 + 24)) != 1 )
                goto LABEL_109;
            }
            else
            {
              v69 = *((_QWORD *)v68 + 3);
              if ( !v69 || (v69 & (v69 - 1)) != 0 )
                goto LABEL_109;
            }
            v4 += 8;
LABEL_186:
            v97 = *(unsigned __int8 **)v4;
            if ( **(_BYTE **)v4 == 17 )
              goto LABEL_187;
            goto LABEL_109;
          }
LABEL_197:
          v104 = v113 == v12;
          v102 = v103 & (v113 == v12);
          goto LABEL_198;
        }
        v12 = v4;
        goto LABEL_134;
      }
      goto LABEL_5;
    }
LABEL_183:
    v103 = v113 == (char *)v94;
    goto LABEL_131;
  }
  for ( i = *(unsigned __int8 **)a2; ; i = *v56 )
  {
    if ( !(unsigned __int8)sub_2B0D8B0(i) || (unsigned int)*i - 12 <= 1 )
      goto LABEL_4;
    v48 = *(unsigned __int8 **)(v9 + 8);
    if ( !(unsigned __int8)sub_2B0D8B0(v48) )
      break;
    if ( (unsigned int)*v48 - 12 <= 1 )
      break;
    v51 = *(unsigned __int8 **)(v49 + 16);
    if ( !(unsigned __int8)sub_2B0D8B0(v51) )
      break;
    if ( (unsigned int)*v51 - 12 <= 1 )
      break;
    v53 = *(unsigned __int8 **)(v52 + 24);
    if ( !(unsigned __int8)sub_2B0D8B0(v53) || (unsigned int)*v53 - 12 <= 1 )
      break;
    v56 = (unsigned __int8 **)(v54 + 32);
    if ( v55 == v56 )
    {
      v70 = (v113 - (char *)v56) >> 3;
      goto LABEL_127;
    }
  }
  v9 = v50;
LABEL_4:
  v103 = v113 == (char *)v9;
LABEL_5:
  v10 = v108;
  for ( j = v4; ; j += 32 )
  {
    if ( v6 != *((unsigned __int8 **)j + 1) )
    {
      v12 = j + 8;
LABEL_12:
      v104 = v113 == v12;
      v102 = v103 & (v113 == v12);
      goto LABEL_13;
    }
    if ( v6 != *((unsigned __int8 **)j + 2) )
    {
      v12 = j + 16;
      goto LABEL_12;
    }
    if ( v6 != *((unsigned __int8 **)j + 3) )
    {
      v12 = j + 24;
      goto LABEL_12;
    }
    v12 = j + 32;
    if ( !--v10 )
      break;
    if ( v6 != *((unsigned __int8 **)j + 4) )
      goto LABEL_12;
  }
  v71 = v113 - v12;
  if ( v113 - v12 == 16 )
  {
LABEL_135:
    if ( v6 != *(unsigned __int8 **)v12 )
    {
      v104 = v113 == v12;
      v102 = (v113 == v12) & v103;
      goto LABEL_198;
    }
    goto LABEL_195;
  }
  if ( v71 == 24 )
  {
    if ( v6 != *((unsigned __int8 **)j + 4) )
      goto LABEL_197;
LABEL_134:
    v12 += 8;
    goto LABEL_135;
  }
  if ( v71 == 8 )
    goto LABEL_196;
  v104 = 1;
  v102 = v103;
LABEL_13:
  v105 = v6;
  v13 = v6;
  v14 = v108;
  while ( *v13 == 17 )
  {
    if ( *((_DWORD *)v13 + 8) > 0x40u )
    {
      if ( (unsigned int)sub_C44630((__int64)(v13 + 24)) != 1 )
        break;
    }
    else
    {
      v15 = *((_QWORD *)v13 + 3);
      if ( !v15 || (v15 & (v15 - 1)) != 0 )
        break;
    }
    v58 = *((_QWORD *)v4 + 1);
    v59 = v4 + 8;
    if ( *(_BYTE *)v58 != 17 )
    {
LABEL_90:
      v4 = v59;
      break;
    }
    if ( *(_DWORD *)(v58 + 32) > 0x40u )
    {
      if ( (unsigned int)sub_C44630(v58 + 24) != 1 )
        goto LABEL_90;
    }
    else
    {
      v60 = *(_QWORD *)(v58 + 24);
      if ( !v60 || (v60 & (v60 - 1)) != 0 )
        goto LABEL_90;
    }
    v61 = *((_QWORD *)v4 + 2);
    v59 = v4 + 16;
    if ( *(_BYTE *)v61 != 17 )
      goto LABEL_90;
    if ( *(_DWORD *)(v61 + 32) > 0x40u )
    {
      if ( (unsigned int)sub_C44630(v61 + 24) != 1 )
        goto LABEL_90;
    }
    else
    {
      v62 = *(_QWORD *)(v61 + 24);
      if ( !v62 || (v62 & (v62 - 1)) != 0 )
        goto LABEL_90;
    }
    v63 = *((_QWORD *)v4 + 3);
    v59 = v4 + 24;
    if ( *(_BYTE *)v63 != 17 )
      goto LABEL_90;
    if ( *(_DWORD *)(v63 + 32) > 0x40u )
    {
      if ( (unsigned int)sub_C44630(v63 + 24) != 1 )
        goto LABEL_90;
      v4 += 32;
      if ( !--v14 )
      {
LABEL_99:
        v6 = v105;
        v65 = v113 - v4;
        if ( v113 - v4 == 16 )
          goto LABEL_106;
        if ( v65 == 24 )
        {
          v66 = *(unsigned __int8 **)v4;
          if ( **(_BYTE **)v4 != 17 )
            goto LABEL_109;
          goto LABEL_102;
        }
        if ( v65 == 8 )
          goto LABEL_186;
        v4 = v113;
        v7 = v105;
        break;
      }
    }
    else
    {
      v64 = *(_QWORD *)(v63 + 24);
      if ( !v64 || (v64 & (v64 - 1)) != 0 )
        goto LABEL_90;
      v4 += 32;
      if ( !--v14 )
        goto LABEL_99;
    }
    v13 = *(unsigned __int8 **)v4;
  }
LABEL_17:
  v16 = v108;
  while ( *v7 == 17 )
  {
    v17 = *((_DWORD *)v7 + 8);
    _RAX = *((_QWORD *)v7 + 3);
    v19 = 1LL << ((unsigned __int8)v17 - 1);
    if ( v17 > 0x40 )
    {
      if ( (*(_QWORD *)(_RAX + 8LL * ((v17 - 1) >> 6)) & v19) == 0 )
        break;
      v57 = (__int64)(v7 + 24);
      v107 = v17;
      v109 = sub_C44500(v57);
      LODWORD(_RAX) = sub_C44590(v57);
      v17 = v107;
      v20 = v109;
    }
    else
    {
      if ( (v19 & _RAX) == 0 )
        break;
      if ( v17 )
      {
        v20 = 64;
        if ( _RAX << (64 - (unsigned __int8)v17) != -1 )
        {
          _BitScanReverse64(&v21, ~(_RAX << (64 - (unsigned __int8)v17)));
          v20 = v21 ^ 0x3F;
        }
      }
      else
      {
        v20 = 0;
      }
      __asm { tzcnt   rax, rax }
      if ( (unsigned int)_RAX > v17 )
        LODWORD(_RAX) = *((_DWORD *)v7 + 8);
    }
    if ( v17 != v20 + (_DWORD)_RAX )
      break;
    v23 = *((_QWORD *)v5 + 1);
    v24 = v5 + 8;
    if ( *(_BYTE *)v23 != 17 )
      goto LABEL_60;
    v25 = *(_DWORD *)(v23 + 32);
    _RAX = *(_QWORD *)(v23 + 24);
    v27 = 1LL << ((unsigned __int8)v25 - 1);
    if ( v25 > 0x40 )
    {
      if ( (*(_QWORD *)(_RAX + 8LL * ((v25 - 1) >> 6)) & v27) == 0 )
        goto LABEL_60;
      v110 = sub_C44500(v23 + 24);
      LODWORD(_RAX) = sub_C44590(v23 + 24);
      v24 = v5 + 8;
      v28 = v110;
    }
    else
    {
      if ( (v27 & _RAX) == 0 )
        goto LABEL_60;
      if ( v25 )
      {
        v28 = 64;
        if ( _RAX << (64 - (unsigned __int8)v25) != -1 )
        {
          _BitScanReverse64(&v29, ~(_RAX << (64 - (unsigned __int8)v25)));
          v28 = v29 ^ 0x3F;
        }
      }
      else
      {
        v28 = 0;
      }
      __asm { tzcnt   rax, rax }
      if ( (unsigned int)_RAX > v25 )
        LODWORD(_RAX) = *(_DWORD *)(v23 + 32);
    }
    if ( v25 != v28 + (_DWORD)_RAX )
      goto LABEL_60;
    v31 = *((_QWORD *)v5 + 2);
    v24 = v5 + 16;
    if ( *(_BYTE *)v31 != 17 )
      goto LABEL_60;
    v32 = *(_DWORD *)(v31 + 32);
    _RAX = *(_QWORD *)(v31 + 24);
    v34 = 1LL << ((unsigned __int8)v32 - 1);
    if ( v32 > 0x40 )
    {
      if ( (*(_QWORD *)(_RAX + 8LL * ((v32 - 1) >> 6)) & v34) == 0 )
        goto LABEL_60;
      v111 = sub_C44500(v31 + 24);
      LODWORD(_RAX) = sub_C44590(v31 + 24);
      v24 = v5 + 16;
      v35 = v111;
    }
    else
    {
      if ( (v34 & _RAX) == 0 )
        goto LABEL_60;
      if ( v32 )
      {
        v35 = 64;
        if ( _RAX << (64 - (unsigned __int8)v32) != -1 )
        {
          _BitScanReverse64(&v36, ~(_RAX << (64 - (unsigned __int8)v32)));
          v35 = v36 ^ 0x3F;
        }
      }
      else
      {
        v35 = 0;
      }
      __asm { tzcnt   rax, rax }
      if ( (unsigned int)_RAX > v32 )
        LODWORD(_RAX) = *(_DWORD *)(v31 + 32);
    }
    if ( v32 != v35 + (_DWORD)_RAX || (v38 = *((_QWORD *)v5 + 3), v24 = v5 + 24, *(_BYTE *)v38 != 17) )
    {
LABEL_60:
      v5 = v24;
      v45 = 2;
      if ( v102 )
        goto LABEL_57;
      goto LABEL_121;
    }
    v39 = *(_DWORD *)(v38 + 32);
    _RAX = *(_QWORD *)(v38 + 24);
    v41 = 1LL << ((unsigned __int8)v39 - 1);
    if ( v39 > 0x40 )
    {
      if ( (*(_QWORD *)(_RAX + 8LL * ((v39 - 1) >> 6)) & v41) == 0 )
        goto LABEL_60;
      v112 = sub_C44500(v38 + 24);
      LODWORD(_RAX) = sub_C44590(v38 + 24);
      v24 = v5 + 24;
      v42 = v112;
    }
    else
    {
      if ( (v41 & _RAX) == 0 )
        goto LABEL_60;
      if ( v39 )
      {
        v42 = 64;
        if ( _RAX << (64 - (unsigned __int8)v39) != -1 )
        {
          _BitScanReverse64(&v43, ~(_RAX << (64 - (unsigned __int8)v39)));
          v42 = v43 ^ 0x3F;
        }
      }
      else
      {
        v42 = 0;
      }
      __asm { tzcnt   rax, rax }
      if ( (unsigned int)_RAX > v39 )
        LODWORD(_RAX) = *(_DWORD *)(v38 + 32);
    }
    if ( v39 != v42 + (_DWORD)_RAX )
      goto LABEL_60;
    v5 += 32;
    if ( !--v16 )
    {
      v106 = (v113 - v5) >> 3;
      goto LABEL_116;
    }
    v7 = *(unsigned __int8 **)v5;
  }
LABEL_56:
  v45 = 2;
  if ( !v102 )
  {
LABEL_121:
    v45 = 3;
    if ( v103 )
      goto LABEL_57;
    goto LABEL_122;
  }
LABEL_57:
  v46 = 2;
  if ( v113 != v5 )
    v46 = v113 == v4;
  return (v46 << 32) | v45;
}
