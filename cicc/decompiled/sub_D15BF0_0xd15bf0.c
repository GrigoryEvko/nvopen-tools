// Function: sub_D15BF0
// Address: 0xd15bf0
//
void __fastcall sub_D15BF0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        unsigned int a4,
        __int64 a5,
        _QWORD *a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  unsigned int v12; // r13d
  __int64 v13; // rdx
  __int64 v16; // rax
  unsigned int v17; // eax
  unsigned int v18; // eax
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // rcx
  int v21; // ecx
  bool v22; // zf
  int v23; // edx
  int v24; // eax
  unsigned int v25; // eax
  unsigned int v26; // r12d
  unsigned int v27; // esi
  unsigned int v28; // esi
  int v29; // eax
  int v30; // eax
  __int64 *v31; // rdx
  __int64 *v32; // rdx
  const void **v33; // rsi
  unsigned int v34; // eax
  unsigned __int64 v35; // rdx
  __int64 v36; // rdx
  __int64 v37; // rdi
  unsigned int v38; // r15d
  _QWORD *v39; // rcx
  unsigned int v40; // r12d
  unsigned int v41; // eax
  __int64 v42; // rsi
  unsigned __int64 v43; // rdx
  unsigned __int64 v44; // rax
  unsigned int v45; // edx
  unsigned int v46; // esi
  unsigned __int64 v47; // rax
  unsigned __int64 v48; // r15
  bool v49; // r14
  char v50; // cl
  unsigned __int64 v51; // rdx
  unsigned __int64 v52; // rdi
  unsigned int v53; // r12d
  char v54; // cl
  unsigned __int64 v55; // r12
  unsigned __int64 v56; // r12
  bool v57; // r13
  unsigned int v58; // edi
  __int64 v59; // rsi
  __int64 v60; // rdx
  __int64 v61; // rdx
  __int64 v62; // rdi
  unsigned int v63; // r15d
  unsigned __int64 *v64; // r12
  unsigned int v65; // r8d
  unsigned int v66; // eax
  __int64 v67; // rsi
  unsigned __int64 v68; // rdx
  unsigned __int64 v69; // rax
  __int64 *v70; // rdx
  unsigned __int64 v71; // rdx
  bool v72; // cc
  unsigned int v73; // esi
  unsigned __int64 v74; // rax
  __int64 *v75; // rdx
  unsigned int v76; // r15d
  int v77; // r12d
  unsigned int v78; // r14d
  unsigned __int64 v79; // rax
  __int64 v80; // rdx
  __int64 v81; // rdi
  unsigned int v82; // r15d
  unsigned __int64 **v83; // r12
  unsigned __int64 v84; // r15
  unsigned int v85; // eax
  unsigned int v86; // esi
  char v87; // cl
  unsigned __int64 v88; // r12
  int v89; // eax
  int v90; // eax
  __int64 v91; // rdi
  __int64 v92; // rdi
  unsigned int v93; // r12d
  __int64 v94; // rax
  unsigned int v95; // edx
  unsigned int v96; // r13d
  __int64 v97; // rsi
  unsigned __int64 v98; // rax
  unsigned int v99; // eax
  unsigned int v100; // r12d
  unsigned int v101; // edx
  unsigned __int64 v102; // rax
  unsigned __int64 v103; // rax
  unsigned __int64 v104; // rax
  unsigned __int64 v105; // rax
  unsigned __int64 v106; // rax
  unsigned __int64 v107; // rax
  unsigned __int64 v108; // rdx
  unsigned __int64 v109; // rax
  unsigned int v110; // r14d
  __int64 *v111; // rsi
  __int64 v112; // rdx
  _BYTE *v113; // rax
  __int64 v114; // rdx
  _BYTE *v115; // rax
  __int64 v116; // rdx
  _BYTE *v117; // rax
  unsigned int v118; // eax
  unsigned int v119; // r12d
  unsigned int v120; // r12d
  unsigned __int64 v121; // rdx
  int v122; // eax
  unsigned int v123; // edi
  int v124; // eax
  unsigned int v125; // r14d
  unsigned int v126; // edi
  __int64 v127; // rsi
  __int64 v128; // rdx
  __int64 v129; // rax
  unsigned int v132; // eax
  _BYTE *v133; // rax
  unsigned int v136; // eax
  int v137; // [rsp+8h] [rbp-B8h]
  int v138; // [rsp+8h] [rbp-B8h]
  __int64 v139; // [rsp+8h] [rbp-B8h]
  __int64 v140; // [rsp+10h] [rbp-B0h]
  unsigned int v141; // [rsp+10h] [rbp-B0h]
  unsigned int v142; // [rsp+10h] [rbp-B0h]
  __int64 v143; // [rsp+10h] [rbp-B0h]
  unsigned int v144; // [rsp+10h] [rbp-B0h]
  unsigned int v145; // [rsp+10h] [rbp-B0h]
  __int64 v146; // [rsp+18h] [rbp-A8h] BYREF
  unsigned __int64 v147; // [rsp+20h] [rbp-A0h] BYREF
  unsigned int v148; // [rsp+28h] [rbp-98h]
  unsigned __int64 v149; // [rsp+30h] [rbp-90h] BYREF
  unsigned int v150; // [rsp+38h] [rbp-88h]
  unsigned __int64 v151; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v152; // [rsp+48h] [rbp-78h]
  unsigned __int64 v153; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v154; // [rsp+58h] [rbp-68h]
  _QWORD v155[12]; // [rsp+60h] [rbp-60h] BYREF

  v146 = a2;
  v12 = *((_DWORD *)a6 + 2);
  v155[3] = a1;
  v155[0] = a9;
  v155[1] = &v146;
  v13 = a2;
  v155[2] = a7;
  v155[4] = a8;
  switch ( *(_BYTE *)a2 )
  {
    case '"':
    case 'U':
      if ( *(_BYTE *)a2 != 85 )
        return;
      v16 = *(_QWORD *)(a2 - 32);
      if ( !v16 || *(_BYTE *)v16 || *(_QWORD *)(v16 + 24) != *(_QWORD *)(a2 + 80) || (*(_BYTE *)(v16 + 33) & 0x20) == 0 )
        return;
      v17 = *(_DWORD *)(v16 + 36);
      if ( v17 == 67 )
      {
        if ( !a4 )
        {
          sub_D148C0((__int64)v155, v12, a3, 0);
          v118 = *(_DWORD *)(a7 + 24);
          if ( v118 <= 0x40 )
          {
            v119 = 64;
            _RDX = *(_QWORD *)(a7 + 16);
            __asm { tzcnt   rcx, rdx }
            if ( _RDX )
              v119 = _RCX;
            if ( v118 <= v119 )
              v119 = *(_DWORD *)(a7 + 24);
          }
          else
          {
            v119 = sub_C44590(a7 + 16);
          }
          v120 = v119 + 1;
          v154 = v12;
          if ( v120 > v12 )
            v120 = v12;
          if ( v12 > 0x40 )
            sub_C43690((__int64)&v153, 0, 0);
          else
            v153 = 0;
          if ( v120 )
          {
            if ( v120 > 0x40 )
            {
              sub_C43C90(&v153, 0, v120);
            }
            else
            {
              v121 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v120);
              if ( v154 > 0x40 )
                *(_QWORD *)v153 |= v121;
              else
                v153 |= v121;
            }
          }
          goto LABEL_3;
        }
        return;
      }
      if ( v17 <= 0x43 )
      {
        switch ( v17 )
        {
          case 0xFu:
            sub_C496B0((__int64)&v153, a5);
            if ( *((_DWORD *)a6 + 2) <= 0x40u )
              goto LABEL_6;
            goto LABEL_4;
          case 0x41u:
            if ( !a4 )
            {
              sub_D148C0((__int64)v155, v12, a3, 0);
              v18 = *(_DWORD *)(a7 + 24);
              if ( v18 > 0x40 )
              {
                v24 = sub_C444A0(a7 + 16);
              }
              else
              {
                v19 = *(_QWORD *)(a7 + 16);
                _BitScanReverse64(&v20, v19);
                v21 = v20 ^ 0x3F;
                v22 = v19 == 0;
                v23 = 64;
                if ( !v22 )
                  v23 = v21;
                v24 = v18 + v23 - 64;
              }
              v25 = v24 + 1;
              v154 = v12;
              if ( v25 > v12 )
                v25 = v12;
              v26 = v25;
              if ( v12 > 0x40 )
              {
                sub_C43690((__int64)&v153, 0, 0);
                v12 = v154;
              }
              else
              {
                v153 = 0;
              }
              v27 = v12 - v26;
              if ( v12 - v26 != v12 )
              {
                if ( v27 > 0x3F || v12 > 0x40 )
                  sub_C43C90(&v153, v27, v12);
                else
                  v153 |= 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v26) << v27;
              }
              goto LABEL_3;
            }
            break;
          case 0xEu:
            sub_C48440((__int64)&v153, (unsigned __int8 *)a5);
            if ( *((_DWORD *)a6 + 2) <= 0x40u )
              goto LABEL_6;
            goto LABEL_4;
        }
        return;
      }
      if ( v17 > 0x14A )
      {
        if ( v17 - 365 > 1 )
          return;
      }
      else if ( v17 <= 0x148 )
      {
        if ( v17 - 180 > 1 )
          return;
        if ( a4 == 2 )
        {
          if ( v12 )
          {
            v129 = v12 - 1;
            if ( ((unsigned int)v129 & v12) == 0 )
            {
              if ( v12 > 0x40 )
              {
                *(_QWORD *)*a6 = v129;
                memset(
                  (void *)(*a6 + 8LL),
                  0,
                  8 * (unsigned int)(((unsigned __int64)*((unsigned int *)a6 + 2) + 63) >> 6) - 8);
              }
              else
              {
                *a6 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v12) & v129;
              }
            }
          }
          return;
        }
        v91 = *(_QWORD *)(a2 + 32 * (2LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
        if ( *(_BYTE *)v91 == 17 )
        {
          v92 = v91 + 24;
        }
        else
        {
          if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v91 + 8) + 8LL) - 17 > 1 )
            return;
          if ( *(_BYTE *)v91 > 0x15u )
            return;
          v133 = sub_AD7630(v91, 0, a2);
          if ( !v133 || *v133 != 17 )
            return;
          v13 = a2;
          v92 = (__int64)(v133 + 24);
        }
        v139 = v13;
        v93 = sub_C459C0(v92, v12);
        v94 = *(_QWORD *)(v139 - 32);
        if ( !v94 || *(_BYTE *)v94 || *(_QWORD *)(v94 + 24) != *(_QWORD *)(v139 + 80) )
          BUG();
        if ( *(_DWORD *)(v94 + 36) == 181 )
          v93 = v12 - v93;
        if ( a4 )
        {
          if ( a4 != 1 )
            return;
          v95 = *(_DWORD *)(a5 + 8);
          v96 = v12 - v93;
          v154 = v95;
          if ( v95 <= 0x40 )
          {
            v153 = *(_QWORD *)a5;
LABEL_216:
            if ( v96 == v95 )
              v97 = 0;
            else
              v97 = v153 << v96;
            v98 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v95;
            if ( !v95 )
              v98 = 0;
            v153 = v97 & v98;
            goto LABEL_221;
          }
          sub_C43780((__int64)&v153, (const void **)a5);
          v95 = v154;
          if ( v154 <= 0x40 )
            goto LABEL_216;
          sub_C47690((__int64 *)&v153, v96);
LABEL_221:
          if ( *((_DWORD *)a6 + 2) > 0x40u && *a6 )
            j_j___libc_free_0_0(*a6);
          *a6 = v153;
          v99 = v154;
          v154 = 0;
          *((_DWORD *)a6 + 2) = v99;
          sub_969240((__int64 *)&v153);
          return;
        }
        v136 = *(_DWORD *)(a5 + 8);
        v154 = v136;
        if ( v136 > 0x40 )
        {
          sub_C43780((__int64)&v153, (const void **)a5);
          v136 = v154;
          if ( v154 > 0x40 )
          {
            sub_C482E0((__int64)&v153, v93);
            goto LABEL_221;
          }
        }
        else
        {
          v153 = *(_QWORD *)a5;
        }
        if ( v93 == v136 )
          v153 = 0;
        else
          v153 >>= v93;
        goto LABEL_221;
      }
      v100 = *(_DWORD *)(a5 + 8);
      if ( v100 <= 0x40 )
      {
        _RAX = *(_QWORD *)a5;
        __asm { tzcnt   rdx, rax }
        v132 = 64;
        if ( *(_QWORD *)a5 )
          v132 = _RDX;
        if ( v100 > v132 )
          v100 = v132;
      }
      else
      {
        v100 = sub_C44590(a5);
      }
      v154 = v12;
      if ( v12 > 0x40 )
      {
        sub_C43690((__int64)&v153, 0, 0);
        v12 = v154;
        if ( v154 == v100 )
          goto LABEL_3;
      }
      else
      {
        v153 = 0;
        if ( v12 == v100 )
          goto LABEL_6;
      }
      if ( v100 > 0x3F || v12 > 0x40 )
        sub_C43C90(&v153, v100, v12);
      else
        v153 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v100 + 64 - (unsigned __int8)v12) << v100;
      goto LABEL_3;
    case '*':
      v28 = *(_DWORD *)(a5 + 8);
      if ( v28 <= 0x40 )
      {
        v74 = *(_QWORD *)a5;
        if ( *(_QWORD *)a5 && (v74 & (v74 + 1)) == 0 )
          goto LABEL_191;
      }
      else
      {
        v140 = a2;
        v29 = sub_C445E0(a5);
        v13 = v140;
        v137 = v29;
        if ( v29 )
        {
          v30 = sub_C444A0(a5);
          v13 = v140;
          if ( v28 == v30 + v137 )
            goto LABEL_11;
        }
      }
      if ( (*(_BYTE *)(v13 + 7) & 0x40) != 0 )
        v31 = *(__int64 **)(v13 - 8);
      else
        v31 = (__int64 *)(v13 - 32LL * (*(_DWORD *)(v13 + 4) & 0x7FFFFFF));
      sub_D148C0((__int64)v155, v12, *v31, v31[4]);
      sub_D15AB0((__int64)&v153, a4, (unsigned __int8 *)a5, a7, a8);
      if ( *((_DWORD *)a6 + 2) > 0x40u )
        goto LABEL_4;
      goto LABEL_6;
    case ',':
      v73 = *(_DWORD *)(a5 + 8);
      if ( v73 > 0x40 )
      {
        v143 = a2;
        v89 = sub_C445E0(a5);
        v13 = v143;
        v138 = v89;
        if ( !v89 || (v90 = sub_C444A0(a5), v13 = v143, v73 != v90 + v138) )
        {
LABEL_151:
          if ( (*(_BYTE *)(v13 + 7) & 0x40) != 0 )
            v75 = *(__int64 **)(v13 - 8);
          else
            v75 = (__int64 *)(v13 - 32LL * (*(_DWORD *)(v13 + 4) & 0x7FFFFFF));
          sub_D148C0((__int64)v155, v12, *v75, v75[4]);
          sub_D15AE0((__int64)&v153, a4, (unsigned __int8 *)a5, a7, a8);
          if ( *((_DWORD *)a6 + 2) > 0x40u )
          {
LABEL_4:
            if ( *a6 )
              j_j___libc_free_0_0(*a6);
          }
LABEL_6:
          *a6 = v153;
          *((_DWORD *)a6 + 2) = v154;
          return;
        }
      }
      else
      {
        v74 = *(_QWORD *)a5;
        if ( !*(_QWORD *)a5 || (v74 & (v74 + 1)) != 0 )
          goto LABEL_151;
LABEL_191:
        if ( v12 <= 0x40 )
        {
LABEL_192:
          *a6 = v74;
          *((_DWORD *)a6 + 2) = *(_DWORD *)(a5 + 8);
          return;
        }
      }
LABEL_11:
      sub_C43990((__int64)a6, a5);
      return;
    case '.':
      v76 = *(_DWORD *)(a5 + 8);
      if ( v76 <= 0x40 )
      {
        v77 = *(_DWORD *)(a5 + 8);
        if ( *(_QWORD *)a5 )
        {
          _BitScanReverse64(&v88, *(_QWORD *)a5);
          v77 = v76 - 64 + (v88 ^ 0x3F);
        }
      }
      else
      {
        v77 = sub_C444A0(a5);
      }
      v154 = v12;
      v78 = v76 - v77;
      if ( v12 > 0x40 )
      {
        sub_C43690((__int64)&v153, 0, 0);
        if ( !v78 )
          goto LABEL_161;
        if ( v78 > 0x40 )
        {
LABEL_160:
          sub_C43C90(&v153, 0, v78);
          goto LABEL_161;
        }
        v108 = v153;
        v109 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v77 - (unsigned __int8)v76 + 64);
        if ( v154 > 0x40 )
        {
          *(_QWORD *)v153 |= v109;
          goto LABEL_161;
        }
      }
      else
      {
        v153 = 0;
        v79 = 0;
        if ( !v78 )
          goto LABEL_165;
        if ( v78 > 0x40 )
          goto LABEL_160;
        v108 = 0;
        v109 = 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v77 - (unsigned __int8)v76 + 64);
      }
      v153 = v108 | v109;
LABEL_161:
      if ( *((_DWORD *)a6 + 2) > 0x40u && *a6 )
        j_j___libc_free_0_0(*a6);
      v79 = v153;
      v12 = v154;
LABEL_165:
      *a6 = v79;
      *((_DWORD *)a6 + 2) = v12;
      return;
    case '6':
      if ( a4 )
        return;
      v80 = (*(_BYTE *)(a2 + 7) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
      v81 = *(_QWORD *)(v80 + 32);
      if ( *(_BYTE *)v81 != 17 )
      {
        v116 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v81 + 8) + 8LL) - 17;
        if ( (unsigned int)v116 > 1 )
          return;
        if ( *(_BYTE *)v81 > 0x15u )
          return;
        v117 = sub_AD7630(v81, 0, v116);
        v81 = (__int64)v117;
        if ( !v117 || *v117 != 17 )
          return;
      }
      v82 = *(_DWORD *)(v81 + 32);
      v83 = (unsigned __int64 **)(v81 + 24);
      v142 = v12 - 1;
      if ( v82 > 0x40 )
      {
        v122 = sub_C444A0(v81 + 24);
        v123 = v82;
        LODWORD(v84) = v12 - 1;
        if ( v123 - v122 <= 0x40 )
        {
          v84 = **v83;
          if ( v12 - 1 < v84 )
            LODWORD(v84) = v12 - 1;
        }
      }
      else
      {
        v84 = (unsigned __int64)*v83;
        if ( v12 - 1 < (unsigned __int64)*v83 )
          LODWORD(v84) = v12 - 1;
      }
      v85 = *(_DWORD *)(a5 + 8);
      v154 = v85;
      if ( v85 > 0x40 )
      {
        sub_C43780((__int64)&v153, (const void **)a5);
        v85 = v154;
        if ( v154 > 0x40 )
        {
          sub_C482E0((__int64)&v153, v84);
          goto LABEL_177;
        }
      }
      else
      {
        v153 = *(_QWORD *)a5;
      }
      if ( (_DWORD)v84 == v85 )
        v153 = 0;
      else
        v153 >>= v84;
LABEL_177:
      if ( *((_DWORD *)a6 + 2) > 0x40u && *a6 )
        j_j___libc_free_0_0(*a6);
      *a6 = v153;
      *((_DWORD *)a6 + 2) = v154;
      if ( ((*(_BYTE *)(v146 + 1) >> 1) & 2) != 0 )
      {
        v154 = v12;
        if ( v12 > 0x40 )
        {
          sub_C43690((__int64)&v153, 0, 0);
          v12 = v154;
          v142 = v154 - 1;
        }
        else
        {
          v153 = 0;
        }
        v86 = v142 - v84;
        if ( v142 - (_DWORD)v84 != v12 )
        {
          if ( v86 <= 0x3F && v12 <= 0x40 )
          {
            v87 = 63 - v84;
            goto LABEL_188;
          }
          sub_C43C90(&v153, v86, v12);
        }
      }
      else
      {
        if ( (*(_BYTE *)(v146 + 1) & 2) == 0 )
          return;
        v154 = v12;
        if ( v12 > 0x40 )
          sub_C43690((__int64)&v153, 0, 0);
        else
          v153 = 0;
        v86 = v154 - v84;
        if ( v154 != v154 - (_DWORD)v84 )
        {
          if ( v86 <= 0x3F && v154 <= 0x40 )
          {
            v87 = 64 - v84;
LABEL_188:
            v153 |= 0xFFFFFFFFFFFFFFFFLL >> v87 << v86;
            goto LABEL_88;
          }
          sub_C43C90(&v153, v86, v154);
        }
      }
      goto LABEL_88;
    case '7':
      if ( a4 )
        return;
      v61 = (*(_BYTE *)(a2 + 7) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
      v62 = *(_QWORD *)(v61 + 32);
      if ( *(_BYTE *)v62 != 17 )
      {
        v114 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v62 + 8) + 8LL) - 17;
        if ( (unsigned int)v114 > 1 )
          return;
        if ( *(_BYTE *)v62 > 0x15u )
          return;
        v115 = sub_AD7630(v62, 0, v114);
        v62 = (__int64)v115;
        if ( !v115 || *v115 != 17 )
          return;
      }
      v63 = *(_DWORD *)(v62 + 32);
      v64 = (unsigned __int64 *)(v62 + 24);
      v65 = v12 - 1;
      if ( v63 > 0x40 )
      {
        v124 = sub_C444A0(v62 + 24);
        v65 = v12 - 1;
        if ( v63 - v124 <= 0x40 && (unsigned __int64)(v12 - 1) >= *(_QWORD *)*v64 )
          v65 = *(_QWORD *)*v64;
      }
      else if ( v12 - 1 >= *v64 )
      {
        v65 = *v64;
      }
      v66 = *(_DWORD *)(a5 + 8);
      v154 = v66;
      if ( v66 > 0x40 )
      {
        v144 = v65;
        sub_C43780((__int64)&v153, (const void **)a5);
        v66 = v154;
        v65 = v144;
        if ( v154 > 0x40 )
        {
          sub_C47690((__int64 *)&v153, v144);
          v65 = v144;
          goto LABEL_120;
        }
      }
      else
      {
        v153 = *(_QWORD *)a5;
      }
      v67 = 0;
      if ( v66 != v65 )
        v67 = v153 << v65;
      v68 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v66;
      v22 = v66 == 0;
      v69 = 0;
      if ( !v22 )
        v69 = v68;
      v153 = v67 & v69;
LABEL_120:
      if ( *((_DWORD *)a6 + 2) > 0x40u && *a6 )
      {
        v141 = v65;
        j_j___libc_free_0_0(*a6);
        v65 = v141;
      }
      *a6 = v153;
      *((_DWORD *)a6 + 2) = v154;
      if ( (*(_BYTE *)(v146 + 1) & 2) == 0 )
        return;
      v154 = v12;
      if ( v12 > 0x40 )
      {
        v145 = v65;
        sub_C43690((__int64)&v153, 0, 0);
        v65 = v145;
      }
      else
      {
        v153 = 0;
      }
      if ( !v65 )
        goto LABEL_88;
      if ( v65 <= 0x40 )
      {
        v50 = 64 - v65;
        goto LABEL_86;
      }
      sub_C43C90(&v153, 0, v65);
      goto LABEL_88;
    case '8':
      if ( a4 )
        return;
      v36 = (*(_BYTE *)(a2 + 7) & 0x40) != 0 ? *(_QWORD *)(a2 - 8) : a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
      v37 = *(_QWORD *)(v36 + 32);
      if ( *(_BYTE *)v37 != 17 )
      {
        v112 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v37 + 8) + 8LL) - 17;
        if ( (unsigned int)v112 > 1 )
          return;
        if ( *(_BYTE *)v37 > 0x15u )
          return;
        v113 = sub_AD7630(v37, 0, v112);
        v37 = (__int64)v113;
        if ( !v113 || *v113 != 17 )
          return;
      }
      v38 = *(_DWORD *)(v37 + 32);
      v39 = (_QWORD *)(v37 + 24);
      v40 = v12 - 1;
      if ( v38 > 0x40 )
      {
        if ( v38 - (unsigned int)sub_C444A0(v37 + 24) <= 0x40 && (unsigned __int64)(v12 - 1) >= **(_QWORD **)(v37 + 24) )
          v40 = **(_QWORD **)(v37 + 24);
      }
      else if ( (unsigned __int64)(v12 - 1) >= *v39 )
      {
        v40 = *v39;
      }
      v41 = *(_DWORD *)(a5 + 8);
      v154 = v41;
      if ( v41 > 0x40 )
      {
        sub_C43780((__int64)&v153, (const void **)a5);
        v41 = v154;
        if ( v154 > 0x40 )
        {
          sub_C47690((__int64 *)&v153, v40);
          goto LABEL_69;
        }
      }
      else
      {
        v153 = *(_QWORD *)a5;
      }
      v42 = 0;
      if ( v41 != v40 )
        v42 = v153 << v40;
      v43 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v41;
      v22 = v41 == 0;
      v44 = 0;
      if ( !v22 )
        v44 = v43;
      v153 = v42 & v44;
LABEL_69:
      if ( *((_DWORD *)a6 + 2) > 0x40u && *a6 )
        j_j___libc_free_0_0(*a6);
      v152 = v12;
      *a6 = v153;
      *((_DWORD *)a6 + 2) = v154;
      if ( v12 > 0x40 )
      {
        sub_C43690((__int64)&v151, 0, 0);
        v45 = v152;
        v46 = v152 - v40;
        if ( v152 == v152 - v40 )
          goto LABEL_357;
      }
      else
      {
        v151 = 0;
        v45 = v12;
        v46 = v12 - v40;
        if ( v12 == v12 - v40 )
        {
          v47 = 0;
          goto LABEL_77;
        }
      }
      if ( v46 <= 0x3F && v45 <= 0x40 )
      {
        v47 = v151 | (0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v40) << v46);
LABEL_77:
        v48 = *(_QWORD *)a5 & v47;
LABEL_78:
        v49 = v48 == 0;
        goto LABEL_79;
      }
      sub_C43C90(&v151, v46, v45);
LABEL_357:
      if ( v152 <= 0x40 )
      {
        v47 = v151;
        goto LABEL_77;
      }
      sub_C43B90(&v151, (__int64 *)a5);
      v125 = v152;
      v48 = v151;
      v152 = 0;
      v154 = v125;
      v153 = v151;
      if ( v125 <= 0x40 )
        goto LABEL_78;
      v49 = v125 == (unsigned int)sub_C444A0((__int64)&v153);
      if ( v48 )
      {
        j_j___libc_free_0_0(v48);
        if ( v152 > 0x40 )
        {
          if ( v151 )
            j_j___libc_free_0_0(v151);
        }
      }
LABEL_79:
      if ( !v49 )
      {
        v126 = *((_DWORD *)a6 + 2);
        v127 = *a6;
        v128 = 1LL << ((unsigned __int8)v126 - 1);
        if ( v126 > 0x40 )
          *(_QWORD *)(v127 + 8LL * ((v126 - 1) >> 6)) |= v128;
        else
          *a6 = v128 | v127;
      }
      if ( (*(_BYTE *)(v146 + 1) & 2) == 0 )
        return;
      v154 = v12;
      if ( v12 > 0x40 )
        sub_C43690((__int64)&v153, 0, 0);
      else
        v153 = 0;
      if ( v40 )
      {
        if ( v40 > 0x40 )
        {
          sub_C43C90(&v153, 0, v40);
        }
        else
        {
          v50 = 64 - v40;
LABEL_86:
          v51 = 0xFFFFFFFFFFFFFFFFLL >> v50;
          if ( v154 > 0x40 )
            *(_QWORD *)v153 |= v51;
          else
            v153 |= v51;
        }
      }
LABEL_88:
      if ( *((_DWORD *)a6 + 2) > 0x40u )
        sub_C43BD0(a6, (__int64 *)&v153);
      else
        *a6 |= v153;
      if ( v154 <= 0x40 )
        return;
      v52 = v153;
      if ( !v153 )
        return;
      goto LABEL_147;
    case '9':
      if ( v12 <= 0x40 && *(_DWORD *)(a5 + 8) <= 0x40u )
      {
        *a6 = *(_QWORD *)a5;
        *((_DWORD *)a6 + 2) = *(_DWORD *)(a5 + 8);
      }
      else
      {
        sub_C43990((__int64)a6, a5);
        v13 = v146;
      }
      if ( (*(_BYTE *)(v13 + 7) & 0x40) != 0 )
        v70 = *(__int64 **)(v13 - 8);
      else
        v70 = (__int64 *)(v13 - 32LL * (*(_DWORD *)(v13 + 4) & 0x7FFFFFF));
      sub_D148C0((__int64)v155, v12, *v70, v70[4]);
      if ( !a4 )
      {
        v34 = *(_DWORD *)(a8 + 8);
        v152 = v34;
        if ( v34 <= 0x40 )
        {
          v35 = *(_QWORD *)a8;
          goto LABEL_137;
        }
        sub_C43780((__int64)&v151, (const void **)a8);
        v34 = v152;
        if ( v152 > 0x40 )
        {
LABEL_283:
          sub_C43D10((__int64)&v151);
          v34 = v152;
          v71 = v151;
          goto LABEL_140;
        }
LABEL_416:
        v35 = v151;
LABEL_137:
        v71 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v34) & ~v35;
        if ( !v34 )
          v71 = 0;
        v151 = v71;
LABEL_140:
        v72 = *((_DWORD *)a6 + 2) <= 0x40u;
        v154 = v34;
        v153 = v71;
        v152 = 0;
        if ( v72 )
        {
          *a6 &= v71;
        }
        else
        {
          sub_C43B90(a6, (__int64 *)&v153);
          v34 = v154;
        }
        if ( v34 > 0x40 && v153 )
          j_j___libc_free_0_0(v153);
        if ( v152 <= 0x40 )
          return;
        v52 = v151;
        if ( !v151 )
          return;
LABEL_147:
        j_j___libc_free_0_0(v52);
        return;
      }
      v101 = *(_DWORD *)(a8 + 8);
      v148 = v101;
      if ( v101 <= 0x40 )
      {
        v102 = *(_QWORD *)a8;
LABEL_238:
        v148 = 0;
        v103 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v101) & ~v102;
        if ( !v101 )
          v103 = 0;
        v147 = v103;
        goto LABEL_241;
      }
      sub_C43780((__int64)&v147, (const void **)a8);
      v101 = v148;
      if ( v148 <= 0x40 )
      {
        v102 = v147;
        goto LABEL_238;
      }
      sub_C43D10((__int64)&v147);
      v101 = v148;
      v103 = v147;
      v148 = 0;
      v150 = v101;
      v149 = v147;
      if ( v101 <= 0x40 )
      {
LABEL_241:
        v150 = 0;
        v104 = *(_QWORD *)a7 & v103;
        v149 = v104;
        goto LABEL_242;
      }
      v111 = (__int64 *)a7;
      goto LABEL_291;
    case ':':
      if ( v12 <= 0x40 && *(_DWORD *)(a5 + 8) <= 0x40u )
      {
        *a6 = *(_QWORD *)a5;
        *((_DWORD *)a6 + 2) = *(_DWORD *)(a5 + 8);
      }
      else
      {
        sub_C43990((__int64)a6, a5);
        v13 = v146;
      }
      if ( (*(_BYTE *)(v13 + 7) & 0x40) != 0 )
        v32 = *(__int64 **)(v13 - 8);
      else
        v32 = (__int64 *)(v13 - 32LL * (*(_DWORD *)(v13 + 4) & 0x7FFFFFF));
      sub_D148C0((__int64)v155, v12, *v32, v32[4]);
      v33 = (const void **)(a8 + 16);
      if ( !a4 )
      {
        v34 = *(_DWORD *)(a8 + 24);
        v152 = v34;
        if ( v34 <= 0x40 )
        {
          v35 = *(_QWORD *)(a8 + 16);
          goto LABEL_137;
        }
        sub_C43780((__int64)&v151, v33);
        v34 = v152;
        if ( v152 > 0x40 )
          goto LABEL_283;
        goto LABEL_416;
      }
      v101 = *(_DWORD *)(a8 + 24);
      v148 = v101;
      if ( v101 <= 0x40 )
      {
        v106 = *(_QWORD *)(a8 + 16);
LABEL_262:
        v148 = 0;
        v107 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v101) & ~v106;
        if ( !v101 )
          v107 = 0;
        v147 = v107;
        goto LABEL_265;
      }
      sub_C43780((__int64)&v147, v33);
      v101 = v148;
      if ( v148 <= 0x40 )
      {
        v106 = v147;
        goto LABEL_262;
      }
      sub_C43D10((__int64)&v147);
      v101 = v148;
      v148 = 0;
      v107 = v147;
      v111 = (__int64 *)(a7 + 16);
      v150 = v101;
      v149 = v147;
      if ( v101 <= 0x40 )
      {
LABEL_265:
        v150 = 0;
        v104 = *(_QWORD *)(a7 + 16) & v107;
        v149 = v104;
        goto LABEL_242;
      }
LABEL_291:
      sub_C43B90(&v149, v111);
      v101 = v150;
      v104 = v149;
      v150 = 0;
      v152 = v101;
      v151 = v149;
      if ( v101 <= 0x40 )
      {
LABEL_242:
        v105 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v101) & ~v104;
        if ( !v101 )
          v105 = 0;
        v151 = v105;
      }
      else
      {
        sub_C43D10((__int64)&v151);
        v101 = v152;
        v105 = v151;
      }
      v72 = *((_DWORD *)a6 + 2) <= 0x40u;
      v154 = v101;
      v153 = v105;
      v152 = 0;
      if ( v72 )
      {
        *a6 &= v105;
      }
      else
      {
        sub_C43B90(a6, (__int64 *)&v153);
        v101 = v154;
      }
      if ( v101 > 0x40 && v153 )
        j_j___libc_free_0_0(v153);
      if ( v152 > 0x40 && v151 )
        j_j___libc_free_0_0(v151);
      if ( v150 > 0x40 && v149 )
        j_j___libc_free_0_0(v149);
      if ( v148 > 0x40 )
      {
        v52 = v147;
        if ( v147 )
          goto LABEL_147;
      }
      return;
    case ';':
    case 'T':
      goto LABEL_9;
    case 'C':
      sub_C449B0((__int64)&v153, (const void **)a5, v12);
      if ( *((_DWORD *)a6 + 2) > 0x40u )
        goto LABEL_4;
      goto LABEL_6;
    case 'D':
      sub_C44740((__int64)&v153, (char **)a5, v12);
LABEL_3:
      if ( *((_DWORD *)a6 + 2) > 0x40u )
        goto LABEL_4;
      goto LABEL_6;
    case 'E':
      sub_C44740((__int64)&v153, (char **)a5, v12);
      if ( *((_DWORD *)a6 + 2) > 0x40u && *a6 )
        j_j___libc_free_0_0(*a6);
      *a6 = v153;
      *((_DWORD *)a6 + 2) = v154;
      v53 = *(_DWORD *)(a5 + 8);
      v152 = v53;
      if ( v53 > 0x40 )
      {
        sub_C43690((__int64)&v151, 0, 0);
        v54 = v12 - v53;
        v12 = v152 + v12 - v53;
        if ( v152 == v12 )
          goto LABEL_274;
        v53 = v152;
      }
      else
      {
        v151 = 0;
        v54 = v12 - v53;
        if ( v12 == v53 )
        {
          v55 = 0;
          goto LABEL_101;
        }
      }
      if ( v12 <= 0x3F && v53 <= 0x40 )
      {
        v55 = v151 | (0xFFFFFFFFFFFFFFFFLL >> (v54 + 64) << v12);
LABEL_101:
        v56 = *(_QWORD *)a5 & v55;
LABEL_102:
        v57 = v56 == 0;
        goto LABEL_103;
      }
      sub_C43C90(&v151, v12, v53);
      v12 = v152;
LABEL_274:
      if ( v12 <= 0x40 )
      {
        v55 = v151;
        goto LABEL_101;
      }
      sub_C43B90(&v151, (__int64 *)a5);
      v110 = v152;
      v56 = v151;
      v152 = 0;
      v154 = v110;
      v153 = v151;
      if ( v110 <= 0x40 )
        goto LABEL_102;
      v57 = v110 == (unsigned int)sub_C444A0((__int64)&v153);
      if ( v56 )
      {
        j_j___libc_free_0_0(v56);
        if ( v152 > 0x40 )
        {
          if ( v151 )
            j_j___libc_free_0_0(v151);
        }
      }
LABEL_103:
      if ( !v57 )
      {
        v58 = *((_DWORD *)a6 + 2);
        v59 = *a6;
        v60 = 1LL << ((unsigned __int8)v58 - 1);
        if ( v58 > 0x40 )
          *(_QWORD *)(v59 + 8LL * ((v58 - 1) >> 6)) |= v60;
        else
          *a6 = v60 | v59;
      }
      return;
    case 'V':
      if ( a4 )
        goto LABEL_9;
      return;
    case 'Z':
      if ( !a4 )
        goto LABEL_9;
      return;
    case '[':
    case '\\':
      if ( a4 > 1 )
        return;
LABEL_9:
      if ( v12 > 0x40 || *(_DWORD *)(a5 + 8) > 0x40u )
        goto LABEL_11;
      v74 = *(_QWORD *)a5;
      goto LABEL_192;
    default:
      return;
  }
}
