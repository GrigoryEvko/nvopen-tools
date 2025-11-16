// Function: sub_2DD6EE0
// Address: 0x2dd6ee0
//
__int64 __fastcall sub_2DD6EE0(__int64 a1, _QWORD *a2, __int64 *a3, _QWORD **a4, char a5, unsigned int a6)
{
  int v9; // eax
  __int64 v10; // r9
  int v11; // ecx
  unsigned int v12; // edi
  __int64 v13; // rax
  unsigned __int64 v17; // r12
  __int64 *v18; // rax
  unsigned __int8 v19; // bl
  __int64 v20; // rax
  __int64 v21; // rdx
  int v22; // eax
  _BYTE *v23; // rsi
  _BYTE *v24; // rsi
  unsigned __int64 v25; // rdx
  _BYTE *v26; // rsi
  unsigned __int8 v27; // al
  __int64 v28; // rdi
  int v29; // eax
  unsigned int v30; // edx
  unsigned int v31; // r10d
  unsigned int v32; // esi
  __int64 v33; // r11
  int v34; // ecx
  unsigned __int64 v35; // r9
  __int64 v36; // rdx
  unsigned __int64 v37; // r9
  unsigned __int64 v38; // rbx
  unsigned __int8 v41; // r14
  char v42; // r13
  char v43; // bl
  __int64 v44; // r12
  char v45; // al
  __int64 v46; // rdx
  __int64 v47; // rsi
  __int64 v48; // rdi
  __int64 v49; // rbx
  __int64 v50; // r13
  __int64 v51; // rdi
  unsigned __int64 v52; // rdx
  const char *v53; // r9
  size_t v54; // r8
  unsigned __int64 v55; // rax
  _QWORD *v56; // rdx
  __int64 v57; // rax
  unsigned int v58; // eax
  __int64 v59; // rax
  __int64 v60; // r8
  __int64 v61; // rcx
  char v62; // al
  char v63; // al
  unsigned int v64; // ebx
  int v65; // eax
  unsigned int v66; // r8d
  unsigned int v67; // esi
  int v68; // ebx
  __int64 v69; // r9
  unsigned __int64 v70; // rdi
  __int64 v71; // rdx
  unsigned __int64 v72; // rdi
  unsigned __int64 v73; // r10
  _QWORD *v76; // rax
  _BYTE *v77; // rsi
  _BYTE *v78; // rsi
  __int64 v79; // rax
  _BYTE *v80; // rsi
  __int64 v81; // rax
  _QWORD *v82; // rdi
  __int64 v83; // rdx
  __int64 v84; // [rsp+8h] [rbp-1B8h]
  __int64 v85; // [rsp+10h] [rbp-1B0h]
  __int64 *v86; // [rsp+18h] [rbp-1A8h]
  __int64 v87; // [rsp+20h] [rbp-1A0h]
  unsigned __int8 v89; // [rsp+2Bh] [rbp-195h]
  const char *v92; // [rsp+40h] [rbp-180h]
  __int64 v93; // [rsp+40h] [rbp-180h]
  char v94; // [rsp+48h] [rbp-178h]
  __int64 **v95; // [rsp+48h] [rbp-178h]
  __int64 v96; // [rsp+50h] [rbp-170h]
  char n; // [rsp+70h] [rbp-150h]
  size_t na; // [rsp+70h] [rbp-150h]
  void *srca; // [rsp+78h] [rbp-148h]
  void *src; // [rsp+78h] [rbp-148h]
  void *srcb; // [rsp+78h] [rbp-148h]
  void *srcc; // [rsp+78h] [rbp-148h]
  const char *srcd; // [rsp+78h] [rbp-148h]
  unsigned int v106; // [rsp+80h] [rbp-140h]
  char v107; // [rsp+80h] [rbp-140h]
  char v108; // [rsp+84h] [rbp-13Ch]
  int v109; // [rsp+84h] [rbp-13Ch]
  int v110; // [rsp+88h] [rbp-138h]
  unsigned __int8 *v111; // [rsp+88h] [rbp-138h]
  int v112; // [rsp+88h] [rbp-138h]
  int v113; // [rsp+88h] [rbp-138h]
  unsigned __int8 v114; // [rsp+90h] [rbp-130h]
  __int64 v115; // [rsp+98h] [rbp-128h]
  __int64 v116[2]; // [rsp+A0h] [rbp-120h] BYREF
  _QWORD *v117; // [rsp+B0h] [rbp-110h] BYREF
  _BYTE *v118; // [rsp+B8h] [rbp-108h]
  _BYTE *v119; // [rsp+C0h] [rbp-100h]
  __int64 *v120; // [rsp+D0h] [rbp-F0h] BYREF
  _BYTE *v121; // [rsp+D8h] [rbp-E8h]
  _BYTE *v122; // [rsp+E0h] [rbp-E0h]
  unsigned __int64 v123; // [rsp+F0h] [rbp-D0h] BYREF
  _BYTE *v124; // [rsp+F8h] [rbp-C8h]
  _BYTE *v125; // [rsp+100h] [rbp-C0h]
  unsigned __int64 v126[2]; // [rsp+110h] [rbp-B0h] BYREF
  _QWORD v127[2]; // [rsp+120h] [rbp-A0h] BYREF
  __int64 v128[4]; // [rsp+130h] [rbp-90h] BYREF
  __int16 v129; // [rsp+150h] [rbp-70h]
  unsigned __int64 v130; // [rsp+160h] [rbp-60h] BYREF
  __int64 v131; // [rsp+168h] [rbp-58h]
  unsigned __int64 v132; // [rsp+170h] [rbp-50h]
  unsigned int v133; // [rsp+178h] [rbp-48h]
  __int16 v134; // [rsp+180h] [rbp-40h]

  v96 = sub_BCB2D0(*a4);
  v86 = (__int64 *)sub_BCB2B0(*a4);
  v84 = (__int64)(a4 + 39);
  v9 = *((_DWORD *)a3 + 16);
  if ( !v9 )
    return 0;
  v10 = *a3;
  v11 = -v9;
  v12 = (unsigned int)(v9 - 1) >> 6;
  v13 = 0;
  while ( 1 )
  {
    _RDX = *(_QWORD *)(v10 + 8 * v13);
    if ( v12 == (_DWORD)v13 )
      _RDX = (0xFFFFFFFFFFFFFFFFLL >> v11) & *(_QWORD *)(v10 + 8 * v13);
    if ( _RDX )
      break;
    if ( v12 + 1 == ++v13 )
      return 0;
  }
  __asm { tzcnt   rdx, rdx }
  v89 = 0;
  LODWORD(_RDX) = ((_DWORD)v13 << 6) + _RDX;
  v87 = (int)_RDX;
  if ( (_DWORD)_RDX != -1 )
  {
LABEL_10:
    v117 = 0;
    v17 = 0;
    v118 = 0;
    v119 = 0;
    v120 = 0;
    v121 = 0;
    v122 = 0;
    v123 = 0;
    v124 = 0;
    v125 = 0;
    v115 = v87;
    v85 = 0;
    v92 = 0;
    v110 = 0;
    v94 = 0;
    v114 = 0;
    while ( 1 )
    {
      v18 = (__int64 *)(8 * v115 + *a2);
      v128[0] = *(_QWORD *)(*v18 + 24);
      v19 = sub_AE5270(v84, *v18);
      srca = (void *)v128[0];
      v106 = ((v17 + (1 << v19) - 1) & -(1 << v19)) - v17;
      v108 = sub_AE5020(v84, v128[0]);
      v20 = sub_9208B0(v84, (__int64)srca);
      v131 = v21;
      v130 = ((1LL << v108) + ((unsigned __int64)(v20 + 7) >> 3) - 1) >> v108 << v108;
      v17 += v106 + sub_CA1930(&v130);
      if ( *(unsigned int *)(a1 + 8) < v17 )
      {
LABEL_42:
        v41 = v114;
        goto LABEL_44;
      }
      v22 = v110;
      if ( v106 )
      {
        v76 = sub_BCD420(v86, v106);
        v77 = v118;
        v130 = (unsigned __int64)v76;
        if ( v118 == v119 )
        {
          sub_9183A0((__int64)&v117, v118, &v130);
          v78 = v118;
        }
        else
        {
          if ( v118 )
          {
            *(_QWORD *)v118 = v76;
            v77 = v118;
          }
          v78 = v77 + 8;
          v118 = v78;
        }
        v79 = sub_AC9350(*((__int64 ***)v78 - 1));
        v80 = v121;
        v130 = v79;
        if ( v121 == v122 )
        {
          sub_262AD50((__int64)&v120, v121, &v130);
        }
        else
        {
          if ( v121 )
          {
            *(_QWORD *)v121 = v79;
            v80 = v121;
          }
          v121 = v80 + 8;
        }
        v23 = v118;
        v22 = v110 + 1;
        if ( v118 == v119 )
        {
LABEL_100:
          v112 = v22;
          sub_918210((__int64)&v117, v23, v128);
          v22 = v112;
          goto LABEL_17;
        }
      }
      else
      {
        v23 = v118;
        if ( v118 == v119 )
          goto LABEL_100;
      }
      if ( v23 )
      {
        *(_QWORD *)v23 = v128[0];
        v23 = v118;
      }
      v118 = v23 + 8;
LABEL_17:
      v24 = v121;
      v25 = *(_QWORD *)(*(_QWORD *)(*a2 + 8 * v115) - 32LL);
      v130 = v25;
      if ( v121 == v122 )
      {
        v113 = v22;
        sub_262AD50((__int64)&v120, v121, &v130);
        v22 = v113;
      }
      else
      {
        if ( v121 )
        {
          *(_QWORD *)v121 = v25;
          v24 = v121;
        }
        v121 = v24 + 8;
      }
      LODWORD(v130) = v22;
      v110 = v22 + 1;
      v26 = v124;
      if ( v124 == v125 )
      {
        sub_C88AB0((__int64)&v123, v124, &v130);
      }
      else
      {
        if ( v124 )
        {
          *(_DWORD *)v124 = v22;
          v26 = v124;
        }
        v124 = v26 + 4;
      }
      v27 = v114;
      if ( v114 < v19 )
        v27 = v19;
      v114 = v27;
      v28 = *(_QWORD *)(*a2 + 8 * v115);
      if ( (*(_BYTE *)(v28 + 32) & 0xF) == 0 && !v94 )
      {
        v94 = 1;
        v92 = sub_BD5D20(v28);
        v85 = v83;
      }
      v29 = *((_DWORD *)a3 + 16);
      v30 = v115 + 1;
      if ( v29 == (_DWORD)v115 + 1 || (v31 = v30 >> 6, v32 = (unsigned int)(v29 - 1) >> 6, v30 >> 6 > v32) )
      {
LABEL_43:
        v115 = -1;
        v41 = v114;
LABEL_44:
        if ( (unsigned __int64)(v118 - (_BYTE *)v117) <= 8 )
        {
          if ( v123 )
            j_j___libc_free_0(v123);
          if ( v120 )
            j_j___libc_free_0((unsigned __int64)v120);
          if ( v117 )
            j_j___libc_free_0((unsigned __int64)v117);
        }
        else
        {
          v42 = v94;
          v43 = v94 == 0 ? 7 : 0;
          v95 = (__int64 **)sub_BD0B90(*a4, v117, (v118 - (_BYTE *)v117) >> 3, 1);
          v44 = sub_AD24A0(v95, v120, (v121 - (_BYTE *)v120) >> 3);
          v45 = *(_BYTE *)(a1 + 24);
          if ( v45 && v42 )
          {
            v128[0] = (__int64)"_MergedGlobals_";
            v129 = 1283;
            v128[2] = (__int64)v92;
            v128[3] = v85;
          }
          else
          {
            HIBYTE(v129) = 1;
            v128[0] = (__int64)"_MergedGlobals";
            if ( !v45 )
              v43 = 8;
            LOBYTE(v129) = 3;
          }
          BYTE4(v130) = 1;
          LODWORD(v130) = a6;
          v111 = (unsigned __int8 *)sub_BD2C40(88, unk_3F0FAE8);
          if ( v111 )
            sub_B30000((__int64)v111, (__int64)a4, v95, a5, v43, v44, (__int64)v128, 0, 0, v130, 0);
          sub_B2F770((__int64)v111, v41);
          v46 = 0;
          v47 = 0;
          v48 = *(_QWORD *)(*a2 + 8 * v87);
          if ( (*(_BYTE *)(v48 + 35) & 4) != 0 )
            v47 = sub_B31D10(v48, 0, 0);
          sub_B31A00((__int64)v111, v47, v46);
          v93 = sub_AE4AC0(v84, (__int64)v95);
          if ( v87 != v115 )
          {
            v49 = v87;
            v50 = 0;
            while ( 1 )
            {
              v51 = *(_QWORD *)(*a2 + 8 * v49);
              v109 = *(_BYTE *)(v51 + 32) & 0xF;
              v53 = sub_BD5D20(v51);
              v54 = v52;
              v55 = v52;
              v126[0] = (unsigned __int64)v127;
              if ( &v53[v52] && !v53 )
                sub_426248((__int64)"basic_string::_M_construct null not valid");
              v130 = v52;
              if ( v52 > 0xF )
                break;
              if ( v52 == 1 )
              {
                LOBYTE(v127[0]) = *v53;
                v56 = v127;
              }
              else
              {
                if ( v52 )
                {
                  v82 = v127;
                  goto LABEL_111;
                }
                v56 = v127;
              }
LABEL_61:
              v126[1] = v55;
              *((_BYTE *)v56 + v55) = 0;
              v57 = *(_QWORD *)(*a2 + 8 * v49);
              n = *(_BYTE *)(v57 + 33) & 3;
              v107 = (*(_BYTE *)(v57 + 32) >> 4) & 3;
              LOBYTE(v57) = *(_BYTE *)(v93 + 16LL * *(unsigned int *)(v123 + v50) + 32);
              v130 = *(_QWORD *)(v93 + 16LL * *(unsigned int *)(v123 + v50) + 24);
              LOBYTE(v131) = v57;
              v58 = sub_CA1930(&v130);
              sub_B9E560((__int64)v111, *(_QWORD *)(*a2 + 8 * v49), v58);
              v116[0] = sub_AD64C0(v96, 0, 0);
              v116[1] = sub_AD64C0(v96, *(unsigned int *)(v123 + v50), 0);
              LOBYTE(v134) = 0;
              v59 = sub_AD9FD0((__int64)v95, v111, v116, 2, 3u, (__int64)&v130, 0);
              v60 = v59;
              if ( (_BYTE)v134 )
              {
                LOBYTE(v134) = 0;
                if ( v133 > 0x40 && v132 )
                {
                  srcb = (void *)v59;
                  j_j___libc_free_0_0(v132);
                  v60 = (__int64)srcb;
                }
                if ( (unsigned int)v131 > 0x40 && v130 )
                {
                  srcc = (void *)v60;
                  j_j___libc_free_0_0(v130);
                  v60 = (__int64)srcc;
                }
              }
              src = (void *)v60;
              sub_BD84D0(*(_QWORD *)(*a2 + 8 * v49), v60);
              sub_B30290(*(_QWORD *)(*a2 + 8 * v49));
              if ( *(_BYTE *)(a1 + 24) != 1 || !v109 )
              {
                v134 = 260;
                v130 = (unsigned __int64)v126;
                v61 = sub_B30500(
                        (_QWORD *)v117[*(unsigned int *)(v123 + v50)],
                        a6,
                        v109,
                        (__int64)&v130,
                        (__int64)src,
                        (__int64)a4);
                v62 = *(_BYTE *)(v61 + 32) & 0xCF | (16 * (v107 & 3));
                *(_BYTE *)(v61 + 32) = v62;
                if ( (v62 & 0xFu) - 7 <= 1 || (v62 & 0x30) != 0 && (v62 & 0xF) != 9 )
                {
                  v63 = *(_BYTE *)(v61 + 33) | 0x40;
                  *(_BYTE *)(v61 + 33) = v63;
                }
                else
                {
                  v63 = *(_BYTE *)(v61 + 33);
                }
                *(_BYTE *)(v61 + 33) = n | v63 & 0xFC;
              }
              if ( (_QWORD *)v126[0] != v127 )
                j_j___libc_free_0(v126[0]);
              v64 = v49 + 1;
              v65 = *((_DWORD *)a3 + 16);
              if ( v65 == v64 || (v66 = v64 >> 6, v67 = (unsigned int)(v65 - 1) >> 6, v64 >> 6 > v67) )
              {
LABEL_101:
                v49 = -1;
              }
              else
              {
                v68 = v64 & 0x3F;
                v69 = *a3;
                v70 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v68);
                if ( v68 == 0 )
                  v70 = 0;
                v71 = v66;
                v72 = ~v70;
                v73 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v65;
                while ( 1 )
                {
                  _RAX = *(_QWORD *)(v69 + 8 * v71);
                  if ( v66 == (_DWORD)v71 )
                    _RAX = v72 & *(_QWORD *)(v69 + 8 * v71);
                  if ( v67 == (_DWORD)v71 )
                    _RAX &= v73;
                  if ( _RAX )
                    break;
                  if ( v67 < (unsigned int)++v71 )
                    goto LABEL_101;
                }
                __asm { tzcnt   rax, rax }
                v49 = ((_DWORD)v71 << 6) + (int)_RAX;
              }
              v50 += 4;
              if ( v49 == v115 )
                goto LABEL_82;
            }
            na = v52;
            srcd = v53;
            v81 = sub_22409D0((__int64)v126, &v130, 0);
            v53 = srcd;
            v54 = na;
            v126[0] = v81;
            v82 = (_QWORD *)v81;
            v127[0] = v130;
LABEL_111:
            memcpy(v82, v53, v54);
            v55 = v130;
            v56 = (_QWORD *)v126[0];
            goto LABEL_61;
          }
LABEL_82:
          if ( v123 )
            j_j___libc_free_0(v123);
          if ( v120 )
            j_j___libc_free_0((unsigned __int64)v120);
          if ( v117 )
            j_j___libc_free_0((unsigned __int64)v117);
          v89 = 1;
        }
        if ( v115 != -1 )
        {
          v87 = v115;
          goto LABEL_10;
        }
        return v89;
      }
      v33 = *a3;
      v34 = 64 - (v30 & 0x3F);
      v35 = 0xFFFFFFFFFFFFFFFFLL >> v34;
      if ( v34 == 64 )
        v35 = 0;
      v36 = v31;
      v37 = ~v35;
      v38 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v29;
      while ( 1 )
      {
        _RAX = *(_QWORD *)(v33 + 8 * v36);
        if ( v31 == (_DWORD)v36 )
          _RAX = v37 & *(_QWORD *)(v33 + 8 * v36);
        if ( v32 == (_DWORD)v36 )
          _RAX &= v38;
        if ( _RAX )
          break;
        if ( v32 < (unsigned int)++v36 )
          goto LABEL_43;
      }
      __asm { tzcnt   rax, rax }
      LODWORD(_RAX) = ((_DWORD)v36 << 6) + _RAX;
      v115 = (int)_RAX;
      if ( (_DWORD)_RAX == -1 )
        goto LABEL_42;
    }
  }
  return v89;
}
