// Function: sub_1F75B00
// Address: 0x1f75b00
//
__int64 *__fastcall sub_1F75B00(
        __int64 **a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        double a7,
        double a8,
        __m128i a9,
        char a10,
        int a11,
        int a12,
        __int64 a13,
        unsigned int a14,
        unsigned int a15,
        __int64 a16)
{
  __int64 v20; // rax
  __int64 v21; // r15
  char v22; // di
  const void **v23; // rdx
  __int64 *v24; // rax
  unsigned int v25; // eax
  int v27; // ecx
  __int64 v28; // r8
  int v29; // r9d
  unsigned int v30; // r10d
  __int16 v31; // ax
  __int64 *result; // rax
  int v34; // ecx
  __int64 v35; // r8
  __int64 v36; // r11
  __int64 v37; // rax
  unsigned int v38; // r10d
  __int64 v39; // r9
  int v40; // r15d
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  unsigned int v44; // r10d
  __int64 *v45; // r14
  unsigned int v46; // r14d
  _QWORD *v47; // r15
  int v48; // r14d
  __int64 v49; // r15
  int v50; // eax
  unsigned __int64 v51; // rdi
  unsigned int v52; // eax
  __int64 v53; // rax
  __int64 *v54; // rdx
  __int64 v55; // rsi
  unsigned __int64 v56; // r11
  int v57; // r11d
  unsigned __int64 v58; // rax
  unsigned __int64 v59; // rcx
  __int64 v60; // rdx
  __int64 v61; // rax
  unsigned int v62; // ecx
  __int64 v63; // rax
  __int64 *v64; // rdx
  __int64 v65; // rax
  unsigned __int64 v66; // rax
  unsigned __int64 v67; // rcx
  int v68; // eax
  const void **v69; // rsi
  unsigned int v70; // eax
  unsigned int v71; // eax
  bool v72; // al
  __int64 v73; // rdx
  unsigned int v74; // r14d
  unsigned int v76; // eax
  int v77; // eax
  unsigned int v78; // eax
  unsigned int v79; // eax
  bool v80; // al
  _QWORD *v81; // rdi
  __int64 *v82; // rax
  __int64 *v83; // rcx
  unsigned int v84; // edx
  __int64 *v85; // rdi
  __int128 *v86; // rax
  int v87; // r8d
  unsigned int v89; // eax
  __int64 v90; // [rsp+10h] [rbp-C0h]
  unsigned int v91; // [rsp+10h] [rbp-C0h]
  unsigned int v92; // [rsp+10h] [rbp-C0h]
  unsigned int v93; // [rsp+18h] [rbp-B8h]
  _QWORD *v94; // [rsp+18h] [rbp-B8h]
  bool v95; // [rsp+18h] [rbp-B8h]
  __int64 v96; // [rsp+20h] [rbp-B0h]
  __int64 v97; // [rsp+20h] [rbp-B0h]
  unsigned int v98; // [rsp+20h] [rbp-B0h]
  unsigned int v99; // [rsp+20h] [rbp-B0h]
  unsigned int v100; // [rsp+20h] [rbp-B0h]
  _QWORD *v101; // [rsp+20h] [rbp-B0h]
  unsigned int v102; // [rsp+20h] [rbp-B0h]
  unsigned int v103; // [rsp+28h] [rbp-A8h]
  unsigned int v104; // [rsp+28h] [rbp-A8h]
  const void **v105; // [rsp+28h] [rbp-A8h]
  unsigned int v106; // [rsp+28h] [rbp-A8h]
  unsigned int v107; // [rsp+28h] [rbp-A8h]
  __int64 v108; // [rsp+28h] [rbp-A8h]
  __int64 v109; // [rsp+28h] [rbp-A8h]
  __int64 v110; // [rsp+28h] [rbp-A8h]
  bool v111; // [rsp+28h] [rbp-A8h]
  __int64 v112; // [rsp+28h] [rbp-A8h]
  __int64 v113; // [rsp+30h] [rbp-A0h]
  __int64 v114; // [rsp+30h] [rbp-A0h]
  unsigned int v115; // [rsp+30h] [rbp-A0h]
  unsigned int v116; // [rsp+30h] [rbp-A0h]
  __int64 v117; // [rsp+30h] [rbp-A0h]
  __int64 v118; // [rsp+30h] [rbp-A0h]
  const void **v119; // [rsp+30h] [rbp-A0h]
  __int64 v120; // [rsp+30h] [rbp-A0h]
  unsigned int v121; // [rsp+38h] [rbp-98h]
  _QWORD *v122; // [rsp+38h] [rbp-98h]
  _QWORD v123[2]; // [rsp+40h] [rbp-90h] BYREF
  unsigned int v124; // [rsp+50h] [rbp-80h] BYREF
  const void **v125; // [rsp+58h] [rbp-78h]
  _QWORD *v126; // [rsp+60h] [rbp-70h] BYREF
  unsigned int v127; // [rsp+68h] [rbp-68h]
  _QWORD *v128; // [rsp+70h] [rbp-60h] BYREF
  unsigned int v129; // [rsp+78h] [rbp-58h]
  unsigned __int64 v130; // [rsp+80h] [rbp-50h] BYREF
  __int64 v131; // [rsp+88h] [rbp-48h]
  __int64 v132; // [rsp+90h] [rbp-40h]
  __int64 v133; // [rsp+98h] [rbp-38h]

  v20 = *(_QWORD *)(a2 + 40) + 16LL * (unsigned int)a3;
  v21 = a13;
  v123[0] = a4;
  v22 = *(_BYTE *)v20;
  v23 = *(const void ***)(v20 + 8);
  v123[1] = a5;
  v24 = *a1;
  LOBYTE(v124) = v22;
  v125 = v23;
  v113 = (__int64)v24;
  if ( !v22 )
  {
    v105 = v23;
    if ( sub_1F58D20((__int64)&v124) )
    {
      LOBYTE(v130) = sub_1F596B0((__int64)&v124);
      v22 = v130;
      v131 = v60;
      if ( (_BYTE)v130 )
        goto LABEL_3;
    }
    else
    {
      LOBYTE(v130) = 0;
      v131 = (__int64)v105;
    }
    v52 = sub_1F58D40((__int64)&v130);
    v30 = 0;
    v121 = v52;
    v31 = *(_WORD *)(v21 + 24);
    if ( v31 != 118 )
      goto LABEL_4;
    goto LABEL_32;
  }
  if ( (unsigned __int8)(v22 - 14) <= 0x5Fu )
  {
    switch ( v22 )
    {
      case 24:
      case 25:
      case 26:
      case 27:
      case 28:
      case 29:
      case 30:
      case 31:
      case 32:
      case 62:
      case 63:
      case 64:
      case 65:
      case 66:
      case 67:
        v22 = 3;
        break;
      case 33:
      case 34:
      case 35:
      case 36:
      case 37:
      case 38:
      case 39:
      case 40:
      case 68:
      case 69:
      case 70:
      case 71:
      case 72:
      case 73:
        v22 = 4;
        break;
      case 41:
      case 42:
      case 43:
      case 44:
      case 45:
      case 46:
      case 47:
      case 48:
      case 74:
      case 75:
      case 76:
      case 77:
      case 78:
      case 79:
        v22 = 5;
        break;
      case 49:
      case 50:
      case 51:
      case 52:
      case 53:
      case 54:
      case 80:
      case 81:
      case 82:
      case 83:
      case 84:
      case 85:
        v22 = 6;
        break;
      case 55:
        v22 = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        v22 = 8;
        break;
      case 89:
      case 90:
      case 91:
      case 92:
      case 93:
      case 101:
      case 102:
      case 103:
      case 104:
      case 105:
        v22 = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        v22 = 10;
        break;
      default:
        v22 = 2;
        break;
    }
  }
LABEL_3:
  v25 = sub_1F6C8D0(v22);
  v30 = 0;
  v121 = v25;
  v31 = *(_WORD *)(v21 + 24);
  if ( v31 != 118 )
    goto LABEL_4;
LABEL_32:
  if ( !v121 || (v121 & (v121 - 1LL)) != 0 )
    return 0;
  v53 = sub_1D1ADA0(
          *(_QWORD *)(*(_QWORD *)(v21 + 32) + 40LL),
          *(_QWORD *)(*(_QWORD *)(v21 + 32) + 48LL),
          _RDX,
          v27,
          v28,
          v29);
  if ( !v53 )
  {
    v31 = *(_WORD *)(v21 + 24);
    v30 = 0;
    goto LABEL_4;
  }
  v54 = *(__int64 **)(v21 + 32);
  v130 = 0;
  v131 = 1;
  v132 = 0;
  v133 = 1;
  v96 = v53;
  sub_1D1F820(v113, *v54, v54[1], &v130, 0);
  v55 = *(_QWORD *)(v96 + 88);
  _BitScanReverse64(&v56, v121);
  v57 = v56 ^ 0x3F;
  v30 = 63 - v57;
  if ( *(_DWORD *)(v55 + 32) <= 0x40u )
  {
    v58 = *(_QWORD *)(v55 + 24);
    if ( v58 )
    {
      _BitScanReverse64(&v59, v58);
      v27 = v59 ^ 0x3F;
      _RDX = (unsigned int)(64 - v27);
      if ( v30 < (unsigned int)_RDX )
        goto LABEL_38;
    }
    goto LABEL_86;
  }
  v91 = 63 - v57;
  v100 = *(_DWORD *)(v55 + 32);
  v68 = sub_16A57B0(v55 + 24);
  _RDX = v100;
  LODWORD(v28) = v68;
  v69 = (const void **)(v55 + 24);
  if ( v91 < v100 - v68 )
  {
LABEL_38:
    v30 = 0;
    goto LABEL_39;
  }
  v129 = v100;
  sub_16A4FD0((__int64)&v128, v69);
  v30 = v91;
  if ( v129 <= 0x40 )
  {
    v58 = (unsigned __int64)v128;
LABEL_86:
    v28 = v130 | v58;
    goto LABEL_87;
  }
  sub_16A89F0((__int64 *)&v128, (__int64 *)&v130);
  v70 = v129;
  v28 = (__int64)v128;
  v129 = 0;
  v30 = v91;
  v127 = v70;
  v126 = v128;
  if ( v70 <= 0x40 )
  {
LABEL_87:
    _RAX = ~v28;
    __asm { tzcnt   rdx, rax }
    v76 = 64;
    if ( v28 != -1 )
      v76 = _RDX;
    v72 = v30 <= v76;
    goto LABEL_78;
  }
  v101 = v128;
  v71 = sub_16A58F0((__int64)&v126);
  v30 = v91;
  LODWORD(v28) = (_DWORD)v101;
  v72 = v91 <= v71;
  if ( v101 )
  {
    v111 = v72;
    j_j___libc_free_0_0(v101);
    v72 = v111;
    v30 = v91;
    if ( v129 > 0x40 )
    {
      if ( v128 )
      {
        j_j___libc_free_0_0(v128);
        v30 = v91;
        v72 = v111;
      }
    }
  }
LABEL_78:
  if ( !v72 )
    goto LABEL_38;
  v21 = **(_QWORD **)(v21 + 32);
LABEL_39:
  if ( (unsigned int)v133 > 0x40 && v132 )
  {
    v106 = v30;
    j_j___libc_free_0_0(v132);
    v30 = v106;
  }
  if ( (unsigned int)v131 > 0x40 && v130 )
  {
    v107 = v30;
    j_j___libc_free_0_0(v130);
    v30 = v107;
  }
  v31 = *(_WORD *)(v21 + 24);
LABEL_4:
  if ( v31 != 53 )
    return 0;
  v103 = v30;
  v36 = sub_1D1ADA0(**(_QWORD **)(v21 + 32), *(_QWORD *)(*(_QWORD *)(v21 + 32) + 8LL), _RDX, v27, v28, v29);
  if ( !v36 )
    return 0;
  v37 = *(_QWORD *)(v21 + 32);
  v38 = v103;
  v39 = *(_QWORD *)(v37 + 40);
  v40 = *(_DWORD *)(v37 + 48);
  if ( v103 )
  {
    if ( *(_WORD *)(a6 + 24) == 118 )
    {
      v93 = v103;
      v97 = *(_QWORD *)(v37 + 40);
      v108 = v36;
      v63 = sub_1D1ADA0(
              *(_QWORD *)(*(_QWORD *)(a6 + 32) + 40LL),
              *(_QWORD *)(*(_QWORD *)(a6 + 32) + 48LL),
              _RDX,
              v34,
              v35,
              v39);
      v36 = v108;
      v39 = v97;
      v90 = v63;
      v38 = v93;
      if ( v63 )
      {
        v64 = *(__int64 **)(a6 + 32);
        v130 = 0;
        v131 = 1;
        v132 = 0;
        v133 = 1;
        sub_1D1F820(v113, *v64, v64[1], &v130, 0);
        v36 = v108;
        v39 = v97;
        v38 = v93;
        v65 = *(_QWORD *)(v90 + 88);
        if ( *(_DWORD *)(v65 + 32) > 0x40u )
        {
          v92 = *(_DWORD *)(v65 + 32);
          v119 = (const void **)(v65 + 24);
          v77 = sub_16A57B0(v65 + 24);
          _RDX = v92;
          v38 = v93;
          LODWORD(v35) = v77;
          v36 = v108;
          v39 = v97;
          if ( v93 < v92 - v77 )
          {
LABEL_56:
            if ( (unsigned int)v133 > 0x40 && v132 )
            {
              v98 = v38;
              v109 = v39;
              v117 = v36;
              j_j___libc_free_0_0(v132);
              v38 = v98;
              v39 = v109;
              v36 = v117;
            }
            if ( (unsigned int)v131 > 0x40 && v130 )
            {
              v99 = v38;
              v110 = v39;
              v118 = v36;
              j_j___libc_free_0_0(v130);
              v38 = v99;
              v39 = v110;
              v36 = v118;
            }
            goto LABEL_10;
          }
          v129 = v92;
          sub_16A4FD0((__int64)&v128, v119);
          v36 = v108;
          v39 = v97;
          v38 = v93;
          if ( v129 > 0x40 )
          {
            v102 = v93;
            v112 = v39;
            v120 = v36;
            sub_16A89F0((__int64 *)&v128, (__int64 *)&v130);
            v78 = v129;
            v35 = (__int64)v128;
            v129 = 0;
            v36 = v120;
            v39 = v112;
            v127 = v78;
            v38 = v93;
            v126 = v128;
            if ( v78 > 0x40 )
            {
              v94 = v128;
              v79 = sub_16A58F0((__int64)&v126);
              v38 = v102;
              LODWORD(v35) = (_DWORD)v94;
              v36 = v120;
              v39 = v112;
              v80 = v102 <= v79;
              if ( v94 )
              {
                v81 = v94;
                v95 = v80;
                j_j___libc_free_0_0(v81);
                v36 = v120;
                v39 = v112;
                v38 = v102;
                v80 = v95;
                if ( v129 > 0x40 )
                {
                  if ( v128 )
                  {
                    j_j___libc_free_0_0(v128);
                    v80 = v95;
                    v38 = v102;
                    v39 = v112;
                    v36 = v120;
                  }
                }
              }
              goto LABEL_97;
            }
LABEL_120:
            _RAX = ~v35;
            __asm { tzcnt   rdx, rax }
            v89 = 64;
            _RDX = (int)_RDX;
            if ( v35 != -1 )
              v89 = _RDX;
            v80 = v38 <= v89;
LABEL_97:
            if ( v80 )
            {
              v82 = *(__int64 **)(a6 + 32);
              a6 = *v82;
              a12 = *((_DWORD *)v82 + 2);
            }
            goto LABEL_56;
          }
          v66 = (unsigned __int64)v128;
        }
        else
        {
          v66 = *(_QWORD *)(v65 + 24);
          if ( v66 )
          {
            _BitScanReverse64(&v67, v66);
            v34 = v67 ^ 0x3F;
            _RDX = (unsigned int)(64 - v34);
            if ( v93 < (unsigned int)_RDX )
              goto LABEL_56;
          }
        }
        v35 = v130 | v66;
        goto LABEL_120;
      }
    }
  }
LABEL_10:
  LODWORD(v131) = 1;
  v130 = 0;
  if ( v40 != a12 || v39 != a6 )
  {
    if ( *(_WORD *)(a6 + 24) != 52 )
      return 0;
    v41 = *(_QWORD *)(a6 + 32);
    if ( v39 != *(_QWORD *)v41 || v40 != *(_DWORD *)(v41 + 8) )
      return 0;
    v104 = v38;
    v114 = v36;
    v42 = sub_1D1ADA0(*(_QWORD *)(v41 + 40), *(_QWORD *)(v41 + 48), _RDX, v34, v35, v39);
    if ( !v42 )
      goto LABEL_116;
    v43 = *(_QWORD *)(v42 + 88);
    v44 = v104;
    v45 = (__int64 *)(*(_QWORD *)(v114 + 88) + 24LL);
    v129 = *(_DWORD *)(v43 + 32);
    if ( v129 > 0x40 )
    {
      sub_16A4FD0((__int64)&v128, (const void **)(v43 + 24));
      v44 = v104;
    }
    else
    {
      v128 = *(_QWORD **)(v43 + 24);
    }
    v115 = v44;
    sub_16A7200((__int64)&v128, v45);
    v46 = v129;
    v129 = 0;
    v47 = v128;
    v38 = v115;
    if ( (unsigned int)v131 > 0x40 && v130 )
    {
      j_j___libc_free_0_0(v130);
      v130 = (unsigned __int64)v47;
      LODWORD(v131) = v46;
      v38 = v115;
      if ( v129 > 0x40 && v128 )
      {
        j_j___libc_free_0_0(v128);
        v38 = v115;
      }
    }
    else
    {
      v130 = (unsigned __int64)v128;
      LODWORD(v131) = v46;
    }
LABEL_23:
    if ( !v38 )
    {
      v48 = v131;
      v49 = v121;
      if ( (unsigned int)v131 > 0x40 )
      {
        v50 = sub_16A57B0((__int64)&v130);
        v51 = v130;
        if ( (unsigned int)(v48 - v50) > 0x40 || v121 != *(_QWORD *)v130 )
          goto LABEL_27;
        goto LABEL_128;
      }
      goto LABEL_100;
    }
    goto LABEL_81;
  }
  v61 = *(_QWORD *)(v36 + 88);
  v62 = *(_DWORD *)(v61 + 32);
  if ( v62 > 0x40 )
  {
    v116 = v38;
    sub_16A51C0((__int64)&v130, v61 + 24);
    v38 = v116;
    goto LABEL_23;
  }
  v73 = *(_QWORD *)(v61 + 24);
  LODWORD(v131) = *(_DWORD *)(v61 + 32);
  v49 = v121;
  v130 = v73 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v62);
  if ( !v38 )
  {
LABEL_100:
    if ( v49 == v130 )
      goto LABEL_101;
LABEL_116:
    if ( (unsigned int)v131 <= 0x40 )
      return 0;
LABEL_84:
    v51 = v130;
LABEL_27:
    if ( v51 )
      j_j___libc_free_0_0(v51);
    return 0;
  }
LABEL_81:
  sub_16A88B0((__int64)&v128, (__int64)&v130, v38);
  v74 = v129;
  if ( v129 <= 0x40 )
  {
    if ( v128 )
    {
      if ( (unsigned int)v131 > 0x40 )
        goto LABEL_84;
      return 0;
    }
    if ( (unsigned int)v131 <= 0x40 )
      goto LABEL_101;
LABEL_127:
    v51 = v130;
LABEL_128:
    if ( v51 )
      j_j___libc_free_0_0(v51);
LABEL_101:
    v83 = a1[1];
    v84 = 1;
    if ( (_BYTE)v124 == 1 || (_BYTE)v124 && (v84 = (unsigned __int8)v124, v83[(unsigned __int8)v124 + 15]) )
    {
      v85 = *a1;
      v86 = (__int128 *)v123;
      if ( (*((_BYTE *)v83 + 259 * v84 + a14 + 2422) & 0xFB) == 0 )
        return sub_1D332F0(v85, a14, a16, v124, v125, 0, a7, a8, a9, a2, a3, *v86);
    }
    else
    {
      v85 = *a1;
    }
    a14 = a15;
    v86 = (__int128 *)&a10;
    return sub_1D332F0(v85, a14, a16, v124, v125, 0, a7, a8, a9, a2, a3, *v86);
  }
  v87 = sub_16A57B0((__int64)&v128);
  result = v128;
  if ( v74 - v87 <= 0x40 )
  {
    if ( !*v128 )
    {
      j_j___libc_free_0_0(v128);
      if ( (unsigned int)v131 <= 0x40 )
        goto LABEL_101;
      goto LABEL_127;
    }
LABEL_114:
    j_j___libc_free_0_0(v128);
    if ( (unsigned int)v131 > 0x40 )
      goto LABEL_84;
    return 0;
  }
  if ( v128 )
    goto LABEL_114;
  if ( (unsigned int)v131 > 0x40 )
  {
    v122 = v128;
    if ( v130 )
    {
      j_j___libc_free_0_0(v130);
      return v122;
    }
    return 0;
  }
  return result;
}
