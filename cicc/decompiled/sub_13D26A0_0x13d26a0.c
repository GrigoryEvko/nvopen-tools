// Function: sub_13D26A0
// Address: 0x13d26a0
//
char __fastcall sub_13D26A0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v3; // r15d
  __int64 v4; // rax
  _BYTE *v7; // rsi
  _BYTE *v8; // rsi
  __int64 *v9; // rsi
  unsigned int v10; // r15d
  bool v11; // al
  unsigned int v12; // eax
  unsigned int v13; // eax
  _BYTE *v14; // rsi
  __int64 *v15; // rsi
  _BYTE *v16; // rsi
  _BYTE *v17; // rsi
  __int64 *v18; // r8
  unsigned int v19; // r12d
  int v20; // eax
  bool v21; // al
  _BYTE *v22; // rsi
  _BYTE *v23; // rsi
  __int64 v24; // rdx
  __int64 v25; // rcx
  unsigned int v26; // r8d
  __int64 *v27; // r15
  _BYTE *v28; // rsi
  __int64 *v29; // rsi
  _BYTE *v30; // rsi
  unsigned int v31; // ecx
  unsigned __int64 v32; // rdx
  unsigned int v33; // eax
  unsigned int v34; // eax
  _BYTE *v35; // rsi
  unsigned int v36; // r15d
  unsigned int v37; // edx
  unsigned __int64 v38; // r15
  unsigned __int64 v39; // rax
  _BYTE *v40; // rsi
  __int64 *v41; // r9
  unsigned __int64 v42; // rsi
  unsigned __int64 v43; // rax
  __int64 v44; // r12
  __int64 *v45; // r12
  unsigned int v46; // eax
  __int64 *v47; // rdi
  _BYTE *v48; // rsi
  _BYTE *v49; // rsi
  _BYTE *v50; // rsi
  unsigned int v51; // eax
  __int64 *v52; // r15
  unsigned int v53; // esi
  bool v54; // al
  __int64 v55; // rsi
  __int64 v56; // rax
  __int64 v57; // rsi
  __int64 *v58; // r8
  unsigned __int64 v59; // rsi
  unsigned __int64 v60; // rax
  unsigned int v61; // eax
  unsigned int v62; // ebx
  __int64 v63; // rax
  __int64 *v64; // r8
  unsigned int v65; // edx
  unsigned int v66; // eax
  int v67; // eax
  int v68; // eax
  __int64 v69; // rsi
  __int64 v70; // rax
  __int64 v71; // rsi
  unsigned int v72; // eax
  unsigned int v73; // eax
  unsigned int v74; // eax
  unsigned int v75; // eax
  unsigned __int64 *v76; // r15
  unsigned int v77; // ebx
  int v78; // esi
  unsigned __int64 v79; // rsi
  int v80; // ebx
  char v81; // al
  unsigned int v84; // eax
  char v85; // al
  unsigned __int64 v88; // rax
  unsigned int v89; // eax
  unsigned int v90; // eax
  __int64 *v91; // rbx
  __int64 *v92; // rbx
  unsigned int v93; // eax
  unsigned int v94; // eax
  __int64 v95; // rdx
  unsigned __int64 v96; // rsi
  unsigned int v97; // eax
  unsigned int v98; // eax
  int v99; // edx
  unsigned __int64 v100; // rax
  int v101; // ebx
  int v102; // ecx
  __int64 v103; // rdx
  unsigned int v104; // eax
  unsigned int v105; // eax
  int v107; // [rsp+8h] [rbp-98h]
  int v108; // [rsp+10h] [rbp-90h]
  __int64 *v109; // [rsp+10h] [rbp-90h]
  __int64 *v110; // [rsp+10h] [rbp-90h]
  __int64 *v111; // [rsp+10h] [rbp-90h]
  __int64 *v112; // [rsp+10h] [rbp-90h]
  __int64 v113; // [rsp+10h] [rbp-90h]
  __int64 *v114; // [rsp+10h] [rbp-90h]
  int v115; // [rsp+18h] [rbp-88h]
  __int64 *v116; // [rsp+18h] [rbp-88h]
  unsigned int v117; // [rsp+18h] [rbp-88h]
  __int64 v118; // [rsp+18h] [rbp-88h]
  unsigned int v119; // [rsp+18h] [rbp-88h]
  unsigned int v120; // [rsp+18h] [rbp-88h]
  __int64 *v121; // [rsp+18h] [rbp-88h]
  __int64 *v122; // [rsp+28h] [rbp-78h] BYREF
  __int64 v123; // [rsp+30h] [rbp-70h] BYREF
  unsigned int v124; // [rsp+38h] [rbp-68h]
  unsigned __int64 v125; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v126; // [rsp+48h] [rbp-58h]
  unsigned __int64 v127; // [rsp+50h] [rbp-50h] BYREF
  unsigned int v128; // [rsp+58h] [rbp-48h]
  unsigned __int64 v129; // [rsp+60h] [rbp-40h] BYREF
  unsigned int v130; // [rsp+68h] [rbp-38h]

  v3 = *(_DWORD *)(a2 + 8);
  LOBYTE(v4) = *(_BYTE *)(a1 + 16) - 35;
  switch ( *(_BYTE *)(a1 + 16) )
  {
    case '#':
      v16 = *(_BYTE **)(a1 - 24);
      v129 = (unsigned __int64)&v122;
      LOBYTE(v4) = sub_13D2630((_QWORD **)&v129, v16);
      if ( (_BYTE)v4 )
      {
        if ( *((_DWORD *)v122 + 2) <= 0x40u )
        {
          LOBYTE(v4) = *v122 == 0;
        }
        else
        {
          v115 = *((_DWORD *)v122 + 2);
          LOBYTE(v4) = v115 == (unsigned int)sub_16A57B0(v122);
        }
        if ( !(_BYTE)v4 )
        {
          if ( (unsigned __int8)sub_15F2370(a1) )
          {
            LOBYTE(v4) = sub_13CC340(a2, v122);
          }
          else
          {
            LOBYTE(v4) = sub_15F2380(a1);
            if ( (_BYTE)v4 )
            {
              v91 = v122;
              if ( sub_13D0200(v122, *((_DWORD *)v122 + 2) - 1) )
              {
                sub_13D00B0((__int64)&v129, v3);
                sub_13CBE00((__int64 *)a2, (__int64 *)&v129);
                sub_135E100((__int64 *)&v129);
                v92 = v122;
                sub_13D0020((__int64)&v125, v3);
                sub_16A7200(&v125, v92);
                v93 = v126;
                v126 = 0;
                v128 = v93;
                v127 = v125;
                sub_16A7490(&v127, 1);
                v94 = v128;
                v128 = 0;
                v130 = v94;
                v129 = v127;
                sub_13CBE00((__int64 *)a3, (__int64 *)&v129);
                sub_135E100((__int64 *)&v129);
                sub_135E100((__int64 *)&v127);
                LOBYTE(v4) = sub_135E100((__int64 *)&v125);
              }
              else
              {
                sub_13D00B0((__int64)&v127, v3);
                sub_16A7200(&v127, v91);
                v104 = v128;
                v128 = 0;
                v130 = v104;
                v129 = v127;
                sub_13CBE00((__int64 *)a2, (__int64 *)&v129);
                sub_135E100((__int64 *)&v129);
                sub_135E100((__int64 *)&v127);
                sub_13D0020((__int64)&v127, v3);
                sub_16A7490(&v127, 1);
                v105 = v128;
                v128 = 0;
                v130 = v105;
                v129 = v127;
                sub_13CBE00((__int64 *)a3, (__int64 *)&v129);
                sub_135E100((__int64 *)&v129);
                LOBYTE(v4) = sub_135E100((__int64 *)&v127);
              }
            }
          }
        }
      }
      return v4;
    case ')':
      v17 = *(_BYTE **)(a1 - 24);
      v129 = (unsigned __int64)&v122;
      if ( (unsigned __int8)sub_13D2630((_QWORD **)&v129, v17) )
      {
        v18 = v122;
        v19 = *((_DWORD *)v122 + 2);
        if ( v19 <= 0x40 )
        {
          v21 = *v122 == 0;
        }
        else
        {
          v116 = v122;
          v20 = sub_16A57B0(v122);
          v18 = v116;
          v21 = v19 == v20;
        }
        if ( !v21 )
        {
          v126 = v3;
          if ( v3 > 0x40 )
          {
            sub_16A4EF0(&v125, -1, 1);
            v18 = v122;
          }
          else
          {
            v125 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v3;
          }
          sub_16A9D70(&v127, &v125, v18);
          goto LABEL_85;
        }
      }
      v49 = *(_BYTE **)(a1 - 48);
      v129 = (unsigned __int64)&v122;
      LOBYTE(v4) = sub_13D2630((_QWORD **)&v129, v49);
      if ( (_BYTE)v4 )
        goto LABEL_12;
      return v4;
    case '*':
      v22 = *(_BYTE **)(a1 - 24);
      v129 = (unsigned __int64)&v122;
      if ( !(unsigned __int8)sub_13D2630((_QWORD **)&v129, v22) )
      {
        v23 = *(_BYTE **)(a1 - 48);
        v129 = (unsigned __int64)&v122;
        LOBYTE(v4) = sub_13D2630((_QWORD **)&v129, v23);
        if ( (_BYTE)v4 )
        {
          v27 = v122;
          if ( (unsigned __int8)sub_13CFF40(v122, (__int64)v23, v24, v25, v26) )
          {
            sub_13CC340(a2, v27);
            sub_13A38D0((__int64)&v127, a2);
            if ( v128 > 0x40 )
            {
              sub_16A8110(&v127, 1);
            }
            else if ( v128 == 1 )
            {
              v127 = 0;
            }
            else
            {
              v127 >>= 1;
            }
            goto LABEL_14;
          }
          sub_13A3E40((__int64)&v127, (__int64)v27);
          sub_16A7490(&v127, 1);
          v73 = v128;
          v128 = 0;
          v130 = v73;
          v129 = v127;
          sub_13CBE00((__int64 *)a3, (__int64 *)&v129);
          sub_135E100((__int64 *)&v129);
          sub_135E100((__int64 *)&v127);
          sub_13A38D0((__int64)&v125, a3);
          if ( v126 > 0x40 )
            sub_16A8F40(&v125);
          else
            v125 = ~v125 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v126);
          sub_16A7400(&v125);
          v74 = v126;
          v126 = 0;
          v128 = v74;
          v127 = v125;
          sub_16A7490(&v127, 1);
          v75 = v128;
          v47 = (__int64 *)a2;
          v128 = 0;
          v130 = v75;
          v129 = v127;
          goto LABEL_65;
        }
        return v4;
      }
      v62 = v3 - 1;
      v124 = v3;
      if ( v3 > 0x40 )
      {
        v113 = 1LL << ((unsigned __int8)v3 - 1);
        sub_16A4EF0(&v123, 0, 0);
        if ( v124 <= 0x40 )
          v123 |= v113;
        else
          *(_QWORD *)(v123 + 8LL * (v62 >> 6)) |= v113;
        v126 = v3;
        sub_16A4EF0(&v125, -1, 1);
        v63 = ~(1LL << ((unsigned __int8)v3 - 1));
        if ( v126 > 0x40 )
        {
          *(_QWORD *)(v125 + 8LL * (v62 >> 6)) &= ~(1LL << ((unsigned __int8)v3 - 1));
LABEL_90:
          v64 = v122;
          v65 = *((_DWORD *)v122 + 2);
          if ( v65 <= 0x40 )
          {
            v88 = *v122;
            if ( *v122 != 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v65) )
            {
              if ( v88 )
              {
                _BitScanReverse64(&v88, v88);
                v65 = v65 - 64 + (v88 ^ 0x3F);
              }
              goto LABEL_93;
            }
          }
          else
          {
            v107 = *((_DWORD *)v122 + 2);
            v110 = v122;
            if ( v107 != (unsigned int)sub_16A58F0(v122) )
            {
              v66 = sub_16A57B0(v110);
              v64 = v110;
              v65 = v66;
LABEL_93:
              if ( v62 > v65 )
              {
                sub_16A9F90(&v129, &v123, v64);
                sub_13CBE00((__int64 *)a2, (__int64 *)&v129);
                sub_135E100((__int64 *)&v129);
                sub_16A9F90(&v129, &v125, v122);
                sub_13CBE00((__int64 *)a3, (__int64 *)&v129);
                sub_135E100((__int64 *)&v129);
                if ( (int)sub_16AEA10(a2, a3) > 0 )
                {
                  v97 = *(_DWORD *)(a2 + 8);
                  *(_DWORD *)(a2 + 8) = 0;
                  v130 = v97;
                  v129 = *(_QWORD *)a2;
                  *(_QWORD *)a2 = *(_QWORD *)a3;
                  *(_DWORD *)(a2 + 8) = *(_DWORD *)(a3 + 8);
                  *(_DWORD *)(a3 + 8) = 0;
                  sub_13CBE00((__int64 *)a3, (__int64 *)&v129);
                  sub_135E100((__int64 *)&v129);
                }
                sub_13A38D0((__int64)&v127, a3);
                sub_16A7490(&v127, 1);
                v98 = v128;
                v128 = 0;
                v130 = v98;
                v129 = v127;
                sub_13CBE00((__int64 *)a3, (__int64 *)&v129);
                sub_135E100((__int64 *)&v129);
                sub_135E100((__int64 *)&v127);
              }
              goto LABEL_94;
            }
          }
          sub_13A38D0((__int64)&v127, (__int64)&v123);
          sub_16A7490(&v127, 1);
          v89 = v128;
          v128 = 0;
          v130 = v89;
          v129 = v127;
          sub_13CBE00((__int64 *)a2, (__int64 *)&v129);
          sub_135E100((__int64 *)&v129);
          sub_135E100((__int64 *)&v127);
          sub_13A38D0((__int64)&v127, (__int64)&v125);
          sub_16A7490(&v127, 1);
          v90 = v128;
          v128 = 0;
          v130 = v90;
          v129 = v127;
          sub_13CBE00((__int64 *)a3, (__int64 *)&v129);
          sub_135E100((__int64 *)&v129);
          sub_135E100((__int64 *)&v127);
LABEL_94:
          sub_135E100((__int64 *)&v125);
          LOBYTE(v4) = sub_135E100(&v123);
          return v4;
        }
      }
      else
      {
        v123 = 1LL << ((unsigned __int8)v3 - 1);
        v63 = ~v123;
        v126 = v3;
        v125 = 0xFFFFFFFFFFFFFFFFLL >> (63 - ((v3 - 1) & 0x3F));
      }
      v125 &= v63;
      goto LABEL_90;
    case ',':
      v28 = *(_BYTE **)(a1 - 24);
      v129 = (unsigned __int64)&v122;
      LOBYTE(v4) = sub_13D2630((_QWORD **)&v129, v28);
      if ( (_BYTE)v4 )
      {
        v29 = v122;
        if ( *(_DWORD *)(a3 + 8) <= 0x40u && *((_DWORD *)v122 + 2) <= 0x40u )
        {
          v95 = *v122;
          *(_QWORD *)a3 = *v122;
          v4 = *((unsigned int *)v29 + 2);
          *(_DWORD *)(a3 + 8) = v4;
          v96 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v4;
          if ( (unsigned int)v4 > 0x40 )
            goto LABEL_162;
          *(_QWORD *)a3 = v96 & v95;
        }
        else
        {
          LOBYTE(v4) = sub_16A51C0(a3, v122);
        }
      }
      return v4;
    case '-':
      v30 = *(_BYTE **)(a1 - 24);
      v129 = (unsigned __int64)&v122;
      LOBYTE(v4) = sub_13D2630((_QWORD **)&v129, v30);
      if ( !(_BYTE)v4 )
        return v4;
      sub_13A3E40((__int64)&v129, (__int64)v122);
      sub_13CBE00((__int64 *)a3, (__int64 *)&v129);
      sub_135E100((__int64 *)&v129);
      v31 = *(_DWORD *)(a3 + 8);
      v126 = v31;
      if ( v31 <= 0x40 )
      {
        v32 = *(_QWORD *)a3;
LABEL_46:
        v125 = ~v32 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v31);
        goto LABEL_47;
      }
      sub_16A4FD0(&v125, a3);
      LOBYTE(v31) = v126;
      if ( v126 <= 0x40 )
      {
        v32 = v125;
        goto LABEL_46;
      }
      sub_16A8F40(&v125);
LABEL_47:
      sub_16A7400(&v125);
      v33 = v126;
      v126 = 0;
      v128 = v33;
      v127 = v125;
      sub_16A7490(&v127, 1);
      v34 = v128;
      v128 = 0;
      v130 = v34;
      v129 = v127;
      sub_13CBE00((__int64 *)a2, (__int64 *)&v129);
      sub_135E100((__int64 *)&v129);
      sub_135E100((__int64 *)&v127);
      LOBYTE(v4) = sub_135E100((__int64 *)&v125);
      return v4;
    case '/':
      v35 = *(_BYTE **)(a1 - 48);
      v129 = (unsigned __int64)&v122;
      LOBYTE(v4) = sub_13D2630((_QWORD **)&v129, v35);
      if ( (_BYTE)v4 )
      {
        if ( (unsigned __int8)sub_15F2370(a1) )
        {
          sub_13CC340(a2, v122);
          v36 = *(_DWORD *)(a2 + 8);
          if ( v36 > 0x40 )
          {
            v36 = sub_16A57B0(a2);
          }
          else
          {
            v37 = v36 - 64;
            if ( *(_QWORD *)a2 )
            {
              _BitScanReverse64(&v38, *(_QWORD *)a2);
              v36 = v37 + (v38 ^ 0x3F);
            }
          }
          sub_13A38D0((__int64)&v127, a2);
          if ( v128 > 0x40 )
          {
            sub_16A7DC0(&v127, v36);
          }
          else
          {
            v39 = 0;
            if ( v128 != v36 )
              v39 = (v127 << v36) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v128);
            v127 = v39;
          }
          goto LABEL_14;
        }
        LOBYTE(v4) = sub_15F2380(a1);
        if ( (_BYTE)v4 )
        {
          v76 = (unsigned __int64 *)v122;
          v77 = *((_DWORD *)v122 + 2);
          if ( sub_13D0200(v122, v77 - 1) )
          {
            if ( v77 > 0x40 )
            {
              v80 = sub_16A5810(v76);
            }
            else
            {
              v78 = 64;
              if ( *v76 << (64 - (unsigned __int8)v77) != -1 )
              {
                _BitScanReverse64(&v79, ~(*v76 << (64 - (unsigned __int8)v77)));
                v78 = v79 ^ 0x3F;
              }
              v80 = v78;
            }
            sub_13A38D0((__int64)&v129, (__int64)v76);
            sub_13CC1B0((__int64)&v129, (unsigned int)(v80 - 1));
            goto LABEL_103;
          }
          if ( v77 > 0x40 )
          {
            v101 = sub_16A57B0(v76);
          }
          else
          {
            v99 = 64;
            if ( *v76 )
            {
              _BitScanReverse64(&v100, *v76);
              v99 = v100 ^ 0x3F;
            }
            v101 = v77 + v99 - 64;
          }
          sub_13CC340(a2, (__int64 *)v76);
          sub_13A38D0((__int64)&v127, (__int64)v122);
          v102 = v101 - 1;
          if ( v128 > 0x40 )
          {
            sub_16A7DC0(&v127, (unsigned int)(v101 - 1));
          }
          else
          {
            v103 = 0;
            if ( v102 != v128 )
              v103 = v127 << v102;
            v127 = v103 & (0xFFFFFFFFFFFFFFFFLL >> -(char)v128);
          }
          goto LABEL_14;
        }
      }
      return v4;
    case '0':
      v7 = *(_BYTE **)(a1 - 24);
      v129 = (unsigned __int64)&v122;
      if ( !(unsigned __int8)sub_13D2630((_QWORD **)&v129, v7) )
        goto LABEL_3;
      v58 = v122;
      v59 = v3;
      v120 = *((_DWORD *)v122 + 2);
      if ( v120 > 0x40 )
      {
        v111 = v122;
        v67 = sub_16A57B0(v122);
        v58 = v111;
        v59 = v3;
        if ( v120 - v67 > 0x40 )
          goto LABEL_3;
        v60 = *(_QWORD *)*v111;
      }
      else
      {
        v60 = *v122;
      }
      if ( v59 > v60 )
      {
        v126 = v3;
        if ( v3 > 0x40 )
        {
          sub_16A4EF0(&v125, -1, 1);
          v58 = v122;
        }
        else
        {
          v125 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v3;
        }
        v121 = v58;
        sub_13A38D0((__int64)&v127, (__int64)&v125);
        sub_16A81B0(&v127, v121);
LABEL_85:
        sub_16A7490(&v127, 1);
        v61 = v128;
        v128 = 0;
        v130 = v61;
        v129 = v127;
        sub_13CBE00((__int64 *)a3, (__int64 *)&v129);
        sub_135E100((__int64 *)&v129);
        sub_135E100((__int64 *)&v127);
        LOBYTE(v4) = sub_135E100((__int64 *)&v125);
        return v4;
      }
LABEL_3:
      v8 = *(_BYTE **)(a1 - 48);
      v129 = (unsigned __int64)&v122;
      LOBYTE(v4) = sub_13D2630((_QWORD **)&v129, v8);
      if ( !(_BYTE)v4 )
        return v4;
      v9 = v122;
      v10 = v3 - 1;
      if ( *((_DWORD *)v122 + 2) <= 0x40u )
      {
        v11 = *v122 == 0;
      }
      else
      {
        v108 = *((_DWORD *)v122 + 2);
        v9 = v122;
        v11 = v108 == (unsigned int)sub_16A57B0(v122);
      }
      if ( !v11 )
      {
        v85 = sub_15F23D0(a1);
        v9 = v122;
        if ( v85 )
        {
          v10 = *((_DWORD *)v122 + 2);
          if ( v10 > 0x40 )
          {
            v9 = v122;
            v10 = sub_16A58A0(v122);
          }
          else
          {
            _RDX = *v122;
            __asm { tzcnt   rax, rdx }
            if ( !*v122 )
              LODWORD(_RAX) = 64;
            if ( v10 >= (unsigned int)_RAX )
              v10 = _RAX;
          }
        }
      }
      v12 = *((_DWORD *)v9 + 2);
      v130 = v12;
      if ( v12 > 0x40 )
      {
        sub_16A4FD0(&v129, v9);
        v12 = v130;
        if ( v130 > 0x40 )
        {
          sub_16A8110(&v129, v10);
LABEL_11:
          sub_13CBE00((__int64 *)a2, (__int64 *)&v129);
          sub_135E100((__int64 *)&v129);
          goto LABEL_12;
        }
      }
      else
      {
        v129 = *v9;
      }
      if ( v10 == v12 )
        v129 = 0;
      else
        v129 >>= v10;
      goto LABEL_11;
    case '1':
      v40 = *(_BYTE **)(a1 - 24);
      v129 = (unsigned __int64)&v122;
      if ( !(unsigned __int8)sub_13D2630((_QWORD **)&v129, v40) )
        goto LABEL_70;
      v41 = v122;
      v42 = v3;
      v117 = *((_DWORD *)v122 + 2);
      if ( v117 > 0x40 )
      {
        v112 = v122;
        v68 = sub_16A57B0(v122);
        v41 = v112;
        v42 = v3;
        if ( v117 - v68 > 0x40 )
          goto LABEL_70;
        v43 = *(_QWORD *)*v112;
      }
      else
      {
        v43 = *v122;
      }
      if ( v42 > v43 )
      {
        v128 = v3;
        v118 = 1LL << ((unsigned __int8)v3 - 1);
        if ( v3 > 0x40 )
        {
          sub_16A4EF0(&v127, 0, 0);
          if ( v128 <= 0x40 )
            v127 |= v118;
          else
            *(_QWORD *)(v127 + 8LL * ((v3 - 1) >> 6)) |= v118;
          v114 = v122;
          sub_13A38D0((__int64)&v129, (__int64)&v127);
          sub_16A6020(&v129, v114);
          sub_13CBE00((__int64 *)a2, (__int64 *)&v129);
          sub_135E100((__int64 *)&v129);
          sub_135E100((__int64 *)&v127);
          v126 = v3;
          v44 = ~v118;
          sub_16A4EF0(&v125, -1, 1);
          if ( v126 > 0x40 )
          {
            *(_QWORD *)(v125 + 8LL * ((v3 - 1) >> 6)) &= v44;
LABEL_64:
            v45 = v122;
            sub_13A38D0((__int64)&v127, (__int64)&v125);
            sub_16A6020(&v127, v45);
            sub_16A7490(&v127, 1);
            v46 = v128;
            v47 = (__int64 *)a3;
            v128 = 0;
            v130 = v46;
            v129 = v127;
LABEL_65:
            sub_13CBE00(v47, (__int64 *)&v129);
            sub_135E100((__int64 *)&v129);
            sub_135E100((__int64 *)&v127);
            LOBYTE(v4) = sub_135E100((__int64 *)&v125);
            return v4;
          }
        }
        else
        {
          v109 = v41;
          v127 = 1LL << ((unsigned __int8)v3 - 1);
          sub_13A38D0((__int64)&v129, (__int64)&v127);
          sub_16A6020(&v129, v109);
          sub_13CBE00((__int64 *)a2, (__int64 *)&v129);
          sub_135E100((__int64 *)&v129);
          sub_135E100((__int64 *)&v127);
          v126 = v3;
          v44 = ~v118;
          v125 = 0xFFFFFFFFFFFFFFFFLL >> (63 - ((v3 - 1) & 0x3F));
        }
        v125 &= v44;
        goto LABEL_64;
      }
LABEL_70:
      v50 = *(_BYTE **)(a1 - 48);
      v129 = (unsigned __int64)&v122;
      LOBYTE(v4) = sub_13D2630((_QWORD **)&v129, v50);
      if ( (_BYTE)v4 )
      {
        v51 = v3 - 1;
        v52 = v122;
        v119 = v51;
        v53 = *((_DWORD *)v122 + 2);
        if ( v53 <= 0x40 )
        {
          v54 = *v122 == 0;
        }
        else
        {
          v53 = *((_DWORD *)v122 + 2);
          v54 = v53 == (unsigned int)sub_16A57B0(v122);
        }
        if ( !v54 )
        {
          v81 = sub_15F23D0(a1);
          v52 = v122;
          v53 = *((_DWORD *)v122 + 2);
          if ( v81 )
          {
            if ( v53 > 0x40 )
            {
              v53 = *((_DWORD *)v122 + 2);
              v119 = sub_16A58A0(v122);
            }
            else
            {
              _RAX = *v122;
              __asm { tzcnt   rdx, rax }
              v84 = 64;
              if ( *v122 )
                v84 = _RDX;
              if ( v53 < v84 )
                v84 = *((_DWORD *)v122 + 2);
              v119 = v84;
            }
          }
        }
        if ( sub_13D0200(v52, v53 - 1) )
        {
          sub_13CC340(a2, v52);
          sub_13A38D0((__int64)&v127, (__int64)v122);
          if ( v128 > 0x40 )
          {
            sub_16A5E70(&v127, v119);
          }
          else
          {
            v55 = (__int64)(v127 << (64 - (unsigned __int8)v128)) >> (64 - (unsigned __int8)v128);
            v56 = v55 >> v119;
            v57 = v55 >> 63;
            if ( v119 == v128 )
              v56 = v57;
            v127 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v128) & v56;
          }
LABEL_14:
          sub_16A7490(&v127, 1);
          v13 = v128;
          v128 = 0;
          v130 = v13;
          v129 = v127;
          sub_13CBE00((__int64 *)a3, (__int64 *)&v129);
          sub_135E100((__int64 *)&v129);
          LOBYTE(v4) = sub_135E100((__int64 *)&v127);
        }
        else
        {
          sub_13A38D0((__int64)&v129, (__int64)v52);
          if ( v130 > 0x40 )
          {
            sub_16A5E70(&v129, v119);
          }
          else
          {
            v69 = (__int64)(v129 << (64 - (unsigned __int8)v130)) >> (64 - (unsigned __int8)v130);
            v70 = v69 >> v119;
            v71 = v69 >> 63;
            if ( v119 == v130 )
              v70 = v71;
            v129 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v130) & v70;
          }
LABEL_103:
          sub_13CBE00((__int64 *)a2, (__int64 *)&v129);
          sub_135E100((__int64 *)&v129);
          sub_13A38D0((__int64)&v127, (__int64)v122);
          sub_16A7490(&v127, 1);
          v72 = v128;
          v128 = 0;
          v130 = v72;
          v129 = v127;
          sub_13CBE00((__int64 *)a3, (__int64 *)&v129);
          sub_135E100((__int64 *)&v129);
          LOBYTE(v4) = sub_135E100((__int64 *)&v127);
        }
      }
      return v4;
    case '2':
      v48 = *(_BYTE **)(a1 - 24);
      v129 = (unsigned __int64)&v122;
      LOBYTE(v4) = sub_13D2630((_QWORD **)&v129, v48);
      if ( !(_BYTE)v4 )
        return v4;
LABEL_12:
      v128 = *((_DWORD *)v122 + 2);
      if ( v128 > 0x40 )
        sub_16A4FD0(&v127, v122);
      else
        v127 = *v122;
      goto LABEL_14;
    case '3':
      v14 = *(_BYTE **)(a1 - 24);
      v129 = (unsigned __int64)&v122;
      LOBYTE(v4) = sub_13D2630((_QWORD **)&v129, v14);
      if ( (_BYTE)v4 )
      {
        v15 = v122;
        if ( *(_DWORD *)(a2 + 8) <= 0x40u && *((_DWORD *)v122 + 2) <= 0x40u )
        {
          v95 = *v122;
          *(_QWORD *)a2 = *v122;
          v4 = *((unsigned int *)v15 + 2);
          *(_DWORD *)(a2 + 8) = v4;
          v96 = 0xFFFFFFFFFFFFFFFFLL >> -(char)v4;
          if ( (unsigned int)v4 > 0x40 )
          {
LABEL_162:
            v4 = (unsigned int)((unsigned __int64)(v4 + 63) >> 6) - 1;
            *(_QWORD *)(v95 + 8 * v4) &= v96;
          }
          else
          {
            *(_QWORD *)a2 = v96 & v95;
          }
        }
        else
        {
          LOBYTE(v4) = sub_16A51C0(a2, v122);
        }
      }
      return v4;
    default:
      return v4;
  }
}
