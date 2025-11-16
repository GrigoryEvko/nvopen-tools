// Function: sub_27CF680
// Address: 0x27cf680
//
__int64 __fastcall sub_27CF680(
        __int64 a1,
        unsigned __int64 a2,
        unsigned int a3,
        __int64 a4,
        unsigned __int64 a5,
        __int64 a6)
{
  __int64 v7; // r12
  __int64 v8; // rbx
  __int64 v9; // rdi
  char v10; // dh
  __int64 v11; // r14
  char v12; // dl
  __int64 *v13; // rax
  _QWORD *v14; // rax
  __int64 v15; // r15
  __int64 v16; // rdx
  __int64 *v21; // rax
  __int64 v22; // r9
  __int64 v23; // r10
  int v24; // ecx
  __int64 v25; // rax
  char v26; // al
  int v27; // eax
  unsigned __int64 v28; // r8
  __int64 v29; // r10
  __int64 v30; // r9
  int v31; // r11d
  __int64 v32; // r11
  unsigned __int64 *v33; // rbx
  unsigned __int64 v34; // r11
  unsigned __int64 *v35; // r12
  unsigned __int64 *v36; // rbx
  __int64 v37; // rax
  unsigned __int64 v38; // rdx
  __int64 v39; // rcx
  int v40; // edx
  __int64 v41; // rax
  __int64 v42; // rdx
  _QWORD *v43; // rax
  const char *v44; // rsi
  const char **v45; // r12
  __int64 *v46; // rdx
  __int64 v47; // r13
  _QWORD *v48; // rax
  __int64 v49; // rbx
  __int64 *v50; // r14
  int v51; // edx
  int v52; // eax
  _QWORD *v53; // rax
  const char **v54; // r14
  const char *v55; // rsi
  __int64 v56; // rbx
  __int64 v57; // r14
  __int64 *v58; // rdx
  __int64 v59; // r13
  _QWORD *v60; // rax
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rax
  __int64 v64; // rax
  __int64 v65; // rax
  __int64 v66; // rax
  int v67; // ebx
  __int64 v68; // rax
  int v69; // ebx
  __int64 v70; // rbx
  int v71; // eax
  unsigned int v72; // edx
  __int64 v73; // rax
  __int64 v74; // rdx
  __int64 v75; // rdx
  __int64 v76; // r14
  __int64 v77; // r13
  int v78; // eax
  __int64 v79; // r13
  _QWORD *v80; // rax
  __int64 v81; // rdx
  __int64 v82; // rax
  __int64 *v83; // rax
  int v84; // eax
  __int64 v85; // rdx
  char *v86; // rax
  _QWORD *v87; // r13
  __int64 v88; // rdx
  __int64 v89; // r14
  __int64 *v90; // r13
  __int64 v91; // r14
  unsigned int v92; // ecx
  __int64 v93; // r10
  char v94; // al
  __int64 v95; // rax
  unsigned __int64 *v96; // rdi
  __int64 v97; // rsi
  unsigned __int8 *v98; // rsi
  __int64 v99; // rsi
  unsigned __int8 *v100; // rsi
  __int64 *v101; // rdi
  __int64 *v102; // rsi
  __int64 v103; // rax
  int v104; // edx
  char v105; // dl
  __int64 v106; // rax
  __int64 v107; // [rsp+0h] [rbp-100h]
  int v108; // [rsp+10h] [rbp-F0h]
  __int64 v109; // [rsp+18h] [rbp-E8h]
  __int64 v110; // [rsp+20h] [rbp-E0h]
  __int64 v111; // [rsp+20h] [rbp-E0h]
  __int64 v112; // [rsp+28h] [rbp-D8h]
  __int64 v114; // [rsp+28h] [rbp-D8h]
  __int64 v115; // [rsp+28h] [rbp-D8h]
  __int64 v116; // [rsp+28h] [rbp-D8h]
  __int64 v117; // [rsp+28h] [rbp-D8h]
  __int64 v118; // [rsp+28h] [rbp-D8h]
  __int64 v119; // [rsp+28h] [rbp-D8h]
  __int64 v120; // [rsp+28h] [rbp-D8h]
  __int64 v121; // [rsp+30h] [rbp-D0h]
  __int64 v122; // [rsp+38h] [rbp-C8h]
  _BYTE v123[32]; // [rsp+40h] [rbp-C0h] BYREF
  __int16 v124; // [rsp+60h] [rbp-A0h]
  _BYTE *v125; // [rsp+70h] [rbp-90h] BYREF
  __int64 v126; // [rsp+78h] [rbp-88h]
  _BYTE v127[32]; // [rsp+80h] [rbp-80h] BYREF
  char *v128; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v129; // [rsp+A8h] [rbp-58h]
  _QWORD v130[2]; // [rsp+B0h] [rbp-50h] BYREF
  __int16 v131; // [rsp+C0h] [rbp-40h]

  v7 = a2;
  if ( *(_BYTE *)a2 == 22 )
  {
    v8 = 1;
    v9 = *(_QWORD *)(*(_QWORD *)(a2 + 24) + 80LL);
    if ( v9 )
      v9 -= 24;
    v11 = sub_AA4FF0(v9);
    v12 = 0;
    if ( v11 )
      v12 = v10;
    BYTE1(v8) = v12;
    v13 = (__int64 *)sub_BD5C60(a2);
    v131 = 257;
    v112 = sub_BCE3C0(v13, a3);
    v14 = sub_BD2C40(72, unk_3F10A14);
    v15 = (__int64)v14;
    if ( v14 )
      sub_B51C90((__int64)v14, a2, v112, (__int64)&v128, 0, 0);
    sub_B44220((_QWORD *)v15, v11, v8);
    v128 = (char *)sub_BD5D20(a2);
    v131 = 773;
    v129 = v16;
    v130[0] = ".cast";
    sub_BD6B50((unsigned __int8 *)v15, (const char **)&v128);
    return v15;
  }
  if ( *(_BYTE *)a2 > 0x1Cu )
  {
    v110 = *(_QWORD *)(a2 + 8);
    v21 = (__int64 *)sub_BCE3C0(*(__int64 **)v110, a3);
    v22 = a6;
    v23 = (__int64)v21;
    v24 = *(unsigned __int8 *)(v110 + 8);
    if ( (unsigned int)(v24 - 17) <= 1 )
    {
      BYTE4(v121) = (_BYTE)v24 == 18;
      LODWORD(v121) = *(_DWORD *)(v110 + 32);
      v25 = sub_BCE1B0(v21, v121);
      v22 = a6;
      v23 = v25;
    }
    v26 = *(_BYTE *)a2;
    if ( *(_BYTE *)a2 == 79 )
    {
      if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
        v46 = *(__int64 **)(a2 - 8);
      else
        v46 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
      v47 = *v46;
      v116 = v23;
      if ( *(_QWORD *)(*v46 + 8) == v23 )
      {
        v15 = *v46;
        goto LABEL_35;
      }
      v131 = 257;
      v48 = sub_BD2C40(72, unk_3F10A14);
      v15 = (__int64)v48;
      if ( !v48 )
        return v15;
      sub_B51BF0((__int64)v48, v47, v116, (__int64)&v128, 0, 0);
LABEL_34:
      if ( !v15 )
        return v15;
      goto LABEL_35;
    }
    if ( ((v26 - 61) & 0xDF) == 0 )
    {
      v115 = v23;
      v131 = 257;
      v43 = sub_BD2C40(72, unk_3F10A14);
      v15 = (__int64)v43;
      if ( v43 )
        sub_B51C90((__int64)v43, a2, v115, (__int64)&v128, 0, 0);
      sub_B43DD0(v15, a2);
      goto LABEL_34;
    }
    if ( v26 == 85 )
    {
      v95 = *(_QWORD *)(a2 - 32);
      if ( v95 )
      {
        if ( !*(_BYTE *)v95 && *(_QWORD *)(v95 + 24) == *(_QWORD *)(a2 + 80) && (*(_BYTE *)(v95 + 33) & 0x20) != 0 )
        {
          if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
            v96 = *(unsigned __int64 **)(a2 - 8);
          else
            v96 = (unsigned __int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
          sub_27CE9A0(v96, a3, a4, a5, v22);
          v15 = sub_DF9C00(*(_QWORD *)(a1 + 24));
          if ( !v15 )
            return v15;
LABEL_35:
          if ( *(_BYTE *)v15 > 0x1Cu && !*(_QWORD *)(v15 + 40) )
          {
            sub_B44220((_QWORD *)v15, v7 + 24, 0);
            sub_BD6B90((unsigned __int8 *)v15, (unsigned __int8 *)v7);
            v44 = *(const char **)(v7 + 48);
            v45 = (const char **)(v15 + 48);
            v128 = (char *)v44;
            if ( v44 )
            {
              sub_B96E90((__int64)&v128, (__int64)v44, 1);
              if ( v45 == (const char **)&v128 )
              {
                if ( v128 )
                  sub_B91220(v15 + 48, (__int64)v128);
                return v15;
              }
              v97 = *(_QWORD *)(v15 + 48);
              if ( !v97 )
              {
LABEL_130:
                v98 = (unsigned __int8 *)v128;
                *(_QWORD *)(v15 + 48) = v128;
                if ( v98 )
                  sub_B976B0((__int64)&v128, v98, v15 + 48);
                return v15;
              }
LABEL_129:
              sub_B91220(v15 + 48, v97);
              goto LABEL_130;
            }
            if ( v45 != (const char **)&v128 )
            {
              v97 = *(_QWORD *)(v15 + 48);
              if ( v97 )
                goto LABEL_129;
            }
          }
          return v15;
        }
      }
    }
    v111 = v22;
    v114 = v23;
    v27 = sub_DF9B70(*(_QWORD *)(a1 + 24));
    v29 = v114;
    v30 = v111;
    if ( v27 == -1 )
    {
      v31 = *(_DWORD *)(a2 + 4);
      v125 = v127;
      v126 = 0x400000000LL;
      v32 = 4LL * (v31 & 0x7FFFFFF);
      if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      {
        v33 = *(unsigned __int64 **)(a2 - 8);
        v34 = (unsigned __int64)&v33[v32];
      }
      else
      {
        v33 = (unsigned __int64 *)(a2 - v32 * 8);
        v34 = a2;
      }
      if ( v33 != (unsigned __int64 *)v34 )
      {
        v35 = v33;
        v36 = (unsigned __int64 *)v34;
        do
        {
          v39 = *(_QWORD *)(*v35 + 8);
          v40 = *(unsigned __int8 *)(v39 + 8);
          if ( (unsigned int)(v40 - 17) <= 1 )
            LOBYTE(v40) = *(_BYTE *)(**(_QWORD **)(v39 + 16) + 8LL);
          if ( (_BYTE)v40 == 14 )
          {
            v41 = sub_27CE9A0(v35, a3, a4, a5, v111);
            v42 = (unsigned int)v126;
            v28 = (unsigned int)v126 + 1LL;
            if ( v28 > HIDWORD(v126) )
            {
              v107 = v41;
              sub_C8D5F0((__int64)&v125, v127, (unsigned int)v126 + 1LL, 8u, v28, v30);
              v42 = (unsigned int)v126;
              v41 = v107;
            }
            *(_QWORD *)&v125[8 * v42] = v41;
            LODWORD(v126) = v126 + 1;
          }
          else
          {
            v37 = (unsigned int)v126;
            v38 = (unsigned int)v126 + 1LL;
            if ( v38 > HIDWORD(v126) )
            {
              sub_C8D5F0((__int64)&v125, v127, v38, 8u, v28, v30);
              v37 = (unsigned int)v126;
            }
            *(_QWORD *)&v125[8 * v37] = 0;
            LODWORD(v126) = v126 + 1;
          }
          v35 += 4;
        }
        while ( v36 != v35 );
        v29 = v114;
        v7 = a2;
      }
      switch ( *(_BYTE *)v7 )
      {
        case '?':
          v84 = *(_DWORD *)(v7 + 4);
          v124 = 257;
          v128 = (char *)v130;
          v85 = 32 * (1LL - (v84 & 0x7FFFFFF));
          v129 = 0x400000000LL;
          v86 = (char *)v130;
          v87 = (_QWORD *)(v7 + v85);
          v88 = -v85;
          v89 = v88 >> 5;
          if ( (unsigned __int64)v88 > 0x80 )
          {
            sub_C8D5F0((__int64)&v128, v130, v88 >> 5, 8u, v28, v30);
            v86 = &v128[8 * (unsigned int)v129];
          }
          for ( ; (_QWORD *)v7 != v87; v86 += 8 )
          {
            if ( v86 )
              *(_QWORD *)v86 = *v87;
            v87 += 4;
          }
          v90 = (__int64 *)v128;
          LODWORD(v129) = v89 + v129;
          v91 = (unsigned int)v129;
          v120 = *(_QWORD *)v125;
          v109 = *(_QWORD *)(v7 + 72);
          v108 = v129 + 1;
          v15 = (__int64)sub_BD2C40(88, (int)v129 + 1);
          if ( !v15 )
            goto LABEL_115;
          v92 = v108 & 0x7FFFFFF;
          v93 = *(_QWORD *)(v120 + 8);
          if ( (unsigned int)*(unsigned __int8 *)(v93 + 8) - 17 <= 1 )
            goto LABEL_114;
          v101 = &v90[v91];
          if ( v90 == v101 )
            goto LABEL_114;
          v102 = v90;
          break;
        case 'M':
          if ( (*(_BYTE *)(v7 + 7) & 0x40) != 0 )
            v81 = *(_QWORD *)(v7 - 8);
          else
            v81 = v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF);
          v82 = *(_QWORD *)v81;
          if ( (*(_BYTE *)(*(_QWORD *)v81 + 7LL) & 0x40) != 0 )
            v83 = *(__int64 **)(v82 - 8);
          else
            v83 = (__int64 *)(v82 - 32LL * (*(_DWORD *)(v82 + 4) & 0x7FFFFFF));
          v15 = *v83;
          if ( *(_QWORD *)(*v83 + 8) != v29 )
          {
            v131 = 257;
            v15 = sub_B52190(v15, v29, (__int64)&v128, 0, 0);
          }
          goto LABEL_82;
        case 'N':
          v119 = v29;
          v79 = *(_QWORD *)v125;
          v131 = 257;
          v80 = sub_BD2C40(72, unk_3F10A14);
          v15 = (__int64)v80;
          if ( v80 )
            sub_B51BF0((__int64)v80, v79, v119, (__int64)&v128, 0, 0);
          goto LABEL_82;
        case 'T':
          v67 = *(_DWORD *)(v7 + 4);
          v118 = v29;
          v131 = 257;
          v68 = sub_BD2DA0(80);
          v69 = v67 & 0x7FFFFFF;
          v15 = v68;
          if ( v68 )
          {
            sub_B44260(v68, v118, 55, 0x8000000u, 0, 0);
            *(_DWORD *)(v15 + 72) = v69;
            sub_BD6B50((unsigned __int8 *)v15, (const char **)&v128);
            sub_BD2A10(v15, *(_DWORD *)(v15 + 72), 1);
          }
          if ( (*(_DWORD *)(v7 + 4) & 0x7FFFFFF) != 0 )
          {
            v70 = 0;
            do
            {
              v76 = *(_QWORD *)(*(_QWORD *)(v7 - 8) + 32LL * *(unsigned int *)(v7 + 72) + 8 * v70);
              v77 = *(_QWORD *)&v125[8 * v70];
              v78 = *(_DWORD *)(v15 + 4) & 0x7FFFFFF;
              if ( v78 == *(_DWORD *)(v15 + 72) )
              {
                sub_B48D90(v15);
                v78 = *(_DWORD *)(v15 + 4) & 0x7FFFFFF;
              }
              v71 = (v78 + 1) & 0x7FFFFFF;
              v72 = v71 | *(_DWORD *)(v15 + 4) & 0xF8000000;
              v73 = *(_QWORD *)(v15 - 8) + 32LL * (unsigned int)(v71 - 1);
              *(_DWORD *)(v15 + 4) = v72;
              if ( *(_QWORD *)v73 )
              {
                v74 = *(_QWORD *)(v73 + 8);
                **(_QWORD **)(v73 + 16) = v74;
                if ( v74 )
                  *(_QWORD *)(v74 + 16) = *(_QWORD *)(v73 + 16);
              }
              *(_QWORD *)v73 = v77;
              if ( v77 )
              {
                v75 = *(_QWORD *)(v77 + 16);
                *(_QWORD *)(v73 + 8) = v75;
                if ( v75 )
                  *(_QWORD *)(v75 + 16) = v73 + 8;
                *(_QWORD *)(v73 + 16) = v77 + 16;
                *(_QWORD *)(v77 + 16) = v73;
              }
              ++v70;
              *(_QWORD *)(*(_QWORD *)(v15 - 8)
                        + 32LL * *(unsigned int *)(v15 + 72)
                        + 8LL * ((*(_DWORD *)(v15 + 4) & 0x7FFFFFFu) - 1)) = v76;
            }
            while ( (*(_DWORD *)(v7 + 4) & 0x7FFFFFFu) > (unsigned int)v70 );
          }
          goto LABEL_82;
        case 'V':
          v131 = 257;
          v56 = *((_QWORD *)v125 + 2);
          v57 = *((_QWORD *)v125 + 1);
          if ( (*(_BYTE *)(v7 + 7) & 0x40) != 0 )
            v58 = *(__int64 **)(v7 - 8);
          else
            v58 = (__int64 *)(v7 - 32LL * (*(_DWORD *)(v7 + 4) & 0x7FFFFFF));
          v59 = *v58;
          v60 = sub_BD2C40(72, 3u);
          v15 = (__int64)v60;
          if ( v60 )
          {
            v117 = (__int64)v60;
            sub_B44260((__int64)v60, *(_QWORD *)(v57 + 8), 57, 3u, 0, 0);
            if ( *(_QWORD *)(v15 - 96) )
            {
              v61 = *(_QWORD *)(v15 - 88);
              **(_QWORD **)(v15 - 80) = v61;
              if ( v61 )
                *(_QWORD *)(v61 + 16) = *(_QWORD *)(v15 - 80);
            }
            *(_QWORD *)(v15 - 96) = v59;
            if ( v59 )
            {
              v62 = *(_QWORD *)(v59 + 16);
              *(_QWORD *)(v15 - 88) = v62;
              if ( v62 )
                *(_QWORD *)(v62 + 16) = v15 - 88;
              *(_QWORD *)(v15 - 80) = v59 + 16;
              *(_QWORD *)(v59 + 16) = v15 - 96;
            }
            if ( *(_QWORD *)(v15 - 64) )
            {
              v63 = *(_QWORD *)(v15 - 56);
              **(_QWORD **)(v15 - 48) = v63;
              if ( v63 )
                *(_QWORD *)(v63 + 16) = *(_QWORD *)(v15 - 48);
            }
            *(_QWORD *)(v15 - 64) = v57;
            v64 = *(_QWORD *)(v57 + 16);
            *(_QWORD *)(v15 - 56) = v64;
            if ( v64 )
              *(_QWORD *)(v64 + 16) = v15 - 56;
            *(_QWORD *)(v15 - 48) = v57 + 16;
            *(_QWORD *)(v57 + 16) = v15 - 64;
            if ( *(_QWORD *)(v15 - 32) )
            {
              v65 = *(_QWORD *)(v15 - 24);
              **(_QWORD **)(v15 - 16) = v65;
              if ( v65 )
                *(_QWORD *)(v65 + 16) = *(_QWORD *)(v15 - 16);
            }
            *(_QWORD *)(v15 - 32) = v56;
            if ( v56 )
            {
              v66 = *(_QWORD *)(v56 + 16);
              *(_QWORD *)(v15 - 24) = v66;
              if ( v66 )
                *(_QWORD *)(v66 + 16) = v15 - 24;
              *(_QWORD *)(v15 - 16) = v56 + 16;
              *(_QWORD *)(v56 + 16) = v15 - 32;
            }
            sub_BD6B50((unsigned __int8 *)v15, (const char **)&v128);
            sub_B47C00(v117, v7, 0, 0);
          }
          else
          {
            sub_B47C00(0, v7, 0, 0);
          }
          goto LABEL_82;
        default:
          BUG();
      }
      while ( 1 )
      {
        v103 = *(_QWORD *)(*v102 + 8);
        v104 = *(unsigned __int8 *)(v103 + 8);
        if ( v104 == 17 )
        {
          v105 = 0;
          goto LABEL_152;
        }
        if ( v104 == 18 )
          break;
        if ( v101 == ++v102 )
          goto LABEL_114;
      }
      v105 = 1;
LABEL_152:
      BYTE4(v122) = v105;
      LODWORD(v122) = *(_DWORD *)(v103 + 32);
      v106 = sub_BCE1B0((__int64 *)v93, v122);
      v92 = v108 & 0x7FFFFFF;
      v93 = v106;
LABEL_114:
      sub_B44260(v15, v93, 34, v92, 0, 0);
      *(_QWORD *)(v15 + 72) = v109;
      *(_QWORD *)(v15 + 80) = sub_B4DC50(v109, (__int64)v90, v91);
      sub_B4D9A0(v15, v120, v90, v91, (__int64)v123);
LABEL_115:
      if ( v128 != (char *)v130 )
        _libc_free((unsigned __int64)v128);
      v94 = sub_B4DE30(v7);
      sub_B4DE00(v15, v94);
LABEL_82:
      if ( v125 != v127 )
        _libc_free((unsigned __int64)v125);
      goto LABEL_34;
    }
    v49 = *(_QWORD *)(a2 + 8);
    v50 = (__int64 *)sub_BCE3C0(*(__int64 **)v49, v27);
    v51 = *(unsigned __int8 *)(v49 + 8);
    if ( (unsigned int)(v51 - 17) <= 1 )
    {
      v52 = *(_DWORD *)(v49 + 32);
      BYTE4(v125) = (_BYTE)v51 == 18;
      LODWORD(v125) = v52;
      v50 = (__int64 *)sub_BCE1B0(v50, (__int64)v125);
    }
    v131 = 257;
    v53 = sub_BD2C40(72, unk_3F10A14);
    v15 = (__int64)v53;
    if ( v53 )
      sub_B51C90((__int64)v53, a2, (__int64)v50, (__int64)&v128, 0, 0);
    v54 = (const char **)(v15 + 48);
    sub_B43E90(v15, a2 + 24);
    v55 = *(const char **)(a2 + 48);
    v128 = (char *)v55;
    if ( v55 )
    {
      sub_B96E90((__int64)&v128, (__int64)v55, 1);
      if ( v54 == (const char **)&v128 )
      {
        if ( v128 )
          sub_B91220((__int64)&v128, (__int64)v128);
        goto LABEL_34;
      }
      v99 = *(_QWORD *)(v15 + 48);
      if ( !v99 )
        goto LABEL_138;
    }
    else
    {
      if ( v54 == (const char **)&v128 )
        goto LABEL_34;
      v99 = *(_QWORD *)(v15 + 48);
      if ( !v99 )
        goto LABEL_34;
    }
    sub_B91220(v15 + 48, v99);
LABEL_138:
    v100 = (unsigned __int8 *)v128;
    *(_QWORD *)(v15 + 48) = v128;
    if ( v100 )
      sub_B976B0((__int64)&v128, v100, v15 + 48);
    goto LABEL_34;
  }
  return sub_27CED90(a2, a3, a4, a4, a5, a6);
}
