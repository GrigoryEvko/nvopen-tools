// Function: sub_2D60630
// Address: 0x2d60630
//
__int64 __fastcall sub_2D60630(__int64 a1, __int64 a2, _DWORD *a3)
{
  unsigned int v3; // ebx
  _QWORD *v4; // r14
  unsigned __int64 v5; // r12
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v9; // rdi
  __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  _BYTE *v14; // rdi
  __int64 v15; // r10
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // r8
  bool v19; // al
  __int64 v20; // r8
  __int64 v21; // rbx
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // r9
  __int64 v25; // rcx
  __int64 v26; // r10
  __int64 v27; // r8
  _QWORD *v28; // rax
  __int64 v29; // r9
  __int64 v30; // r12
  __int64 v31; // rsi
  __int64 v32; // r15
  unsigned int *v33; // r15
  const char **v34; // rbx
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rcx
  __int64 v40; // r15
  __int64 v41; // r13
  __int64 v42; // r15
  __int64 v43; // rsi
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rdi
  bool v51; // al
  bool v52; // zf
  unsigned __int8 v53; // al
  __int64 v54; // rdi
  __int64 v55; // rdx
  __int64 v56; // r8
  __int64 v57; // rcx
  _BYTE *v58; // rdi
  bool v59; // al
  __int64 v60; // rdi
  bool v61; // al
  unsigned __int8 v62; // al
  __int64 v63; // rdi
  __int64 v64; // rdx
  __int64 v65; // r8
  _BYTE *v66; // rdi
  bool v67; // al
  __int64 *v68; // rax
  __int64 v69; // rdx
  bool v70; // al
  __int64 *v71; // rdx
  __int64 v72; // rax
  __int64 v73; // rax
  __int64 v74; // rdx
  char v75; // cl
  __int64 v76; // rax
  __int64 v77; // rax
  __int64 v78; // rax
  bool v79; // al
  __int64 v80; // [rsp+0h] [rbp-170h]
  __int64 v81; // [rsp+8h] [rbp-168h]
  __int64 v82; // [rsp+10h] [rbp-160h]
  __int64 v83; // [rsp+10h] [rbp-160h]
  __int64 v84; // [rsp+20h] [rbp-150h]
  __int64 v85; // [rsp+28h] [rbp-148h]
  int v86; // [rsp+34h] [rbp-13Ch]
  __int64 v88; // [rsp+40h] [rbp-130h]
  __int64 v89; // [rsp+40h] [rbp-130h]
  __int64 *v90; // [rsp+40h] [rbp-130h]
  char *v91; // [rsp+40h] [rbp-130h]
  __int64 v92; // [rsp+40h] [rbp-130h]
  __int64 v93; // [rsp+40h] [rbp-130h]
  __int64 v94; // [rsp+40h] [rbp-130h]
  _QWORD *v95; // [rsp+48h] [rbp-128h]
  __int64 v97; // [rsp+58h] [rbp-118h]
  __int64 v98; // [rsp+60h] [rbp-110h]
  __int64 v99; // [rsp+60h] [rbp-110h]
  _QWORD *v100; // [rsp+68h] [rbp-108h]
  __int64 v101; // [rsp+78h] [rbp-F8h] BYREF
  _QWORD v102[4]; // [rsp+80h] [rbp-F0h] BYREF
  __int16 v103; // [rsp+A0h] [rbp-D0h]
  const char **v104; // [rsp+B0h] [rbp-C0h] BYREF
  __int64 v105; // [rsp+B8h] [rbp-B8h]
  const char *v106; // [rsp+C0h] [rbp-B0h] BYREF
  __int16 v107; // [rsp+D0h] [rbp-A0h]
  __int64 *v108; // [rsp+E0h] [rbp-90h]
  __int64 *v109; // [rsp+E8h] [rbp-88h]
  __int64 v110; // [rsp+F0h] [rbp-80h]
  __int64 v111; // [rsp+F8h] [rbp-78h]
  void **v112; // [rsp+100h] [rbp-70h]
  void **v113; // [rsp+108h] [rbp-68h]
  __int64 v114; // [rsp+110h] [rbp-60h]
  int v115; // [rsp+118h] [rbp-58h]
  __int16 v116; // [rsp+11Ch] [rbp-54h]
  char v117; // [rsp+11Eh] [rbp-52h]
  __int64 v118; // [rsp+120h] [rbp-50h]
  __int64 v119; // [rsp+128h] [rbp-48h]
  void *v120; // [rsp+130h] [rbp-40h] BYREF
  void *v121; // [rsp+138h] [rbp-38h] BYREF

  if ( (*(_BYTE *)(*(_QWORD *)a1 + 865LL) & 8) == 0 )
    return 0;
  v3 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 16) + 56LL);
  if ( (_BYTE)v3 )
  {
    return 0;
  }
  else
  {
    v4 = *(_QWORD **)(a2 + 80);
    v100 = (_QWORD *)(a2 + 72);
    if ( v4 != (_QWORD *)(a2 + 72) )
    {
      while ( 1 )
      {
        if ( !v4 )
          BUG();
        v5 = v4[3] & 0xFFFFFFFFFFFFFFF8LL;
        if ( (_QWORD *)v5 == v4 + 3 )
          goto LABEL_146;
        if ( !v5 )
          BUG();
        if ( (unsigned int)*(unsigned __int8 *)(v5 - 24) - 30 > 0xA )
LABEL_146:
          BUG();
        if ( *(_BYTE *)(v5 - 24) != 31 )
          goto LABEL_5;
        if ( (*(_DWORD *)(v5 - 20) & 0x7FFFFFF) != 3 )
          goto LABEL_5;
        v6 = *(_QWORD *)(v5 - 120);
        v7 = *(_QWORD *)(v6 + 16);
        if ( !v7 )
          goto LABEL_5;
        if ( *(_QWORD *)(v7 + 8) )
          goto LABEL_5;
        if ( *(_BYTE *)v6 <= 0x1Cu )
          goto LABEL_5;
        v98 = *(_QWORD *)(v5 - 56);
        if ( !v98 )
          goto LABEL_5;
        v97 = *(_QWORD *)(v5 - 88);
        if ( !v97 )
          goto LABEL_5;
        v8 = v5 - 24;
        if ( (*(_BYTE *)(v5 - 17) & 0x20) != 0 )
        {
          if ( sub_B91C10(v5 - 24, 15) )
            goto LABEL_5;
        }
        if ( v98 == v97 )
          goto LABEL_5;
        v9 = *(_QWORD *)(v6 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v9 + 8) - 17 <= 1 )
          v9 = **(_QWORD **)(v9 + 16);
        if ( !sub_BCAC40(v9, 1) )
          break;
        if ( *(_BYTE *)v6 == 57 )
        {
          v68 = (__int64 *)sub_986520(v6);
          v15 = *v68;
          v69 = *(_QWORD *)(*v68 + 16);
          if ( !v69 )
            break;
          if ( *(_QWORD *)(v69 + 8) )
            break;
          v95 = (_QWORD *)v68[4];
          v49 = v95[2];
          if ( !v49 )
            break;
        }
        else
        {
          v10 = *(_QWORD *)(v6 + 8);
          if ( *(_BYTE *)v6 != 86 )
            goto LABEL_26;
          v88 = *(_QWORD *)(v6 - 96);
          if ( *(_QWORD *)(v88 + 8) != v10 || **(_BYTE **)(v6 - 32) > 0x15u )
            goto LABEL_26;
          v95 = *(_QWORD **)(v6 - 64);
          if ( !sub_AC30F0(*(_QWORD *)(v6 - 32)) )
            break;
          v15 = v88;
          v48 = *(_QWORD *)(v88 + 16);
          if ( !v48 )
            break;
          if ( *(_QWORD *)(v48 + 8) )
            break;
          v49 = v95[2];
          if ( !v49 )
            break;
        }
        if ( *(_QWORD *)(v49 + 8) )
          break;
        v86 = 28;
LABEL_39:
        if ( *(_BYTE *)v15 <= 0x1Cu )
          goto LABEL_5;
        if ( (unsigned __int8)(*(_BYTE *)v15 - 82) <= 1u )
          goto LABEL_41;
        v50 = *(_QWORD *)(v15 + 8);
        if ( (unsigned int)*(unsigned __int8 *)(v50 + 8) - 17 <= 1 )
          v50 = **(_QWORD **)(v50 + 16);
        v91 = (char *)v15;
        v51 = sub_BCAC40(v50, 1);
        v15 = (__int64)v91;
        v52 = !v51;
        v53 = *v91;
        if ( v52 )
          goto LABEL_121;
        if ( v53 == 57 )
          goto LABEL_41;
        if ( v53 != 86 )
          goto LABEL_121;
        v54 = *((_QWORD *)v91 + 1);
        if ( *(_QWORD *)(*((_QWORD *)v91 - 12) + 8LL) == v54 && **((_BYTE **)v91 - 4) <= 0x15u )
        {
          v70 = sub_AC30F0(*((_QWORD *)v91 - 4));
          v15 = (__int64)v91;
          if ( v70 )
            goto LABEL_41;
          v53 = *v91;
LABEL_121:
          if ( v53 <= 0x1Cu )
            goto LABEL_5;
          v54 = *(_QWORD *)(v15 + 8);
        }
        if ( (unsigned int)*(unsigned __int8 *)(v54 + 8) - 17 <= 1 )
          v54 = **(_QWORD **)(v54 + 16);
        v92 = v15;
        if ( !sub_BCAC40(v54, 1) )
          goto LABEL_5;
        v15 = v92;
        if ( *(_BYTE *)v92 != 58 )
        {
          if ( *(_BYTE *)v92 != 86 )
            goto LABEL_5;
          v57 = *(_QWORD *)(v92 + 8);
          if ( *(_QWORD *)(*(_QWORD *)(v92 - 96) + 8LL) != v57 )
            goto LABEL_5;
          v58 = *(_BYTE **)(v92 - 64);
          if ( *v58 > 0x15u )
            goto LABEL_5;
          v59 = sub_AD7A80(v58, 1, v55, v57, v56);
          v15 = v92;
          if ( !v59 )
            goto LABEL_5;
        }
LABEL_41:
        if ( *(_BYTE *)v95 > 0x1Cu )
        {
          if ( (unsigned __int8)(*(_BYTE *)v95 - 82) <= 1u )
            goto LABEL_43;
          v60 = v95[1];
          if ( (unsigned int)*(unsigned __int8 *)(v60 + 8) - 17 <= 1 )
            v60 = **(_QWORD **)(v60 + 16);
          v93 = v15;
          v61 = sub_BCAC40(v60, 1);
          v15 = v93;
          if ( !v61 )
          {
            v62 = *(_BYTE *)v95;
            goto LABEL_130;
          }
          v62 = *(_BYTE *)v95;
          if ( *(_BYTE *)v95 == 57 )
            goto LABEL_43;
          if ( v62 == 86 )
          {
            v63 = v95[1];
            if ( *(_QWORD *)(*(v95 - 12) + 8LL) == v63 && *(_BYTE *)*(v95 - 4) <= 0x15u )
            {
              v79 = sub_AC30F0(*(v95 - 4));
              v15 = v93;
              if ( v79 )
                goto LABEL_43;
              v62 = *(_BYTE *)v95;
              goto LABEL_130;
            }
          }
          else
          {
LABEL_130:
            if ( v62 <= 0x1Cu )
              goto LABEL_5;
            v63 = v95[1];
          }
          if ( (unsigned int)*(unsigned __int8 *)(v63 + 8) - 17 <= 1 )
            v63 = **(_QWORD **)(v63 + 16);
          v94 = v15;
          if ( !sub_BCAC40(v63, 1) )
            goto LABEL_5;
          v15 = v94;
          if ( *(_BYTE *)v95 != 58 )
          {
            if ( *(_BYTE *)v95 != 86 )
              goto LABEL_5;
            if ( *(_QWORD *)(*(v95 - 12) + 8LL) != v95[1] )
              goto LABEL_5;
            v66 = (_BYTE *)*(v95 - 8);
            if ( *v66 > 0x15u )
              goto LABEL_5;
            v67 = sub_AD7A80(v66, 1, v64, (__int64)v95, v65);
            v15 = v94;
            if ( !v67 )
              goto LABEL_5;
          }
LABEL_43:
          v18 = v4[1];
          if ( v18 == v4[6] + 72LL || (v19 = v18 != 0, v20 = v18 - 24, !v19) )
            v20 = 0;
          v21 = (__int64)(v4 - 3);
          v84 = v15;
          v82 = v4[6];
          v81 = v20;
          v104 = (const char **)sub_BD5D20((__int64)(v4 - 3));
          v107 = 773;
          v105 = v22;
          v106 = ".cond.split";
          v85 = sub_AA48A0((__int64)(v4 - 3));
          v23 = sub_22077B0(0x50u);
          v25 = v82;
          v90 = (__int64 *)v23;
          v26 = v84;
          v27 = v81;
          if ( v23 )
          {
            sub_AA4D50(v23, v85, (__int64)&v104, v82, v81);
            v26 = v84;
          }
          if ( *(_BYTE *)(a1 + 832) )
          {
            v83 = v26;
            sub_D695C0((__int64)&v104, a1 + 840, v90, v25, v27, v24);
            v26 = v83;
          }
          sub_AC2B30(v5 - 120, v26);
          sub_B43D60((_QWORD *)v6);
          if ( v86 == 28 )
            sub_AC2B30(v5 - 56, (__int64)v90);
          else
            sub_AC2B30(v5 - 88, (__int64)v90);
          v111 = sub_AA48A0((__int64)v90);
          v112 = &v120;
          v113 = &v121;
          v105 = 0x200000000LL;
          v116 = 512;
          v120 = &unk_49DA100;
          v103 = 257;
          v104 = &v106;
          v121 = &unk_49DA0B0;
          LOWORD(v110) = 0;
          v114 = 0;
          v115 = 0;
          v117 = 7;
          v118 = 0;
          v119 = 0;
          v108 = v90;
          v109 = v90 + 6;
          v28 = sub_BD2C40(72, 3u);
          v30 = (__int64)v28;
          if ( v28 )
            sub_B4C9A0((__int64)v28, v98, v97, (__int64)v95, 3u, v29, 0, 0);
          v31 = v30;
          (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64 *, __int64))*v113 + 2))(
            v113,
            v30,
            v102,
            v109,
            v110);
          v32 = 2LL * (unsigned int)v105;
          if ( v104 != &v104[v32] )
          {
            v33 = (unsigned int *)&v104[v32];
            v34 = v104;
            do
            {
              v35 = (__int64)v34[1];
              v31 = *(unsigned int *)v34;
              v34 += 2;
              sub_B99FD0(v30, v31, v35);
            }
            while ( v33 != (unsigned int *)v34 );
            v21 = (__int64)(v4 - 3);
          }
          sub_F94A20(&v104, v31);
          if ( *(_BYTE *)v95 > 0x1Cu )
          {
            sub_B43D10(v95);
            v36 = v80;
            LOWORD(v36) = 0;
            v80 = v36;
            sub_B44220(v95, v30 + 24, v36);
          }
          if ( v86 == 29 )
          {
            sub_AA5D60(v97, v21, (__int64)v90);
            v73 = sub_AA5930(v98);
            v39 = v74;
            v40 = v73;
            if ( v73 != v74 )
              goto LABEL_61;
          }
          else
          {
            sub_AA5D60(v98, v21, (__int64)v90);
            v37 = sub_AA5930(v97);
            v39 = v38;
            v40 = v37;
            if ( v38 == v37 )
              goto LABEL_72;
LABEL_61:
            v99 = v8;
            v41 = v40;
            v42 = v39;
            do
            {
              v43 = *(_QWORD *)(v41 - 8);
              v44 = 0x1FFFFFFFE0LL;
              if ( (*(_DWORD *)(v41 + 4) & 0x7FFFFFF) != 0 )
              {
                v45 = 0;
                do
                {
                  if ( v21 == *(_QWORD *)(v43 + 32LL * *(unsigned int *)(v41 + 72) + 8 * v45) )
                  {
                    v44 = 32 * v45;
                    goto LABEL_67;
                  }
                  ++v45;
                }
                while ( (*(_DWORD *)(v41 + 4) & 0x7FFFFFF) != (_DWORD)v45 );
                v44 = 0x1FFFFFFFE0LL;
              }
LABEL_67:
              sub_F0A850(v41, *(_QWORD *)(v43 + v44), (__int64)v90);
              v46 = *(_QWORD *)(v41 + 32);
              if ( !v46 )
                BUG();
              v41 = 0;
              if ( *(_BYTE *)(v46 - 24) == 84 )
                v41 = v46 - 24;
            }
            while ( v41 != v42 );
            v8 = v99;
            if ( v86 != 29 )
            {
LABEL_72:
              if ( !(unsigned __int8)sub_BC8C50(v8, &v101, v102) )
              {
LABEL_73:
                v3 = 1;
                *a3 = 1;
                goto LABEL_5;
              }
              v78 = sub_BD5C60(v8);
              v75 = 0;
              v104 = (const char **)v78;
LABEL_136:
              v76 = sub_B8C2F0(&v104, v101, v102[0], v75);
              sub_B99FD0(v8, 2u, v76);
              v104 = (const char **)sub_BD5C60(v30);
              v77 = sub_B8C2F0(&v104, v101, v102[0], 0);
              sub_B99FD0(v30, 2u, v77);
              goto LABEL_73;
            }
          }
          if ( !(unsigned __int8)sub_BC8C50(v8, &v101, v102) )
            goto LABEL_73;
          v104 = (const char **)sub_BD5C60(v8);
          v75 = sub_BC87E0(v8);
          goto LABEL_136;
        }
LABEL_5:
        v4 = (_QWORD *)v4[1];
        if ( v100 == v4 )
          return v3;
      }
      v10 = *(_QWORD *)(v6 + 8);
LABEL_26:
      if ( (unsigned int)*(unsigned __int8 *)(v10 + 8) - 17 <= 1 )
        v10 = **(_QWORD **)(v10 + 16);
      if ( !sub_BCAC40(v10, 1) )
        goto LABEL_5;
      if ( *(_BYTE *)v6 == 58 )
      {
        if ( (*(_BYTE *)(v6 + 7) & 0x40) != 0 )
          v71 = *(__int64 **)(v6 - 8);
        else
          v71 = (__int64 *)(v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF));
        v15 = *v71;
        v72 = *(_QWORD *)(*v71 + 16);
        if ( !v72 )
          goto LABEL_5;
        if ( *(_QWORD *)(v72 + 8) )
          goto LABEL_5;
        v95 = (_QWORD *)v71[4];
        v17 = v95[2];
        if ( !v17 )
          goto LABEL_5;
      }
      else
      {
        if ( *(_BYTE *)v6 != 86 )
          goto LABEL_5;
        v89 = *(_QWORD *)(v6 - 96);
        if ( *(_QWORD *)(v89 + 8) != *(_QWORD *)(v6 + 8) )
          goto LABEL_5;
        v14 = *(_BYTE **)(v6 - 64);
        if ( *v14 > 0x15u )
          goto LABEL_5;
        v95 = *(_QWORD **)(v6 - 32);
        if ( !sub_AD7A80(v14, 1, v11, v12, v13) )
          goto LABEL_5;
        v15 = v89;
        v16 = *(_QWORD *)(v89 + 16);
        if ( !v16 )
          goto LABEL_5;
        if ( *(_QWORD *)(v16 + 8) )
          goto LABEL_5;
        v17 = v95[2];
        if ( !v17 )
          goto LABEL_5;
      }
      if ( *(_QWORD *)(v17 + 8) )
        goto LABEL_5;
      v86 = 29;
      goto LABEL_39;
    }
  }
  return v3;
}
