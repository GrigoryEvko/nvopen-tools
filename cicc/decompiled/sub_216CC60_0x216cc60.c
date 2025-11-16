// Function: sub_216CC60
// Address: 0x216cc60
//
__int64 __fastcall sub_216CC60(__int64 *a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r14
  __int64 v8; // r8
  __int64 v9; // r13
  unsigned __int64 v10; // r8
  __int64 v11; // r15
  __int64 v12; // r12
  __int64 v13; // rbx
  __int64 v14; // r13
  __int64 v15; // r8
  __int64 v16; // rsi
  unsigned int v17; // eax
  __int64 v18; // rcx
  __int64 v19; // r8
  unsigned __int64 v20; // r10
  __int64 v21; // rax
  unsigned __int64 v22; // r10
  unsigned int v23; // eax
  char v24; // al
  __int64 v25; // rax
  unsigned int v26; // ebx
  unsigned int v27; // ebx
  __int64 v28; // rax
  __int64 v29; // rdi
  __int64 v30; // rsi
  unsigned __int8 v31; // r12
  unsigned int v32; // r12d
  __int64 v34; // rax
  __int64 v35; // rsi
  __int64 v36; // rax
  __int64 v37; // r10
  __int64 v38; // rax
  __int64 v39; // r10
  __int64 v40; // rax
  _QWORD *v41; // r13
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  unsigned int v45; // esi
  int v46; // eax
  __int64 v47; // rdi
  unsigned int v48; // eax
  __int64 v49; // rsi
  __int64 v50; // r9
  __int64 v51; // r8
  unsigned __int64 v52; // r11
  __int64 v53; // rdi
  _QWORD *v54; // rax
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // rdi
  _QWORD *v58; // rax
  unsigned int v59; // eax
  __int64 v60; // rsi
  __int64 v61; // r8
  unsigned int v62; // esi
  int v63; // eax
  __int64 v64; // rax
  unsigned int v65; // esi
  int v66; // eax
  __int64 v67; // rdi
  __int64 v68; // rax
  _QWORD *v69; // rax
  __int64 v70; // rax
  __int64 v71; // rax
  __int64 v72; // [rsp+0h] [rbp-110h]
  __int64 v73; // [rsp+8h] [rbp-108h]
  __int64 v74; // [rsp+10h] [rbp-100h]
  __int64 v75; // [rsp+10h] [rbp-100h]
  __int64 v76; // [rsp+18h] [rbp-F8h]
  __int64 v77; // [rsp+18h] [rbp-F8h]
  __int64 v78; // [rsp+20h] [rbp-F0h]
  __int64 v79; // [rsp+20h] [rbp-F0h]
  __int64 v80; // [rsp+20h] [rbp-F0h]
  __int64 v81; // [rsp+28h] [rbp-E8h]
  __int64 v82; // [rsp+28h] [rbp-E8h]
  __int64 v83; // [rsp+28h] [rbp-E8h]
  __int64 v84; // [rsp+30h] [rbp-E0h]
  unsigned __int64 v85; // [rsp+30h] [rbp-E0h]
  unsigned __int64 v86; // [rsp+30h] [rbp-E0h]
  __int64 v87; // [rsp+38h] [rbp-D8h]
  __int64 v88; // [rsp+38h] [rbp-D8h]
  __int64 v89; // [rsp+38h] [rbp-D8h]
  __int64 v90; // [rsp+38h] [rbp-D8h]
  __int64 v91; // [rsp+40h] [rbp-D0h]
  unsigned __int64 v92; // [rsp+40h] [rbp-D0h]
  __int64 v93; // [rsp+48h] [rbp-C8h]
  unsigned __int64 v94; // [rsp+48h] [rbp-C8h]
  __int64 v95; // [rsp+48h] [rbp-C8h]
  unsigned __int64 v96; // [rsp+48h] [rbp-C8h]
  __int64 v97; // [rsp+50h] [rbp-C0h]
  char v99; // [rsp+63h] [rbp-ADh]
  unsigned int v100; // [rsp+64h] [rbp-ACh]
  unsigned __int64 v101; // [rsp+68h] [rbp-A8h]
  __int64 v102; // [rsp+70h] [rbp-A0h]
  unsigned __int64 v103; // [rsp+78h] [rbp-98h]
  unsigned __int64 v104; // [rsp+78h] [rbp-98h]
  __int64 v105; // [rsp+78h] [rbp-98h]
  unsigned __int64 v106; // [rsp+78h] [rbp-98h]
  __int64 v107; // [rsp+78h] [rbp-98h]
  __int64 v108; // [rsp+78h] [rbp-98h]
  __int64 v109; // [rsp+78h] [rbp-98h]
  __int64 v110; // [rsp+80h] [rbp-90h]
  __int64 v111; // [rsp+80h] [rbp-90h]
  unsigned __int64 v112; // [rsp+80h] [rbp-90h]
  __int64 v113; // [rsp+80h] [rbp-90h]
  __int64 v114; // [rsp+80h] [rbp-90h]
  __int64 v115; // [rsp+80h] [rbp-90h]
  __int64 v116; // [rsp+80h] [rbp-90h]
  __int64 v117; // [rsp+88h] [rbp-88h]
  __int64 v119; // [rsp+98h] [rbp-78h]
  __int64 v120; // [rsp+A0h] [rbp-70h] BYREF
  unsigned int v121; // [rsp+A8h] [rbp-68h]
  __int64 v122; // [rsp+B0h] [rbp-60h] BYREF
  unsigned int v123; // [rsp+B8h] [rbp-58h]
  __int64 v124; // [rsp+C0h] [rbp-50h] BYREF
  __int64 v125; // [rsp+C8h] [rbp-48h]
  char v126; // [rsp+D0h] [rbp-40h]
  unsigned __int64 v127; // [rsp+D8h] [rbp-38h]

  if ( a3 && (v97 = sub_1649C60((__int64)a3), *(_BYTE *)(v97 + 16) <= 3u) )
  {
    v99 = 0;
  }
  else
  {
    v99 = 1;
    v97 = 0;
  }
  v100 = sub_15A9570(*a1, *a3);
  v121 = v100;
  if ( v100 > 0x40 )
  {
    sub_16A4EF0((__int64)&v120, 0, 0);
    if ( a5 )
      goto LABEL_7;
LABEL_79:
    v32 = v97 != 0;
    goto LABEL_37;
  }
  v120 = 0;
  if ( !a5 )
    goto LABEL_79;
LABEL_7:
  v117 = a4 + 8 * a5;
  if ( a4 != v117 )
  {
    v101 = 0;
    v7 = a4 + 8;
    v8 = a2 | 4;
    while ( 1 )
    {
      v9 = v8;
      v10 = v8 & 0xFFFFFFFFFFFFFFF8LL;
      v119 = v7;
      v11 = *(_QWORD *)(v7 - 8);
      v12 = v10;
      v13 = v10;
      v14 = (v9 >> 2) & 1;
      if ( !(_DWORD)v14 )
      {
        v112 = v10;
        v40 = sub_1643D30(v10, *(_QWORD *)(v7 - 8));
        v11 = *(_QWORD *)(v7 - 8);
        v37 = v112;
        v102 = v40;
        if ( *(_BYTE *)(v11 + 16) != 13 )
        {
          v71 = sub_14C49D0((_BYTE *)v11);
          v11 = v71;
          if ( v71 && *(_BYTE *)(v71 + 16) != 13 )
            v11 = 0;
          v37 = v12;
        }
        v15 = *a1;
        if ( v12 )
        {
          v41 = *(_QWORD **)(v11 + 24);
          if ( *(_DWORD *)(v11 + 32) > 0x40u )
            v41 = (_QWORD *)*v41;
          v113 = v37;
          v42 = sub_15A9930(*a1, v12);
          sub_16A7490((__int64)&v120, *(_QWORD *)(v42 + 8LL * (unsigned int)v41 + 16));
          v39 = v113;
          goto LABEL_53;
        }
        v35 = *(_QWORD *)(v7 - 8);
        goto LABEL_51;
      }
      if ( v10 )
      {
        if ( *(_BYTE *)(v11 + 16) == 13
          || (v43 = sub_14C49D0(*(_BYTE **)(v7 - 8)), (v11 = v43) == 0)
          || *(_BYTE *)(v43 + 16) == 13 )
        {
          v102 = v12;
          v15 = *a1;
          goto LABEL_13;
        }
        v102 = v12;
      }
      else
      {
        v34 = sub_1643D30(0, *(_QWORD *)(v7 - 8));
        v11 = *(_QWORD *)(v7 - 8);
        v102 = v34;
        v35 = v11;
        if ( *(_BYTE *)(v11 + 16) == 13 )
        {
          v15 = *a1;
          goto LABEL_50;
        }
        v36 = sub_14C49D0((_BYTE *)v11);
        v11 = v36;
        if ( !v36 )
        {
          v15 = *a1;
LABEL_49:
          v35 = *(_QWORD *)(v7 - 8);
LABEL_50:
          v37 = 0;
LABEL_51:
          v111 = v15;
          v38 = sub_1643D30(v37, v35);
          v15 = v111;
          v16 = v38;
          goto LABEL_14;
        }
        if ( *(_BYTE *)(v36 + 16) == 13 )
          goto LABEL_48;
      }
      v11 = 0;
LABEL_48:
      v15 = *a1;
      if ( !v12 )
        goto LABEL_49;
LABEL_13:
      v16 = v12;
LABEL_14:
      v110 = v15;
      v17 = sub_15A9FE0(v15, v16);
      v18 = 1;
      v19 = v110;
      v20 = v17;
      while ( 2 )
      {
        switch ( *(_BYTE *)(v16 + 8) )
        {
          case 0:
          case 8:
          case 0xA:
          case 0xC:
          case 0x10:
            v44 = *(_QWORD *)(v16 + 32);
            v16 = *(_QWORD *)(v16 + 24);
            v18 *= v44;
            continue;
          case 1:
            v21 = 16;
            goto LABEL_17;
          case 2:
            v21 = 32;
            goto LABEL_17;
          case 3:
          case 9:
            v21 = 64;
            goto LABEL_17;
          case 4:
            v21 = 80;
            goto LABEL_17;
          case 5:
          case 6:
            v21 = 128;
            goto LABEL_17;
          case 7:
            v104 = v20;
            v45 = 0;
            v114 = v18;
            goto LABEL_70;
          case 0xB:
            v21 = *(_DWORD *)(v16 + 8) >> 8;
            goto LABEL_17;
          case 0xD:
            v53 = v110;
            v106 = v20;
            v116 = v18;
            v54 = (_QWORD *)sub_15A9930(v53, v16);
            v18 = v116;
            v20 = v106;
            v21 = 8LL * *v54;
            goto LABEL_17;
          case 0xE:
            v47 = v110;
            v87 = v20;
            v91 = v18;
            v93 = *(_QWORD *)(v16 + 24);
            v105 = v110;
            v115 = *(_QWORD *)(v16 + 32);
            v48 = sub_15A9FE0(v47, v93);
            v20 = v87;
            v49 = v93;
            v50 = 1;
            v18 = v91;
            v51 = v105;
            v52 = v48;
            while ( 2 )
            {
              switch ( *(_BYTE *)(v49 + 8) )
              {
                case 0:
                case 8:
                case 0xA:
                case 0xC:
                case 0x10:
                  v56 = *(_QWORD *)(v49 + 32);
                  v49 = *(_QWORD *)(v49 + 24);
                  v50 *= v56;
                  continue;
                case 1:
                  v55 = 16;
                  goto LABEL_82;
                case 2:
                  v55 = 32;
                  goto LABEL_82;
                case 3:
                case 9:
                  v55 = 64;
                  goto LABEL_82;
                case 4:
                  v55 = 80;
                  goto LABEL_82;
                case 5:
                case 6:
                  v55 = 128;
                  goto LABEL_82;
                case 7:
                  v62 = 0;
                  v96 = v52;
                  v109 = v50;
                  goto LABEL_92;
                case 0xB:
                  v55 = *(_DWORD *)(v49 + 8) >> 8;
                  goto LABEL_82;
                case 0xD:
                  v57 = v105;
                  v94 = v52;
                  v107 = v50;
                  v58 = (_QWORD *)sub_15A9930(v57, v49);
                  v50 = v107;
                  v52 = v94;
                  v18 = v91;
                  v20 = v87;
                  v55 = 8LL * *v58;
                  goto LABEL_82;
                case 0xE:
                  v74 = v87;
                  v76 = v91;
                  v78 = v52;
                  v81 = v50;
                  v84 = *(_QWORD *)(v49 + 24);
                  v88 = v105;
                  v95 = *(_QWORD *)(v49 + 32);
                  v59 = sub_15A9FE0(v105, v84);
                  v20 = v74;
                  v108 = 1;
                  v18 = v91;
                  v52 = v78;
                  v92 = v59;
                  v60 = v84;
                  v50 = v81;
                  v61 = v88;
                  while ( 2 )
                  {
                    switch ( *(_BYTE *)(v60 + 8) )
                    {
                      case 0:
                      case 8:
                      case 0xA:
                      case 0xC:
                      case 0x10:
                        v70 = v108 * *(_QWORD *)(v60 + 32);
                        v60 = *(_QWORD *)(v60 + 24);
                        v108 = v70;
                        continue;
                      case 1:
                        v64 = 16;
                        goto LABEL_97;
                      case 2:
                        v64 = 32;
                        goto LABEL_97;
                      case 3:
                      case 9:
                        v64 = 64;
                        goto LABEL_97;
                      case 4:
                        v64 = 80;
                        goto LABEL_97;
                      case 5:
                      case 6:
                        v64 = 128;
                        goto LABEL_97;
                      case 7:
                        v79 = v74;
                        v65 = 0;
                        v82 = v76;
                        v85 = v52;
                        v89 = v50;
                        goto LABEL_101;
                      case 0xB:
                        v64 = *(_DWORD *)(v60 + 8) >> 8;
                        goto LABEL_97;
                      case 0xD:
                        v69 = (_QWORD *)sub_15A9930(v88, v60);
                        v50 = v81;
                        v52 = v78;
                        v18 = v76;
                        v20 = v74;
                        v64 = 8LL * *v69;
                        goto LABEL_97;
                      case 0xE:
                        v67 = v88;
                        v72 = v74;
                        v75 = v78;
                        v77 = v81;
                        v90 = *(_QWORD *)(v60 + 32);
                        v73 = v18;
                        v80 = *(_QWORD *)(v60 + 24);
                        v83 = v61;
                        v86 = (unsigned int)sub_15A9FE0(v67, v80);
                        v68 = sub_127FA20(v83, v80);
                        v50 = v77;
                        v52 = v75;
                        v20 = v72;
                        v18 = v73;
                        v64 = 8 * v86 * v90 * ((v86 + ((unsigned __int64)(v68 + 7) >> 3) - 1) / v86);
                        goto LABEL_97;
                      case 0xF:
                        v79 = v74;
                        v82 = v76;
                        v85 = v52;
                        v65 = *(_DWORD *)(v60 + 8) >> 8;
                        v89 = v50;
LABEL_101:
                        v66 = sub_15A9520(v61, v65);
                        v50 = v89;
                        v52 = v85;
                        v18 = v82;
                        v20 = v79;
                        v64 = (unsigned int)(8 * v66);
LABEL_97:
                        v55 = 8 * v92 * v95 * ((v92 + ((unsigned __int64)(v108 * v64 + 7) >> 3) - 1) / v92);
                        break;
                    }
                    goto LABEL_82;
                  }
                case 0xF:
                  v96 = v52;
                  v62 = *(_DWORD *)(v49 + 8) >> 8;
                  v109 = v50;
LABEL_92:
                  v63 = sub_15A9520(v51, v62);
                  v50 = v109;
                  v52 = v96;
                  v18 = v91;
                  v20 = v87;
                  v55 = (unsigned int)(8 * v63);
LABEL_82:
                  v21 = 8 * v115 * v52 * ((v52 + ((unsigned __int64)(v50 * v55 + 7) >> 3) - 1) / v52);
                  break;
              }
              goto LABEL_17;
            }
          case 0xF:
            v104 = v20;
            v114 = v18;
            v45 = *(_DWORD *)(v16 + 8) >> 8;
LABEL_70:
            v46 = sub_15A9520(v19, v45);
            v18 = v114;
            v20 = v104;
            v21 = (unsigned int)(8 * v46);
LABEL_17:
            v22 = (v20 + ((unsigned __int64)(v21 * v18 + 7) >> 3) - 1) / v20 * v20;
            if ( v11 )
            {
              v103 = v22;
              sub_16A5D70((__int64)&v122, (__int64 *)(v11 + 24), v100);
              sub_16A7A10((__int64)&v122, v103);
              v23 = v123;
              v123 = 0;
              LODWORD(v125) = v23;
              v124 = v122;
              sub_16A7200((__int64)&v120, &v124);
              if ( (unsigned int)v125 > 0x40 && v124 )
                j_j___libc_free_0_0(v124);
              if ( v123 > 0x40 && v122 )
                j_j___libc_free_0_0(v122);
            }
            else
            {
              if ( v101 )
              {
                v32 = 1;
                goto LABEL_37;
              }
              v101 = v22;
            }
            if ( !(_BYTE)v14 )
            {
              v39 = v12;
              goto LABEL_53;
            }
            if ( !v12 )
            {
              v39 = 0;
LABEL_53:
              v13 = sub_1643D30(v39, *(_QWORD *)(v7 - 8));
            }
            v24 = *(_BYTE *)(v13 + 8);
            if ( ((v24 - 14) & 0xFD) != 0 )
            {
              v8 = 0;
              if ( v24 == 13 )
                v8 = v13;
            }
            else
            {
              v8 = *(_QWORD *)(v13 + 24) | 4LL;
            }
            v7 += 8;
            if ( v117 == v119 )
              goto LABEL_29;
            break;
        }
        break;
      }
    }
  }
  v102 = 0;
  v101 = 0;
LABEL_29:
  v25 = *a3;
  if ( *(_BYTE *)(*a3 + 8) == 16 )
    v25 = **(_QWORD **)(v25 + 16);
  v26 = *(_DWORD *)(v25 + 8);
  sub_16A5D70((__int64)&v122, &v120, 0x40u);
  v27 = v26 >> 8;
  if ( v123 > 0x40 )
    v28 = *(_QWORD *)v122;
  else
    v28 = v122 << (64 - (unsigned __int8)v123) >> (64 - (unsigned __int8)v123);
  v125 = v28;
  v29 = a1[2];
  v30 = *a1;
  v126 = v99;
  v124 = v97;
  v127 = v101;
  v31 = (*(__int64 (__fastcall **)(__int64, __int64, __int64 *, __int64, _QWORD, _QWORD))(*(_QWORD *)v29 + 736LL))(
          v29,
          v30,
          &v124,
          v102,
          v27,
          0);
  if ( v123 > 0x40 && v122 )
    j_j___libc_free_0_0(v122);
  v32 = v31 ^ 1;
LABEL_37:
  if ( v121 > 0x40 && v120 )
    j_j___libc_free_0_0(v120);
  return v32;
}
