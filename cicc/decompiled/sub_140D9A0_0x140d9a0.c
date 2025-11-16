// Function: sub_140D9A0
// Address: 0x140d9a0
//
_QWORD *__fastcall sub_140D9A0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  char v7; // bl
  __int64 v8; // rax
  __int64 v9; // rbx
  _QWORD *v10; // rbx
  __int64 v11; // r12
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 v14; // rax
  unsigned __int64 v15; // r12
  _QWORD *v16; // r14
  __int64 v17; // rbx
  __int64 v18; // rsi
  unsigned int v19; // eax
  __int64 v20; // rcx
  unsigned __int64 v21; // r13
  __int64 v22; // rax
  unsigned __int64 v23; // r13
  _QWORD *v24; // rbx
  __int64 v25; // r13
  char v26; // al
  int v28; // eax
  int v29; // eax
  unsigned int v30; // eax
  __int64 v31; // rsi
  __int64 v32; // r10
  unsigned __int64 v33; // r9
  _QWORD *v34; // rax
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rax
  __int64 v38; // rax
  char *v39; // rdx
  __int64 v40; // rax
  char *v41; // rdx
  __int64 v42; // r13
  __int64 v43; // rax
  __int64 v44; // r13
  __int64 v45; // rax
  char *v46; // rdx
  char *v47; // rdx
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rdi
  unsigned __int64 v51; // rdi
  __int64 v52; // rax
  __int64 v53; // rsi
  _QWORD *v54; // rdx
  __int64 v55; // rsi
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rsi
  int v59; // eax
  _QWORD *v60; // rax
  unsigned int v61; // eax
  __int64 v62; // rdx
  __int64 v63; // rsi
  unsigned __int64 v64; // r11
  __int64 v65; // rax
  __int64 v66; // rax
  unsigned __int64 v67; // rax
  __int64 v68; // [rsp+0h] [rbp-120h]
  __int64 v69; // [rsp+8h] [rbp-118h]
  __int64 v70; // [rsp+8h] [rbp-118h]
  __int64 v71; // [rsp+10h] [rbp-110h]
  __int64 v72; // [rsp+10h] [rbp-110h]
  __int64 v73; // [rsp+10h] [rbp-110h]
  __int64 v74; // [rsp+18h] [rbp-108h]
  __int64 v75; // [rsp+18h] [rbp-108h]
  unsigned __int64 v76; // [rsp+18h] [rbp-108h]
  __int64 v77; // [rsp+20h] [rbp-100h]
  __int64 v78; // [rsp+20h] [rbp-100h]
  __int64 v79; // [rsp+20h] [rbp-100h]
  unsigned __int64 v80; // [rsp+20h] [rbp-100h]
  __int64 v81; // [rsp+20h] [rbp-100h]
  __int64 v82; // [rsp+28h] [rbp-F8h]
  unsigned __int64 v83; // [rsp+28h] [rbp-F8h]
  unsigned __int64 v84; // [rsp+28h] [rbp-F8h]
  __int64 v85; // [rsp+28h] [rbp-F8h]
  __int64 v86; // [rsp+28h] [rbp-F8h]
  __int64 v87; // [rsp+28h] [rbp-F8h]
  __int64 v88; // [rsp+30h] [rbp-F0h]
  __int64 v89; // [rsp+30h] [rbp-F0h]
  __int64 v90; // [rsp+30h] [rbp-F0h]
  __int64 v91; // [rsp+30h] [rbp-F0h]
  unsigned __int64 v93; // [rsp+40h] [rbp-E0h]
  __int64 v95; // [rsp+50h] [rbp-D0h]
  __int64 v96; // [rsp+58h] [rbp-C8h]
  _QWORD *v97; // [rsp+60h] [rbp-C0h]
  __int64 v99; // [rsp+70h] [rbp-B0h]
  __int64 v100; // [rsp+70h] [rbp-B0h]
  __int64 v101; // [rsp+70h] [rbp-B0h]
  __int64 v102; // [rsp+70h] [rbp-B0h]
  unsigned __int64 *v103; // [rsp+70h] [rbp-B0h]
  __int64 v104; // [rsp+78h] [rbp-A8h]
  unsigned __int8 v105; // [rsp+80h] [rbp-A0h]
  char v106; // [rsp+87h] [rbp-99h]
  _QWORD *v107; // [rsp+88h] [rbp-98h]
  __int64 v108; // [rsp+98h] [rbp-88h] BYREF
  _QWORD v109[2]; // [rsp+A0h] [rbp-80h] BYREF
  _QWORD *v110; // [rsp+B0h] [rbp-70h] BYREF
  char *v111; // [rsp+B8h] [rbp-68h]
  __int16 v112; // [rsp+C0h] [rbp-60h]
  _QWORD *v113; // [rsp+D0h] [rbp-50h] BYREF
  char *v114; // [rsp+D8h] [rbp-48h]
  __int16 v115; // [rsp+E0h] [rbp-40h]

  v7 = a4 ^ 1;
  v96 = sub_15A9650(a2, *(_QWORD *)a3, a3, a4, a5, a6);
  v97 = (_QWORD *)sub_15A06D0(v96);
  v105 = v7 & ((*(_BYTE *)(a3 + 17) & 2) != 0);
  v8 = v96;
  if ( *(_BYTE *)(v96 + 8) == 16 )
    v8 = **(_QWORD **)(v96 + 16);
  v93 = 0xFFFFFFFFFFFFFFFFLL >> (64 - BYTE1(*(_DWORD *)(v8 + 8)));
  if ( (*(_BYTE *)(a3 + 23) & 0x40) != 0 )
    v9 = *(_QWORD *)(a3 - 8);
  else
    v9 = a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF);
  v10 = (_QWORD *)(v9 + 24);
  v11 = sub_16348C0(a3) | 4;
  if ( (*(_BYTE *)(a3 + 23) & 0x40) != 0 )
  {
    v12 = *(_QWORD *)(a3 - 8);
    v95 = v12 + 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF);
  }
  else
  {
    v95 = a3;
    v12 = a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF);
  }
  if ( v95 != v12 + 24 )
  {
    v107 = v10;
    v13 = v12 + 48;
    do
    {
      v14 = v11;
      v15 = v11 & 0xFFFFFFFFFFFFFFF8LL;
      v104 = v13;
      v16 = *(_QWORD **)(v13 - 24);
      v17 = v15;
      LODWORD(v14) = (v14 >> 2) & 1;
      v106 = v14;
      if ( !(_DWORD)v14 || (v18 = v15) == 0 )
        v18 = sub_1643D30(v15, *v107);
      v19 = sub_15A9FE0(a2, v18);
      v20 = 1;
      v21 = v19;
      while ( 2 )
      {
        switch ( *(_BYTE *)(v18 + 8) )
        {
          case 0:
          case 8:
          case 0xA:
          case 0xC:
          case 0x10:
            v35 = *(_QWORD *)(v18 + 32);
            v18 = *(_QWORD *)(v18 + 24);
            v20 *= v35;
            continue;
          case 1:
            v22 = 16;
            break;
          case 2:
            v22 = 32;
            break;
          case 3:
          case 9:
            v22 = 64;
            break;
          case 4:
            v22 = 80;
            break;
          case 5:
          case 6:
            v22 = 128;
            break;
          case 7:
            v99 = v20;
            v28 = sub_15A9520(a2, 0);
            v20 = v99;
            v22 = (unsigned int)(8 * v28);
            break;
          case 0xB:
            v22 = *(_DWORD *)(v18 + 8) >> 8;
            break;
          case 0xD:
            v102 = v20;
            v34 = (_QWORD *)sub_15A9930(a2, v18);
            v20 = v102;
            v22 = 8LL * *v34;
            break;
          case 0xE:
            v82 = v20;
            v101 = *(_QWORD *)(v18 + 32);
            v88 = *(_QWORD *)(v18 + 24);
            v30 = sub_15A9FE0(a2, v88);
            v31 = v88;
            v20 = v82;
            v32 = 1;
            v33 = v30;
            while ( 2 )
            {
              switch ( *(_BYTE *)(v31 + 8) )
              {
                case 0:
                case 8:
                case 0xA:
                case 0xC:
                case 0x10:
                  v57 = *(_QWORD *)(v31 + 32);
                  v31 = *(_QWORD *)(v31 + 24);
                  v32 *= v57;
                  continue;
                case 1:
                  v56 = 16;
                  goto LABEL_65;
                case 2:
                  v56 = 32;
                  goto LABEL_65;
                case 3:
                case 9:
                  v56 = 64;
                  goto LABEL_65;
                case 4:
                  v56 = 80;
                  goto LABEL_65;
                case 5:
                case 6:
                  v56 = 128;
                  goto LABEL_65;
                case 7:
                  v77 = v82;
                  v58 = 0;
                  v83 = v33;
                  v89 = v32;
                  goto LABEL_71;
                case 0xB:
                  v56 = *(_DWORD *)(v31 + 8) >> 8;
                  goto LABEL_65;
                case 0xD:
                  v78 = v82;
                  v84 = v33;
                  v90 = v32;
                  v60 = (_QWORD *)sub_15A9930(a2, v31);
                  v32 = v90;
                  v33 = v84;
                  v20 = v78;
                  v56 = 8LL * *v60;
                  goto LABEL_65;
                case 0xE:
                  v71 = v82;
                  v74 = v33;
                  v79 = v32;
                  v85 = *(_QWORD *)(v31 + 24);
                  v91 = *(_QWORD *)(v31 + 32);
                  v61 = sub_15A9FE0(a2, v85);
                  v20 = v71;
                  v33 = v74;
                  v62 = 1;
                  v63 = v85;
                  v32 = v79;
                  v64 = v61;
                  while ( 2 )
                  {
                    switch ( *(_BYTE *)(v63 + 8) )
                    {
                      case 0:
                      case 8:
                      case 0xA:
                      case 0xC:
                      case 0x10:
                        v66 = *(_QWORD *)(v63 + 32);
                        v63 = *(_QWORD *)(v63 + 24);
                        v62 *= v66;
                        continue;
                      case 1:
                        v65 = 16;
                        goto LABEL_79;
                      case 2:
                        v65 = 32;
                        goto LABEL_79;
                      case 3:
                      case 9:
                        v65 = 64;
                        goto LABEL_79;
                      case 4:
                        v65 = 80;
                        goto LABEL_79;
                      case 5:
                      case 6:
                        v65 = 128;
                        goto LABEL_79;
                      case 7:
                        JUMPOUT(0x140E470);
                      case 0xB:
                        v65 = *(_DWORD *)(v63 + 8) >> 8;
                        goto LABEL_79;
                      case 0xD:
                        v69 = v71;
                        v72 = v74;
                        v75 = v79;
                        v80 = v64;
                        v86 = v62;
                        v65 = 8LL * *(_QWORD *)sub_15A9930(a2, v63);
                        goto LABEL_85;
                      case 0xE:
                        v68 = v71;
                        v70 = v74;
                        v73 = v79;
                        v76 = v64;
                        v81 = v62;
                        v87 = *(_QWORD *)(v63 + 32);
                        v67 = sub_12BE0A0(a2, *(_QWORD *)(v63 + 24));
                        v62 = v81;
                        v64 = v76;
                        v32 = v73;
                        v33 = v70;
                        v20 = v68;
                        v65 = 8 * v87 * v67;
                        goto LABEL_79;
                      case 0xF:
                        v69 = v71;
                        v72 = v74;
                        v75 = v79;
                        v80 = v64;
                        v86 = v62;
                        v65 = 8 * (unsigned int)sub_15A9520(a2, *(_DWORD *)(v63 + 8) >> 8);
LABEL_85:
                        v62 = v86;
                        v64 = v80;
                        v32 = v75;
                        v33 = v72;
                        v20 = v69;
LABEL_79:
                        v56 = 8 * v91 * v64 * ((v64 + ((unsigned __int64)(v65 * v62 + 7) >> 3) - 1) / v64);
                        break;
                    }
                    goto LABEL_65;
                  }
                case 0xF:
                  v77 = v82;
                  v83 = v33;
                  v89 = v32;
                  v58 = *(_DWORD *)(v31 + 8) >> 8;
LABEL_71:
                  v59 = sub_15A9520(a2, v58);
                  v32 = v89;
                  v33 = v83;
                  v20 = v77;
                  v56 = (unsigned int)(8 * v59);
LABEL_65:
                  v22 = 8 * v33 * v101 * ((v33 + ((unsigned __int64)(v56 * v32 + 7) >> 3) - 1) / v33);
                  break;
              }
              break;
            }
            break;
          case 0xF:
            v100 = v20;
            v29 = sub_15A9520(a2, *(_DWORD *)(v18 + 8) >> 8);
            v20 = v100;
            v22 = (unsigned int)(8 * v29);
            break;
        }
        break;
      }
      v23 = v93 & (v21 * ((v21 + ((unsigned __int64)(v22 * v20 + 7) >> 3) - 1) / v21));
      if ( *((_BYTE *)v16 + 16) > 0x10u )
      {
        if ( v96 != *v16 )
        {
          v109[0] = sub_1649960(v16);
          v110 = v109;
          v111 = ".c";
          v109[1] = v36;
          v112 = 773;
          if ( v96 != *v16 )
          {
            if ( *((_BYTE *)v16 + 16) > 0x10u )
            {
              v115 = 257;
              v49 = sub_15FE0A0(v16, v96, 1, &v113, 0);
              v16 = (_QWORD *)v49;
              v50 = a1[1];
              if ( v50 )
              {
                v103 = (unsigned __int64 *)a1[2];
                sub_157E9D0(v50 + 40, v49);
                v51 = *v103;
                v52 = v16[3] & 7LL;
                v16[4] = v103;
                v51 &= 0xFFFFFFFFFFFFFFF8LL;
                v16[3] = v51 | v52;
                *(_QWORD *)(v51 + 8) = v16 + 3;
                *v103 = *v103 & 7 | (unsigned __int64)(v16 + 3);
              }
              sub_164B780(v16, &v110);
              v53 = *a1;
              if ( *a1 )
              {
                v108 = *a1;
                sub_1623A60(&v108, v53, 2);
                v54 = v16 + 6;
                if ( v16[6] )
                {
                  sub_161E7C0(v16 + 6);
                  v54 = v16 + 6;
                }
                v55 = v108;
                v16[6] = v108;
                if ( v55 )
                  sub_1623210(&v108, v55, v54);
              }
            }
            else
            {
              v16 = (_QWORD *)sub_15A4750(v16, v96, 1);
              v37 = sub_14DBA30(v16, a1[8], 0);
              if ( v37 )
                v16 = (_QWORD *)v37;
            }
          }
        }
        if ( v23 != 1 )
        {
          v38 = sub_1649960(a3);
          v111 = v39;
          v110 = (_QWORD *)v38;
          v115 = 773;
          v113 = &v110;
          v114 = ".idx";
          v40 = sub_15A0680(v96, v23, 0);
          v16 = sub_140D0B0(a1, (__int64)v16, v40, (__int64)&v113, v105 & 1, 0);
        }
        v110 = (_QWORD *)sub_1649960(a3);
        v111 = v41;
        v115 = 773;
        v113 = &v110;
        v114 = ".offs";
        v97 = sub_140D830(a1, (__int64)v16, (__int64)v97, (__int64)&v113, 0, 0);
      }
      else if ( !(unsigned __int8)sub_1595F50(v16) )
      {
        if ( !v106 && v15 )
        {
          if ( *(_BYTE *)(*v16 + 8LL) == 16 )
            v16 = (_QWORD *)sub_15A1020(v16);
          v24 = (_QWORD *)v16[3];
          if ( *((_DWORD *)v16 + 8) > 0x40u )
            v24 = (_QWORD *)*v24;
          v25 = *(_QWORD *)(sub_15A9930(a2, v15) + 8LL * (unsigned int)v24 + 16);
          if ( v25 )
          {
            v110 = (_QWORD *)sub_1649960(a3);
            v111 = v47;
            v115 = 773;
            v113 = &v110;
            v114 = ".offs";
            v48 = sub_15A0680(v96, v25, 0);
            v97 = sub_140D830(a1, (__int64)v97, v48, (__int64)&v113, 0, 0);
          }
LABEL_24:
          v17 = sub_1643D30(v15, *v107);
          v26 = *(_BYTE *)(v17 + 8);
          if ( ((v26 - 14) & 0xFD) != 0 )
            goto LABEL_50;
          goto LABEL_25;
        }
        v42 = sub_15A0680(v96, v23, 0);
        v43 = sub_15A4750(v16, v96, 1);
        v44 = sub_15A2C20(v43, v42, v105, 0);
        v45 = sub_1649960(a3);
        v115 = 773;
        v110 = (_QWORD *)v45;
        v111 = v46;
        v113 = &v110;
        v114 = ".offs";
        v97 = sub_140D830(a1, (__int64)v97, v44, (__int64)&v113, 0, 0);
      }
      if ( !v106 || !v15 )
        goto LABEL_24;
      v26 = *(_BYTE *)(v15 + 8);
      if ( ((v26 - 14) & 0xFD) != 0 )
      {
LABEL_50:
        v11 = 0;
        if ( v26 == 13 )
          v11 = v17;
        goto LABEL_26;
      }
LABEL_25:
      v11 = *(_QWORD *)(v17 + 24) | 4LL;
LABEL_26:
      v107 += 3;
      v13 += 24;
    }
    while ( v104 != v95 );
  }
  return v97;
}
