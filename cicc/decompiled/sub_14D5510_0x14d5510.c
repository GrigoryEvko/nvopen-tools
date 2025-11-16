// Function: sub_14D5510
// Address: 0x14d5510
//
__int64 __fastcall sub_14D5510(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, _BYTE *a5, __int64 a6)
{
  unsigned __int8 v7; // al
  _QWORD *v8; // r12
  unsigned int v10; // ebx
  unsigned int v12; // esi
  unsigned __int64 v13; // r9
  unsigned int v14; // esi
  __int64 v15; // rdi
  unsigned __int64 v16; // r12
  char v17; // si
  __int64 v18; // r13
  char v19; // cl
  char v21; // al
  unsigned int v22; // eax
  unsigned __int64 v23; // r9
  _DWORD *v24; // r13
  __int64 v25; // r8
  unsigned int v26; // r14d
  unsigned __int64 v27; // r12
  unsigned int v28; // eax
  __int64 v29; // rsi
  __int64 v30; // r8
  __int64 v31; // rcx
  unsigned __int64 v32; // r11
  __int64 v33; // rax
  __int64 v34; // rax
  __int64 v35; // rsi
  __int64 v36; // rax
  unsigned __int64 v37; // rdx
  int v38; // r9d
  unsigned int v39; // eax
  __int64 v40; // rsi
  __int64 v41; // r8
  unsigned __int64 v42; // rcx
  __int64 v43; // rax
  __int64 v44; // rsi
  int v45; // eax
  unsigned int v46; // eax
  __int64 v47; // rsi
  __int64 v48; // r10
  unsigned __int64 v49; // r9
  _QWORD *v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  unsigned __int64 v55; // rax
  unsigned __int64 v56; // rtt
  unsigned __int64 v57; // r14
  unsigned __int64 v58; // r9
  unsigned int v59; // r12d
  unsigned __int64 v60; // rbx
  __int64 v61; // rax
  unsigned __int64 v62; // rax
  __int64 v63; // rax
  __int64 v64; // rsi
  __int64 v65; // rax
  __int64 v66; // rax
  __int64 v67; // rsi
  int v68; // eax
  __int64 v69; // rax
  _QWORD *v70; // rax
  __int64 v71; // rax
  __int64 v72; // [rsp+0h] [rbp-90h]
  __int64 v73; // [rsp+8h] [rbp-88h]
  unsigned __int64 v74; // [rsp+10h] [rbp-80h]
  __int64 v75; // [rsp+18h] [rbp-78h]
  __int64 v76; // [rsp+18h] [rbp-78h]
  __int64 v77; // [rsp+20h] [rbp-70h]
  __int64 v78; // [rsp+20h] [rbp-70h]
  __int64 v79; // [rsp+20h] [rbp-70h]
  __int64 v80; // [rsp+28h] [rbp-68h]
  unsigned __int64 v81; // [rsp+28h] [rbp-68h]
  __int64 v82; // [rsp+28h] [rbp-68h]
  unsigned __int64 v83; // [rsp+30h] [rbp-60h]
  __int64 v84; // [rsp+30h] [rbp-60h]
  unsigned __int64 v85; // [rsp+30h] [rbp-60h]
  __int64 v86; // [rsp+30h] [rbp-60h]
  unsigned __int64 v87; // [rsp+30h] [rbp-60h]
  __int64 v88; // [rsp+38h] [rbp-58h]
  unsigned __int64 v89; // [rsp+38h] [rbp-58h]
  __int64 v90; // [rsp+40h] [rbp-50h]
  __int64 v91; // [rsp+40h] [rbp-50h]
  __int64 v92; // [rsp+48h] [rbp-48h]
  unsigned __int64 v93; // [rsp+48h] [rbp-48h]
  __int64 v94; // [rsp+48h] [rbp-48h]
  __int64 v95; // [rsp+50h] [rbp-40h]
  __int64 v96; // [rsp+50h] [rbp-40h]
  __int64 v97; // [rsp+50h] [rbp-40h]
  __int64 v98; // [rsp+50h] [rbp-40h]
  _QWORD *v99; // [rsp+50h] [rbp-40h]
  __int64 v100; // [rsp+50h] [rbp-40h]
  __int64 v101; // [rsp+50h] [rbp-40h]
  unsigned __int64 v102; // [rsp+50h] [rbp-40h]
  unsigned __int64 v103; // [rsp+50h] [rbp-40h]
  unsigned __int64 v104; // [rsp+50h] [rbp-40h]
  __int64 v105; // [rsp+58h] [rbp-38h]
  __int64 v106; // [rsp+58h] [rbp-38h]
  __int64 v107; // [rsp+58h] [rbp-38h]
  __int64 v108; // [rsp+58h] [rbp-38h]
  __int64 v109; // [rsp+58h] [rbp-38h]
  __int64 v110; // [rsp+58h] [rbp-38h]
  __int64 v111; // [rsp+58h] [rbp-38h]
  __int64 v112; // [rsp+58h] [rbp-38h]
  __int64 v113; // [rsp+58h] [rbp-38h]
  __int64 v114; // [rsp+58h] [rbp-38h]
  __int64 v115; // [rsp+58h] [rbp-38h]

  v7 = *(_BYTE *)(a1 + 16);
  if ( (unsigned __int8)(v7 - 9) > 1u )
  {
    v8 = (_QWORD *)a1;
    v10 = a4;
    while ( v7 != 13 )
    {
      if ( v7 == 14 )
      {
        v21 = *(_BYTE *)(*v8 + 8LL);
        switch ( v21 )
        {
          case 3:
            v51 = sub_16498A0(v8);
            v34 = sub_1643360(v51);
            break;
          case 2:
            v52 = sub_16498A0(v8);
            v34 = sub_1643350(v52);
            break;
          case 1:
            v33 = sub_16498A0(v8);
            v34 = sub_1643340(v33);
            break;
          default:
            return 0;
        }
        v106 = v34;
        v35 = v34;
        if ( !(unsigned __int8)sub_1593BB0(v8) || *(_BYTE *)(v106 + 8) == 9 )
          v8 = (_QWORD *)sub_14D44C0((__int64)v8, v35, a5);
        else
          v8 = (_QWORD *)sub_15A06D0(v106);
      }
      else
      {
        if ( v7 == 7 )
        {
          v88 = sub_15A9930(a5, *v8);
          v22 = sub_15A8020(v88, a2);
          v23 = a2;
          v90 = a3;
          v24 = v8;
          v25 = v22;
          v26 = v22;
          v92 = *(_QWORD *)(v88 + 8LL * v22 + 16);
          v27 = v23 - v92;
          while ( 1 )
          {
            v95 = v25;
            v105 = **(_QWORD **)&v24[6 * (v25 - (v24[5] & 0xFFFFFFF))];
            v28 = sub_15A9FE0(a5, v105);
            v29 = v105;
            v30 = v95;
            v31 = 1;
            v32 = v28;
            while ( 2 )
            {
              switch ( *(_BYTE *)(v29 + 8) )
              {
                case 0:
                case 8:
                case 0xA:
                case 0xC:
                case 0x10:
                  v43 = *(_QWORD *)(v29 + 32);
                  v29 = *(_QWORD *)(v29 + 24);
                  v31 *= v43;
                  continue;
                case 1:
                  v36 = 16;
                  goto LABEL_31;
                case 2:
                  v36 = 32;
                  goto LABEL_31;
                case 3:
                case 9:
                  v36 = 64;
                  goto LABEL_31;
                case 4:
                  v36 = 80;
                  goto LABEL_31;
                case 5:
                case 6:
                  v36 = 128;
                  goto LABEL_31;
                case 7:
                  v83 = v32;
                  v44 = 0;
                  v96 = v31;
                  v108 = v30;
                  goto LABEL_43;
                case 0xB:
                  v36 = *(_DWORD *)(v29 + 8) >> 8;
                  goto LABEL_31;
                case 0xD:
                  v85 = v32;
                  v98 = v31;
                  v110 = v30;
                  v50 = (_QWORD *)sub_15A9930(a5, v29);
                  v30 = v110;
                  v31 = v98;
                  v32 = v85;
                  v36 = 8LL * *v50;
                  goto LABEL_31;
                case 0xE:
                  v77 = v32;
                  v80 = v31;
                  v84 = v95;
                  v97 = *(_QWORD *)(v29 + 24);
                  v109 = *(_QWORD *)(v29 + 32);
                  v46 = sub_15A9FE0(a5, v97);
                  v32 = v77;
                  v47 = v97;
                  v48 = 1;
                  v31 = v80;
                  v30 = v84;
                  v49 = v46;
                  while ( 2 )
                  {
                    switch ( *(_BYTE *)(v47 + 8) )
                    {
                      case 0:
                      case 8:
                      case 0xA:
                      case 0xC:
                      case 0x10:
                        v63 = *(_QWORD *)(v47 + 32);
                        v47 = *(_QWORD *)(v47 + 24);
                        v48 *= v63;
                        continue;
                      case 1:
                        v53 = 16;
                        goto LABEL_56;
                      case 2:
                        v53 = 32;
                        goto LABEL_56;
                      case 3:
                      case 9:
                        v53 = 64;
                        goto LABEL_56;
                      case 4:
                        v53 = 80;
                        goto LABEL_56;
                      case 5:
                      case 6:
                        v53 = 128;
                        goto LABEL_56;
                      case 7:
                        v75 = v77;
                        v64 = 0;
                        v78 = v80;
                        v81 = v49;
                        v86 = v48;
                        v100 = v30;
                        goto LABEL_73;
                      case 0xB:
                        v53 = *(_DWORD *)(v47 + 8) >> 8;
                        goto LABEL_56;
                      case 0xD:
                        v75 = v77;
                        v78 = v80;
                        v81 = v49;
                        v86 = v48;
                        v100 = v30;
                        v53 = 8LL * *(_QWORD *)sub_15A9930(a5, v47);
                        goto LABEL_74;
                      case 0xE:
                        v72 = v77;
                        v73 = v80;
                        v74 = v49;
                        v76 = v48;
                        v79 = v84;
                        v101 = *(_QWORD *)(v47 + 32);
                        v82 = *(_QWORD *)(v47 + 24);
                        v87 = (unsigned int)sub_15A9FE0(a5, v82);
                        v65 = sub_127FA20((__int64)a5, v82);
                        v30 = v79;
                        v48 = v76;
                        v49 = v74;
                        v31 = v73;
                        v32 = v72;
                        v53 = 8 * v87 * v101 * ((v87 + ((unsigned __int64)(v65 + 7) >> 3) - 1) / v87);
                        goto LABEL_56;
                      case 0xF:
                        v75 = v77;
                        v78 = v80;
                        v81 = v49;
                        v64 = *(_DWORD *)(v47 + 8) >> 8;
                        v86 = v48;
                        v100 = v30;
LABEL_73:
                        v53 = 8 * (unsigned int)sub_15A9520(a5, v64);
LABEL_74:
                        v30 = v100;
                        v48 = v86;
                        v49 = v81;
                        v31 = v78;
                        v32 = v75;
LABEL_56:
                        v36 = 8 * v49 * v109 * ((v49 + ((unsigned __int64)(v48 * v53 + 7) >> 3) - 1) / v49);
                        break;
                    }
                    goto LABEL_31;
                  }
                case 0xF:
                  v83 = v32;
                  v96 = v31;
                  v108 = v30;
                  v44 = *(_DWORD *)(v29 + 8) >> 8;
LABEL_43:
                  v45 = sub_15A9520(a5, v44);
                  v30 = v108;
                  v31 = v96;
                  v32 = v83;
                  v36 = (unsigned int)(8 * v45);
LABEL_31:
                  if ( v27 < v32 * ((v32 + ((unsigned __int64)(v36 * v31 + 7) >> 3) - 1) / v32)
                    && !(unsigned __int8)sub_14D5510(
                                           *(_QWORD *)&v24[6 * (v30 - (v24[5] & 0xFFFFFFF))],
                                           v27,
                                           v90,
                                           v10,
                                           a5) )
                  {
                    return 0;
                  }
                  if ( ++v26 == *(_DWORD *)(*(_QWORD *)v24 + 12LL) )
                    return 1;
                  v25 = v26;
                  v37 = *(_QWORD *)(v88 + 8LL * v26 + 16) - (v27 + v92);
                  if ( v10 <= v37 )
                    return 1;
                  v90 += v37;
                  v38 = v27 + v92;
                  v92 = *(_QWORD *)(v88 + 8LL * v26 + 16);
                  v27 = 0;
                  v10 = v38 + v10 - v92;
                  break;
              }
              break;
            }
          }
        }
        if ( ((v7 - 6) & 0xFD) == 0 || (unsigned int)v7 - 11 <= 1 )
        {
          v107 = **(_QWORD **)(*v8 + 16LL);
          v39 = sub_15A9FE0(a5, v107);
          v40 = v107;
          v41 = 1;
          v42 = v39;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v40 + 8) )
            {
              case 1:
                v54 = 16;
                goto LABEL_59;
              case 2:
                v54 = 32;
                goto LABEL_59;
              case 3:
              case 9:
                v54 = 64;
                goto LABEL_59;
              case 4:
                v54 = 80;
                goto LABEL_59;
              case 5:
              case 6:
                v54 = 128;
                goto LABEL_59;
              case 7:
                v102 = v42;
                v67 = 0;
                v112 = v41;
                goto LABEL_82;
              case 0xB:
                v54 = *(_DWORD *)(v40 + 8) >> 8;
                goto LABEL_59;
              case 0xD:
                v104 = v42;
                v114 = v41;
                v70 = (_QWORD *)sub_15A9930(a5, v40);
                v41 = v114;
                v42 = v104;
                v54 = 8LL * *v70;
                goto LABEL_59;
              case 0xE:
                v89 = v42;
                v91 = v41;
                v94 = *(_QWORD *)(v40 + 24);
                v113 = *(_QWORD *)(v40 + 32);
                v103 = (unsigned int)sub_15A9FE0(a5, v94);
                v69 = sub_127FA20((__int64)a5, v94);
                v41 = v91;
                v42 = v89;
                v54 = 8 * v113 * v103 * ((v103 + ((unsigned __int64)(v69 + 7) >> 3) - 1) / v103);
                goto LABEL_59;
              case 0xF:
                v102 = v42;
                v112 = v41;
                v67 = *(_DWORD *)(v40 + 8) >> 8;
LABEL_82:
                v68 = sub_15A9520(a5, v67);
                v41 = v112;
                v42 = v102;
                v54 = (unsigned int)(8 * v68);
LABEL_59:
                v93 = v42 * ((v42 + ((unsigned __int64)(v41 * v54 + 7) >> 3) - 1) / v42);
                v56 = a2;
                v55 = a2 / v93;
                v57 = a2 % v93;
                v58 = v56 / v93;
                v111 = *(_QWORD *)(*v8 + 32LL);
                if ( *(_BYTE *)(*v8 + 8LL) != 14 )
                  v111 = (unsigned int)v111;
                if ( v55 == v111 )
                  return 1;
                v99 = v8;
                v59 = v10;
                v60 = v58;
                break;
              case 0x10:
                v66 = *(_QWORD *)(v40 + 32);
                v40 = *(_QWORD *)(v40 + 24);
                v41 *= v66;
                continue;
              default:
                JUMPOUT(0x419798);
            }
            break;
          }
          while ( 1 )
          {
            v61 = sub_15A0A60(v99, (unsigned int)v60);
            if ( !(unsigned __int8)sub_14D5510(v61, v57, a3, v59, a5) )
              break;
            v62 = v93 - v57;
            if ( v59 > v93 - v57 )
            {
              v59 -= v62;
              a3 += v62;
              ++v60;
              v57 = 0;
              if ( v60 != v111 )
                continue;
            }
            return 1;
          }
          return 0;
        }
        if ( v7 != 5 )
          return 0;
        if ( *((_WORD *)v8 + 9) != 46 )
          return 0;
        v71 = *((_DWORD *)v8 + 5) & 0xFFFFFFF;
        v115 = *(_QWORD *)v8[-3 * v71];
        if ( v115 != sub_15A9650(a5, *v8, 4 * v71, a4, a5, a6) )
          return 0;
        v8 = (_QWORD *)v8[-3 * (*((_DWORD *)v8 + 5) & 0xFFFFFFF)];
      }
      v7 = *((_BYTE *)v8 + 16);
      if ( (unsigned __int8)(v7 - 9) <= 1u )
        return 1;
    }
    v12 = *((_DWORD *)v8 + 8);
    if ( v12 > 0x40 || (v12 & 7) != 0 )
      return 0;
    v13 = v8[3];
    v14 = v12 >> 3;
    if ( v10 )
    {
      v15 = v14;
      if ( a2 != v14 )
      {
        v16 = a2;
        v17 = v14 - 1;
        v18 = a3 - a2;
        while ( 1 )
        {
          v19 = v17 - v16;
          if ( !*a5 )
            v19 = v16;
          *(_BYTE *)(v18 + v16) = v13 >> (8 * v19);
          if ( v16 == a2 + v10 - 1 || v15 == v16 + 1 )
            break;
          ++v16;
        }
      }
    }
  }
  return 1;
}
