// Function: sub_1833160
// Address: 0x1833160
//
__int64 __fastcall sub_1833160(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // r15
  __int64 v6; // r12
  __int64 v7; // rsi
  unsigned __int64 v8; // rbx
  __int64 v9; // rax
  int v10; // eax
  __int64 v12; // rbx
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // r14
  __int64 v16; // r12
  __int64 v17; // r15
  unsigned int v18; // eax
  __int64 v19; // rcx
  unsigned __int64 v20; // r8
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rbx
  unsigned int v24; // eax
  __int64 v25; // rsi
  __int64 v26; // r8
  unsigned __int64 v27; // r12
  _QWORD *v28; // rax
  unsigned int v29; // esi
  int v30; // eax
  __int64 v31; // rax
  unsigned int v32; // eax
  __int64 v33; // rsi
  __int64 v34; // r10
  unsigned __int64 v35; // r15
  __int64 v36; // rax
  unsigned int v37; // eax
  __int64 v38; // rsi
  __int64 v39; // r8
  unsigned __int64 v40; // r9
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  int v44; // eax
  int v45; // eax
  __int64 v46; // rax
  _QWORD *v47; // rax
  __int64 v48; // rax
  __int64 v49; // rax
  _QWORD *v50; // rax
  unsigned int v51; // esi
  int v52; // eax
  __int64 v53; // rax
  unsigned int v54; // esi
  int v55; // eax
  __int64 v56; // rax
  _QWORD *v57; // rax
  __int64 v58; // rax
  __int64 v59; // [rsp+8h] [rbp-78h]
  __int64 v60; // [rsp+10h] [rbp-70h]
  __int64 v61; // [rsp+18h] [rbp-68h]
  unsigned __int64 v62; // [rsp+18h] [rbp-68h]
  __int64 v63; // [rsp+20h] [rbp-60h]
  __int64 v64; // [rsp+20h] [rbp-60h]
  __int64 v65; // [rsp+20h] [rbp-60h]
  __int64 v66; // [rsp+28h] [rbp-58h]
  __int64 v67; // [rsp+28h] [rbp-58h]
  unsigned __int64 v68; // [rsp+28h] [rbp-58h]
  __int64 v69; // [rsp+28h] [rbp-58h]
  unsigned __int64 v70; // [rsp+30h] [rbp-50h]
  unsigned __int64 v71; // [rsp+30h] [rbp-50h]
  __int64 v72; // [rsp+30h] [rbp-50h]
  __int64 v73; // [rsp+30h] [rbp-50h]
  __int64 v74; // [rsp+30h] [rbp-50h]
  __int64 v75; // [rsp+30h] [rbp-50h]
  __int64 v76; // [rsp+30h] [rbp-50h]
  unsigned __int64 v77; // [rsp+30h] [rbp-50h]
  unsigned __int64 v78; // [rsp+30h] [rbp-50h]
  unsigned __int64 v79; // [rsp+30h] [rbp-50h]
  __int64 v80; // [rsp+38h] [rbp-48h]
  __int64 v81; // [rsp+38h] [rbp-48h]
  unsigned __int64 v82; // [rsp+38h] [rbp-48h]
  __int64 v83; // [rsp+38h] [rbp-48h]
  __int64 v84; // [rsp+38h] [rbp-48h]
  __int64 v85; // [rsp+38h] [rbp-48h]
  __int64 v86; // [rsp+40h] [rbp-40h]
  __int64 v87; // [rsp+40h] [rbp-40h]
  __int64 v88; // [rsp+40h] [rbp-40h]
  __int64 v89; // [rsp+40h] [rbp-40h]
  __int64 v90; // [rsp+40h] [rbp-40h]
  __int64 v91; // [rsp+40h] [rbp-40h]
  __int64 v92; // [rsp+40h] [rbp-40h]
  __int64 v94; // [rsp+48h] [rbp-38h]
  __int64 v95; // [rsp+48h] [rbp-38h]
  __int64 v96; // [rsp+48h] [rbp-38h]

  while ( 2 )
  {
    v3 = *(unsigned __int8 *)(a1 + 8);
    if ( (unsigned __int8)v3 > 0xFu || (v4 = 35454, !_bittest64(&v4, v3)) )
    {
      if ( (unsigned int)(v3 - 13) > 1 )
      {
LABEL_3:
        if ( (_DWORD)v3 != 16 )
          return 0;
      }
      if ( !sub_16435F0(a1, 0) )
        return 0;
      LODWORD(v3) = *(unsigned __int8 *)(a1 + 8);
    }
    switch ( (int)v4 )
    {
      case 0:
        v22 = 8LL * *(_QWORD *)sub_15A9930(a2, a1);
        goto LABEL_8;
      case 1:
        v23 = *(_QWORD *)(a1 + 32);
        v87 = *(_QWORD *)(a1 + 24);
        v24 = sub_15A9FE0(a2, v87);
        v25 = v87;
        v26 = 1;
        v27 = v24;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v25 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v48 = *(_QWORD *)(v25 + 32);
              v25 = *(_QWORD *)(v25 + 24);
              v26 *= v48;
              continue;
            case 1:
              v41 = 16;
              goto LABEL_54;
            case 2:
              v41 = 32;
              goto LABEL_54;
            case 3:
            case 9:
              v41 = 64;
              goto LABEL_54;
            case 4:
              v41 = 80;
              goto LABEL_54;
            case 5:
            case 6:
              v41 = 128;
              goto LABEL_54;
            case 7:
              v89 = v26;
              v44 = sub_15A9520(a2, 0);
              v26 = v89;
              v41 = (unsigned int)(8 * v44);
              goto LABEL_54;
            case 0xB:
              v41 = *(_DWORD *)(v25 + 8) >> 8;
              goto LABEL_54;
            case 0xD:
              v92 = v26;
              v47 = (_QWORD *)sub_15A9930(a2, v25);
              v26 = v92;
              v41 = 8LL * *v47;
              goto LABEL_54;
            case 0xE:
              v67 = v26;
              v91 = *(_QWORD *)(v25 + 32);
              v73 = *(_QWORD *)(v25 + 24);
              v82 = (unsigned int)sub_15A9FE0(a2, v73);
              v46 = sub_127FA20(a2, v73);
              v26 = v67;
              v41 = 8 * v91 * v82 * ((v82 + ((unsigned __int64)(v46 + 7) >> 3) - 1) / v82);
              goto LABEL_54;
            case 0xF:
              v90 = v26;
              v45 = sub_15A9520(a2, *(_DWORD *)(v25 + 8) >> 8);
              v26 = v90;
              v41 = (unsigned int)(8 * v45);
LABEL_54:
              v22 = 8 * v27 * v23 * ((v27 + ((unsigned __int64)(v41 * v26 + 7) >> 3) - 1) / v27);
              break;
            default:
LABEL_91:
              BUG();
          }
          break;
        }
LABEL_8:
        v5 = v22;
        v6 = 1;
        v7 = a1;
        v8 = (unsigned int)sub_15A9FE0(a2, a1);
        break;
      default:
        goto LABEL_3;
    }
LABEL_9:
    switch ( *(_BYTE *)(v7 + 8) )
    {
      case 1:
        v9 = 16;
        goto LABEL_11;
      case 2:
        v9 = 32;
        goto LABEL_11;
      case 3:
      case 9:
        v9 = 64;
        goto LABEL_11;
      case 4:
        v9 = 80;
        goto LABEL_11;
      case 5:
      case 6:
        v9 = 128;
        goto LABEL_11;
      case 7:
        v9 = 8 * (unsigned int)sub_15A9520(a2, 0);
        goto LABEL_11;
      case 0xB:
        v9 = *(_DWORD *)(v7 + 8) >> 8;
        goto LABEL_11;
      case 0xD:
        v9 = 8LL * *(_QWORD *)sub_15A9930(a2, v7);
        goto LABEL_11;
      case 0xE:
        v88 = *(_QWORD *)(v7 + 32);
        v81 = *(_QWORD *)(v7 + 24);
        v37 = sub_15A9FE0(a2, v81);
        v38 = v81;
        v39 = 1;
        v40 = v37;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v38 + 8) )
          {
            case 1:
              v43 = 16;
              goto LABEL_60;
            case 2:
              v43 = 32;
              goto LABEL_60;
            case 3:
            case 9:
              v43 = 64;
              goto LABEL_60;
            case 4:
              v43 = 80;
              goto LABEL_60;
            case 5:
            case 6:
              v43 = 128;
              goto LABEL_60;
            case 7:
              v77 = v40;
              v54 = 0;
              v83 = v39;
              goto LABEL_83;
            case 0xB:
              v43 = *(_DWORD *)(v38 + 8) >> 8;
              goto LABEL_60;
            case 0xD:
              v79 = v40;
              v85 = v39;
              v57 = (_QWORD *)sub_15A9930(a2, v38);
              v39 = v85;
              v40 = v79;
              v43 = 8LL * *v57;
              goto LABEL_60;
            case 0xE:
              v62 = v40;
              v65 = v39;
              v84 = *(_QWORD *)(v38 + 32);
              v69 = *(_QWORD *)(v38 + 24);
              v78 = (unsigned int)sub_15A9FE0(a2, v69);
              v56 = sub_127FA20(a2, v69);
              v39 = v65;
              v40 = v62;
              v43 = 8 * v84 * v78 * ((v78 + ((unsigned __int64)(v56 + 7) >> 3) - 1) / v78);
              goto LABEL_60;
            case 0xF:
              v77 = v40;
              v83 = v39;
              v54 = *(_DWORD *)(v38 + 8) >> 8;
LABEL_83:
              v55 = sub_15A9520(a2, v54);
              v39 = v83;
              v40 = v77;
              v43 = (unsigned int)(8 * v55);
LABEL_60:
              v9 = 8 * v88 * v40 * ((v40 + ((unsigned __int64)(v43 * v39 + 7) >> 3) - 1) / v40);
              goto LABEL_11;
            case 0x10:
              v58 = *(_QWORD *)(v38 + 32);
              v38 = *(_QWORD *)(v38 + 24);
              v39 *= v58;
              continue;
            default:
              goto LABEL_91;
          }
        }
      case 0xF:
        v9 = 8 * (unsigned int)sub_15A9520(a2, *(_DWORD *)(v7 + 8) >> 8);
LABEL_11:
        if ( 8 * v8 * ((v8 + ((unsigned __int64)(v9 * v6 + 7) >> 3) - 1) / v8) != v5 )
          return 0;
        v10 = *(unsigned __int8 *)(a1 + 8);
        if ( v10 == 14 )
          goto LABEL_17;
        if ( v10 != 13 )
        {
          if ( v10 != 16 )
            return 1;
LABEL_17:
          a1 = *(_QWORD *)(a1 + 24);
          continue;
        }
        v12 = a1;
        v13 = a2;
        v86 = sub_15A9930(a2, v12);
        v14 = *(unsigned int *)(v12 + 12);
        if ( !(_DWORD)v14 )
          return 1;
        v15 = 0;
        v16 = 0;
        v80 = 8 * v14;
        while ( 2 )
        {
          v17 = *(_QWORD *)(*(_QWORD *)(v12 + 16) + v15);
          if ( (unsigned __int8)sub_1833160(v17, v13) && 8LL * *(_QWORD *)(v86 + v15 + 16) == v16 )
          {
            v18 = sub_15A9FE0(v13, v17);
            v19 = 1;
            v20 = v18;
LABEL_23:
            switch ( *(_BYTE *)(v17 + 8) )
            {
              case 1:
                v21 = 16;
                goto LABEL_25;
              case 2:
                v21 = 32;
                goto LABEL_25;
              case 3:
              case 9:
                v21 = 64;
                goto LABEL_25;
              case 4:
                v21 = 80;
                goto LABEL_25;
              case 5:
              case 6:
                v21 = 128;
                goto LABEL_25;
              case 7:
                v71 = v20;
                v29 = 0;
                v95 = v19;
                goto LABEL_37;
              case 0xB:
                v21 = *(_DWORD *)(v17 + 8) >> 8;
                goto LABEL_25;
              case 0xD:
                v70 = v20;
                v94 = v19;
                v28 = (_QWORD *)sub_15A9930(v13, v17);
                v19 = v94;
                v20 = v70;
                v21 = 8LL * *v28;
                goto LABEL_25;
              case 0xE:
                v63 = v20;
                v66 = v19;
                v72 = *(_QWORD *)(v17 + 24);
                v96 = *(_QWORD *)(v17 + 32);
                v32 = sub_15A9FE0(v13, v72);
                v20 = v63;
                v33 = v72;
                v34 = 1;
                v19 = v66;
                v35 = v32;
                while ( 2 )
                {
                  switch ( *(_BYTE *)(v33 + 8) )
                  {
                    case 1:
                      v42 = 16;
                      goto LABEL_57;
                    case 2:
                      v42 = 32;
                      goto LABEL_57;
                    case 3:
                    case 9:
                      v42 = 64;
                      goto LABEL_57;
                    case 4:
                      v42 = 80;
                      goto LABEL_57;
                    case 5:
                    case 6:
                      v42 = 128;
                      goto LABEL_57;
                    case 7:
                      v51 = 0;
                      v76 = v34;
                      goto LABEL_76;
                    case 0xB:
                      v42 = *(_DWORD *)(v33 + 8) >> 8;
                      goto LABEL_57;
                    case 0xD:
                      v75 = v34;
                      v50 = (_QWORD *)sub_15A9930(v13, v33);
                      v34 = v75;
                      v19 = v66;
                      v20 = v63;
                      v42 = 8LL * *v50;
                      goto LABEL_57;
                    case 0xE:
                      v59 = v63;
                      v60 = v66;
                      v61 = v34;
                      v64 = *(_QWORD *)(v33 + 24);
                      v74 = *(_QWORD *)(v33 + 32);
                      v68 = (unsigned int)sub_15A9FE0(v13, v64);
                      v49 = sub_127FA20(v13, v64);
                      v34 = v61;
                      v19 = v60;
                      v20 = v59;
                      v42 = 8 * v68 * v74 * ((v68 + ((unsigned __int64)(v49 + 7) >> 3) - 1) / v68);
                      goto LABEL_57;
                    case 0xF:
                      v76 = v34;
                      v51 = *(_DWORD *)(v33 + 8) >> 8;
LABEL_76:
                      v52 = sub_15A9520(v13, v51);
                      v34 = v76;
                      v19 = v66;
                      v20 = v63;
                      v42 = (unsigned int)(8 * v52);
LABEL_57:
                      v21 = 8 * v35 * v96 * ((v35 + ((unsigned __int64)(v42 * v34 + 7) >> 3) - 1) / v35);
                      goto LABEL_25;
                    case 0x10:
                      v53 = *(_QWORD *)(v33 + 32);
                      v33 = *(_QWORD *)(v33 + 24);
                      v34 *= v53;
                      continue;
                    default:
                      goto LABEL_91;
                  }
                }
              case 0xF:
                v71 = v20;
                v95 = v19;
                v29 = *(_DWORD *)(v17 + 8) >> 8;
LABEL_37:
                v30 = sub_15A9520(v13, v29);
                v19 = v95;
                v20 = v71;
                v21 = (unsigned int)(8 * v30);
LABEL_25:
                v15 += 8;
                v16 += 8 * v20 * ((v20 + ((unsigned __int64)(v21 * v19 + 7) >> 3) - 1) / v20);
                if ( v80 == v15 )
                  return 1;
                continue;
              case 0x10:
                v31 = *(_QWORD *)(v17 + 32);
                v17 = *(_QWORD *)(v17 + 24);
                v19 *= v31;
                goto LABEL_23;
              default:
                goto LABEL_91;
            }
          }
          return 0;
        }
      case 0x10:
        v36 = *(_QWORD *)(v7 + 32);
        v7 = *(_QWORD *)(v7 + 24);
        v6 *= v36;
        goto LABEL_9;
      default:
        goto LABEL_91;
    }
  }
}
