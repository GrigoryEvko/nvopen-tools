// Function: sub_1C7EB60
// Address: 0x1c7eb60
//
bool __fastcall sub_1C7EB60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 v8; // r12
  unsigned __int64 v9; // rbx
  unsigned int v10; // r13d
  __int64 v11; // rbx
  __int64 v12; // rsi
  __int64 v13; // r12
  unsigned int v14; // eax
  __int64 v15; // rcx
  unsigned __int64 v16; // r13
  __int64 v17; // rax
  __int64 v19; // rax
  __int64 v20; // r8
  __int64 v21; // rbx
  unsigned __int64 v22; // r14
  __int64 v23; // r14
  unsigned __int64 v24; // r15
  unsigned int v25; // r12d
  __int64 v26; // rbx
  __int64 v27; // rsi
  __int64 v28; // rax
  unsigned __int64 v29; // rbx
  __int64 v30; // rax
  unsigned int v31; // eax
  __int64 v32; // rsi
  __int64 v33; // r9
  unsigned __int64 v34; // r10
  __int64 v35; // rax
  __int64 v36; // rax
  int v37; // eax
  int v38; // eax
  unsigned int v39; // eax
  __int64 v40; // rsi
  __int64 v41; // r9
  unsigned __int64 v42; // r12
  _QWORD *v43; // rax
  __int64 v44; // r14
  unsigned __int64 v45; // r13
  __int64 v46; // rax
  __int64 v47; // rax
  unsigned int v48; // esi
  int v49; // eax
  _QWORD *v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rax
  unsigned int v54; // esi
  int v55; // eax
  __int64 v56; // rax
  _QWORD *v57; // rax
  unsigned __int64 v58; // [rsp+0h] [rbp-70h]
  __int64 v59; // [rsp+8h] [rbp-68h]
  __int64 v60; // [rsp+8h] [rbp-68h]
  __int64 v61; // [rsp+10h] [rbp-60h]
  __int64 v62; // [rsp+10h] [rbp-60h]
  unsigned __int64 v63; // [rsp+18h] [rbp-58h]
  unsigned __int64 v64; // [rsp+18h] [rbp-58h]
  unsigned __int64 v65; // [rsp+18h] [rbp-58h]
  __int64 v66; // [rsp+18h] [rbp-58h]
  __int64 v67; // [rsp+20h] [rbp-50h]
  __int64 v68; // [rsp+20h] [rbp-50h]
  __int64 v69; // [rsp+20h] [rbp-50h]
  __int64 v70; // [rsp+20h] [rbp-50h]
  __int64 v71; // [rsp+20h] [rbp-50h]
  unsigned __int64 v72; // [rsp+20h] [rbp-50h]
  __int64 v73; // [rsp+28h] [rbp-48h]
  __int64 v74; // [rsp+28h] [rbp-48h]
  __int64 v75; // [rsp+28h] [rbp-48h]
  __int64 v76; // [rsp+28h] [rbp-48h]
  __int64 v77; // [rsp+28h] [rbp-48h]
  unsigned __int64 v78; // [rsp+30h] [rbp-40h]
  __int64 v79; // [rsp+30h] [rbp-40h]
  __int64 v80; // [rsp+30h] [rbp-40h]
  __int64 v81; // [rsp+30h] [rbp-40h]
  __int64 v82; // [rsp+30h] [rbp-40h]
  __int64 v83; // [rsp+38h] [rbp-38h]
  __int64 v84; // [rsp+38h] [rbp-38h]
  __int64 v85; // [rsp+38h] [rbp-38h]

  v5 = 1;
  v6 = *(_QWORD *)a2;
  while ( 2 )
  {
    switch ( *(_BYTE *)(v6 + 8) )
    {
      case 0:
      case 8:
      case 0xA:
      case 0xC:
      case 0x10:
        v17 = *(_QWORD *)(v6 + 32);
        v6 = *(_QWORD *)(v6 + 24);
        v5 *= v17;
        continue;
      case 1:
        v7 = 16;
        break;
      case 2:
        v7 = 32;
        break;
      case 3:
      case 9:
        v7 = 64;
        break;
      case 4:
        v7 = 80;
        break;
      case 5:
      case 6:
        v7 = 128;
        break;
      case 7:
        v7 = 8 * (unsigned int)sub_15A9520(a3, 0);
        break;
      case 0xB:
        v7 = *(_DWORD *)(v6 + 8) >> 8;
        break;
      case 0xD:
        v7 = 8LL * *(_QWORD *)sub_15A9930(a3, v6);
        break;
      case 0xE:
        v13 = *(_QWORD *)(v6 + 24);
        v83 = *(_QWORD *)(v6 + 32);
        v14 = sub_15A9FE0(a3, v13);
        v15 = 1;
        v16 = v14;
        while ( 2 )
        {
          switch ( *(_BYTE *)(v13 + 8) )
          {
            case 0:
            case 8:
            case 0xA:
            case 0xC:
            case 0x10:
              v36 = *(_QWORD *)(v13 + 32);
              v13 = *(_QWORD *)(v13 + 24);
              v15 *= v36;
              continue;
            case 1:
              v19 = 16;
              break;
            case 2:
              v19 = 32;
              break;
            case 3:
            case 9:
              v19 = 64;
              break;
            case 4:
              v19 = 80;
              break;
            case 5:
            case 6:
              v19 = 128;
              break;
            case 7:
              v79 = v15;
              v37 = sub_15A9520(a3, 0);
              v15 = v79;
              v19 = (unsigned int)(8 * v37);
              break;
            case 0xB:
              v19 = *(_DWORD *)(v13 + 8) >> 8;
              break;
            case 0xD:
              v82 = v15;
              v43 = (_QWORD *)sub_15A9930(a3, v13);
              v15 = v82;
              v19 = 8LL * *v43;
              break;
            case 0xE:
              v68 = v15;
              v74 = *(_QWORD *)(v13 + 24);
              v81 = *(_QWORD *)(v13 + 32);
              v39 = sub_15A9FE0(a3, v74);
              v40 = v74;
              v15 = v68;
              v41 = 1;
              v42 = v39;
              while ( 2 )
              {
                switch ( *(_BYTE *)(v40 + 8) )
                {
                  case 0:
                  case 8:
                  case 0xA:
                  case 0xC:
                  case 0x10:
                    v53 = *(_QWORD *)(v40 + 32);
                    v40 = *(_QWORD *)(v40 + 24);
                    v41 *= v53;
                    continue;
                  case 1:
                    v52 = 16;
                    goto LABEL_85;
                  case 2:
                    v52 = 32;
                    goto LABEL_85;
                  case 3:
                  case 9:
                    v52 = 64;
                    goto LABEL_85;
                  case 4:
                    v52 = 80;
                    goto LABEL_85;
                  case 5:
                  case 6:
                    v52 = 128;
                    goto LABEL_85;
                  case 7:
                    v54 = 0;
                    v75 = v41;
                    goto LABEL_92;
                  case 0xB:
                    v52 = *(_DWORD *)(v40 + 8) >> 8;
                    goto LABEL_85;
                  case 0xD:
                    v77 = v41;
                    v57 = (_QWORD *)sub_15A9930(a3, v40);
                    v41 = v77;
                    v15 = v68;
                    v52 = 8LL * *v57;
                    goto LABEL_85;
                  case 0xE:
                    v60 = v68;
                    v62 = v41;
                    v66 = *(_QWORD *)(v40 + 24);
                    v76 = *(_QWORD *)(v40 + 32);
                    v72 = (unsigned int)sub_15A9FE0(a3, v66);
                    v56 = sub_127FA20(a3, v66);
                    v41 = v62;
                    v15 = v60;
                    v52 = 8 * v76 * v72 * ((v72 + ((unsigned __int64)(v56 + 7) >> 3) - 1) / v72);
                    goto LABEL_85;
                  case 0xF:
                    v75 = v41;
                    v54 = *(_DWORD *)(v40 + 8) >> 8;
LABEL_92:
                    v55 = sub_15A9520(a3, v54);
                    v41 = v75;
                    v15 = v68;
                    v52 = (unsigned int)(8 * v55);
LABEL_85:
                    v19 = 8 * v42 * v81 * ((v42 + ((unsigned __int64)(v52 * v41 + 7) >> 3) - 1) / v42);
                    break;
                }
                break;
              }
              break;
            case 0xF:
              v80 = v15;
              v38 = sub_15A9520(a3, *(_DWORD *)(v13 + 8) >> 8);
              v15 = v80;
              v19 = (unsigned int)(8 * v38);
              break;
          }
          break;
        }
        v7 = 8 * v16 * v83 * ((v16 + ((unsigned __int64)(v19 * v15 + 7) >> 3) - 1) / v16);
        break;
      case 0xF:
        v7 = 8 * (unsigned int)sub_15A9520(a3, *(_DWORD *)(v6 + 8) >> 8);
        break;
    }
    break;
  }
  if ( (unsigned int)dword_4FBD640 <= (unsigned __int64)(v7 * v5 + 7) >> 3 )
    return 1;
  v8 = *(_QWORD *)a2;
  if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 13 && *(_DWORD *)(v8 + 12) )
  {
    v9 = *(_QWORD *)sub_15A9930(a3, *(_QWORD *)a2) & 0x1FFFFFFFFFFFFFFFLL;
    v10 = 1 << (*(unsigned __int16 *)(a2 + 18) >> 1) >> 1;
    if ( !v10 )
      v10 = sub_15A9FE0(a3, *(_QWORD *)a2);
    v78 = v10;
    if ( !(v9 % v10) )
    {
      v11 = 1;
      v12 = **(_QWORD **)(v8 + 16);
      while ( 2 )
      {
        switch ( *(_BYTE *)(v12 + 8) )
        {
          case 1:
            v20 = 16;
            goto LABEL_30;
          case 2:
            v20 = 32;
            goto LABEL_30;
          case 3:
          case 9:
            v20 = 64;
            goto LABEL_30;
          case 4:
            v20 = 80;
            goto LABEL_30;
          case 5:
          case 6:
            v20 = 128;
            goto LABEL_30;
          case 7:
            v20 = 8 * (unsigned int)sub_15A9520(a3, 0);
            goto LABEL_30;
          case 0xB:
            v20 = *(_DWORD *)(v12 + 8) >> 8;
            goto LABEL_30;
          case 0xD:
            v20 = 8LL * *(_QWORD *)sub_15A9930(a3, v12);
            goto LABEL_30;
          case 0xE:
            v44 = *(_QWORD *)(v12 + 32);
            v85 = *(_QWORD *)(v12 + 24);
            v45 = (unsigned int)sub_15A9FE0(a3, v85);
            v20 = 8 * v45 * v44 * ((v45 + ((unsigned __int64)(sub_127FA20(a3, v85) + 7) >> 3) - 1) / v45);
            goto LABEL_30;
          case 0xF:
            v20 = 8 * (unsigned int)sub_15A9520(a3, *(_DWORD *)(v12 + 8) >> 8);
LABEL_30:
            v21 = v20 * v11;
            v22 = (unsigned __int64)(v21 + 7) >> 3;
            if ( *(_DWORD *)(v8 + 12) > 1u )
            {
              v84 = a3;
              v23 = v8;
              v24 = (unsigned __int64)(v21 + 7) >> 3;
              v25 = 1;
              while ( 2 )
              {
                v26 = 1;
                v27 = *(_QWORD *)(*(_QWORD *)(v23 + 16) + 8LL * v25);
LABEL_33:
                switch ( *(_BYTE *)(v27 + 8) )
                {
                  case 1:
                    v28 = 16;
                    goto LABEL_35;
                  case 2:
                    v28 = 32;
                    goto LABEL_35;
                  case 3:
                  case 9:
                    v28 = 64;
                    goto LABEL_35;
                  case 4:
                    v28 = 80;
                    goto LABEL_35;
                  case 5:
                  case 6:
                    v28 = 128;
                    goto LABEL_35;
                  case 7:
                    v28 = 8 * (unsigned int)sub_15A9520(v84, 0);
                    goto LABEL_35;
                  case 0xB:
                    v28 = *(_DWORD *)(v27 + 8) >> 8;
                    goto LABEL_35;
                  case 0xD:
                    v28 = 8LL * *(_QWORD *)sub_15A9930(v84, v27);
                    goto LABEL_35;
                  case 0xE:
                    v73 = *(_QWORD *)(v27 + 32);
                    v67 = *(_QWORD *)(v27 + 24);
                    v31 = sub_15A9FE0(v84, v67);
                    v32 = v67;
                    v33 = 1;
                    v34 = v31;
                    while ( 2 )
                    {
                      switch ( *(_BYTE *)(v32 + 8) )
                      {
                        case 1:
                          v35 = 16;
                          goto LABEL_53;
                        case 2:
                          v35 = 32;
                          goto LABEL_53;
                        case 3:
                        case 9:
                          v35 = 64;
                          goto LABEL_53;
                        case 4:
                          v35 = 80;
                          goto LABEL_53;
                        case 5:
                        case 6:
                          v35 = 128;
                          goto LABEL_53;
                        case 7:
                          v63 = v34;
                          v48 = 0;
                          v69 = v33;
                          goto LABEL_77;
                        case 0xB:
                          v35 = *(_DWORD *)(v32 + 8) >> 8;
                          goto LABEL_53;
                        case 0xD:
                          v64 = v34;
                          v70 = v33;
                          v50 = (_QWORD *)sub_15A9930(v84, v32);
                          v33 = v70;
                          v34 = v64;
                          v35 = 8LL * *v50;
                          goto LABEL_53;
                        case 0xE:
                          v58 = v34;
                          v59 = v33;
                          v71 = *(_QWORD *)(v32 + 32);
                          v61 = *(_QWORD *)(v32 + 24);
                          v65 = (unsigned int)sub_15A9FE0(v84, v61);
                          v51 = sub_127FA20(v84, v61);
                          v33 = v59;
                          v34 = v58;
                          v35 = 8 * v71 * v65 * ((v65 + ((unsigned __int64)(v51 + 7) >> 3) - 1) / v65);
                          goto LABEL_53;
                        case 0xF:
                          v63 = v34;
                          v69 = v33;
                          v48 = *(_DWORD *)(v32 + 8) >> 8;
LABEL_77:
                          v49 = sub_15A9520(v84, v48);
                          v33 = v69;
                          v34 = v63;
                          v35 = (unsigned int)(8 * v49);
LABEL_53:
                          v28 = 8 * v73 * v34 * ((v34 + ((unsigned __int64)(v35 * v33 + 7) >> 3) - 1) / v34);
                          goto LABEL_35;
                        case 0x10:
                          v47 = *(_QWORD *)(v32 + 32);
                          v32 = *(_QWORD *)(v32 + 24);
                          v33 *= v47;
                          continue;
                        default:
                          goto LABEL_3;
                      }
                    }
                  case 0xF:
                    v28 = 8 * (unsigned int)sub_15A9520(v84, *(_DWORD *)(v27 + 8) >> 8);
LABEL_35:
                    v29 = (unsigned __int64)(v28 * v26 + 7) >> 3;
                    if ( v24 > v29 )
                      v24 = v29;
                    if ( *(_DWORD *)(v23 + 12) > ++v25 )
                      continue;
                    v22 = v24;
                    break;
                  case 0x10:
                    v30 = *(_QWORD *)(v27 + 32);
                    v27 = *(_QWORD *)(v27 + 24);
                    v26 *= v30;
                    goto LABEL_33;
                  default:
                    goto LABEL_3;
                }
                break;
              }
            }
            return v78 > v22;
          case 0x10:
            v46 = *(_QWORD *)(v12 + 32);
            v12 = *(_QWORD *)(v12 + 24);
            v11 *= v46;
            continue;
          default:
LABEL_3:
            BUG();
        }
      }
    }
  }
  return 0;
}
