// Function: sub_1A40240
// Address: 0x1a40240
//
__int64 __fastcall sub_1A40240(__int64 a1, unsigned int a2, __int64 *a3, __int64 a4)
{
  __int64 v6; // rsi
  __int64 v8; // r15
  __int64 v9; // r9
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 v15; // rsi
  __int64 v16; // r12
  _QWORD *v17; // rax
  __int64 v18; // r12
  unsigned int v19; // eax
  __int64 v20; // r9
  unsigned __int64 v21; // rcx
  int v22; // eax
  int v23; // eax
  __int64 v24; // rax
  __int64 v25; // rax
  unsigned int v26; // eax
  __int64 v27; // rsi
  __int64 v28; // r10
  unsigned __int64 v29; // r9
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // r13
  __int64 v33; // r15
  __int64 v34; // rsi
  unsigned __int64 v35; // rcx
  __int64 v36; // rax
  __int64 v37; // rax
  unsigned int v38; // esi
  int v39; // eax
  _QWORD *v40; // rax
  unsigned __int64 v41; // r12
  __int64 v42; // rax
  __int64 v43; // rax
  unsigned int v44; // esi
  int v45; // eax
  _QWORD *v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  __int64 v49; // rax
  int v50; // eax
  __int64 v51; // rax
  __int64 v52; // rax
  int v53; // eax
  _QWORD *v54; // rax
  __int64 v55; // [rsp+0h] [rbp-60h]
  __int64 v56; // [rsp+8h] [rbp-58h]
  unsigned __int64 v57; // [rsp+8h] [rbp-58h]
  unsigned __int64 v58; // [rsp+10h] [rbp-50h]
  __int64 v59; // [rsp+10h] [rbp-50h]
  unsigned __int64 v60; // [rsp+10h] [rbp-50h]
  __int64 v61; // [rsp+18h] [rbp-48h]
  __int64 v62; // [rsp+18h] [rbp-48h]
  __int64 v63; // [rsp+18h] [rbp-48h]
  __int64 v64; // [rsp+18h] [rbp-48h]
  __int64 v65; // [rsp+18h] [rbp-48h]
  unsigned __int64 v66; // [rsp+18h] [rbp-48h]
  __int64 v67; // [rsp+18h] [rbp-48h]
  __int64 v68; // [rsp+20h] [rbp-40h]
  unsigned __int64 v69; // [rsp+20h] [rbp-40h]
  unsigned __int64 v70; // [rsp+20h] [rbp-40h]
  __int64 v71; // [rsp+20h] [rbp-40h]
  unsigned __int64 v72; // [rsp+20h] [rbp-40h]
  unsigned __int64 v73; // [rsp+20h] [rbp-40h]
  __int64 v74; // [rsp+20h] [rbp-40h]
  unsigned __int64 v75; // [rsp+20h] [rbp-40h]
  __int64 v76; // [rsp+28h] [rbp-38h]
  __int64 v77; // [rsp+28h] [rbp-38h]
  unsigned __int64 v78; // [rsp+28h] [rbp-38h]
  __int64 v79; // [rsp+28h] [rbp-38h]
  unsigned __int64 v80; // [rsp+28h] [rbp-38h]
  unsigned __int64 v81; // [rsp+28h] [rbp-38h]

  if ( *(_BYTE *)(a1 + 8) == 16 )
  {
    *a3 = a1;
    v6 = *(_QWORD *)(a1 + 24);
    v8 = 1;
    a3[1] = v6;
    v9 = v6;
    while ( 2 )
    {
      switch ( *(_BYTE *)(v9 + 8) )
      {
        case 0:
        case 8:
        case 0xA:
        case 0xC:
        case 0x10:
          v24 = *(_QWORD *)(v9 + 32);
          v9 = *(_QWORD *)(v9 + 24);
          v8 *= v24;
          continue;
        case 1:
          v11 = 16;
          goto LABEL_7;
        case 2:
          v11 = 32;
          goto LABEL_7;
        case 3:
        case 9:
          v11 = 64;
          goto LABEL_7;
        case 4:
          v11 = 80;
          goto LABEL_7;
        case 5:
        case 6:
          v11 = 128;
          goto LABEL_7;
        case 7:
          v23 = sub_15A9520(a4, 0);
          v6 = a3[1];
          v11 = (unsigned int)(8 * v23);
          goto LABEL_7;
        case 0xB:
          v11 = *(_DWORD *)(v9 + 8) >> 8;
          goto LABEL_7;
        case 0xD:
          v17 = (_QWORD *)sub_15A9930(a4, v9);
          v6 = a3[1];
          v11 = 8LL * *v17;
          goto LABEL_7;
        case 0xE:
          v18 = *(_QWORD *)(v9 + 24);
          v76 = *(_QWORD *)(v9 + 32);
          v19 = sub_15A9FE0(a4, v18);
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
                v43 = *(_QWORD *)(v18 + 32);
                v18 = *(_QWORD *)(v18 + 24);
                v20 *= v43;
                continue;
              case 1:
                v36 = 16;
                goto LABEL_53;
              case 2:
                v36 = 32;
                goto LABEL_53;
              case 3:
              case 9:
                v36 = 64;
                goto LABEL_53;
              case 4:
                v36 = 80;
                goto LABEL_53;
              case 5:
              case 6:
                v36 = 128;
                goto LABEL_53;
              case 7:
                v61 = v20;
                v38 = 0;
                v69 = v21;
                goto LABEL_60;
              case 0xB:
                v36 = *(_DWORD *)(v18 + 8) >> 8;
                goto LABEL_53;
              case 0xD:
                v62 = v20;
                v70 = v21;
                v40 = (_QWORD *)sub_15A9930(a4, v18);
                v21 = v70;
                v20 = v62;
                v36 = 8LL * *v40;
                goto LABEL_53;
              case 0xE:
                v56 = v20;
                v58 = v21;
                v63 = *(_QWORD *)(v18 + 24);
                v71 = *(_QWORD *)(v18 + 32);
                v41 = (unsigned int)sub_15A9FE0(a4, v63);
                v42 = sub_127FA20(a4, v63);
                v21 = v58;
                v20 = v56;
                v36 = 8 * v41 * v71 * ((v41 + ((unsigned __int64)(v42 + 7) >> 3) - 1) / v41);
                goto LABEL_53;
              case 0xF:
                v61 = v20;
                v69 = v21;
                v38 = *(_DWORD *)(v18 + 8) >> 8;
LABEL_60:
                v39 = sub_15A9520(a4, v38);
                v21 = v69;
                v20 = v61;
                v36 = (unsigned int)(8 * v39);
LABEL_53:
                v6 = a3[1];
                v11 = 8 * v76 * v21 * ((v21 + ((unsigned __int64)(v36 * v20 + 7) >> 3) - 1) / v21);
                break;
            }
            goto LABEL_7;
          }
        case 0xF:
          v22 = sub_15A9520(a4, *(_DWORD *)(v9 + 8) >> 8);
          v6 = a3[1];
          v11 = (unsigned int)(8 * v22);
LABEL_7:
          v12 = v11 * v8;
          v13 = 1;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v6 + 8) )
            {
              case 1:
                v14 = 16;
                goto LABEL_11;
              case 2:
                v14 = 32;
                goto LABEL_11;
              case 3:
              case 9:
                v14 = 64;
                goto LABEL_11;
              case 4:
                v14 = 80;
                goto LABEL_11;
              case 5:
              case 6:
                v14 = 128;
                goto LABEL_11;
              case 7:
                v14 = 8 * (unsigned int)sub_15A9520(a4, 0);
                goto LABEL_11;
              case 0xB:
                v14 = *(_DWORD *)(v6 + 8) >> 8;
                goto LABEL_11;
              case 0xD:
                v14 = 8LL * *(_QWORD *)sub_15A9930(a4, v6);
                goto LABEL_11;
              case 0xE:
                v68 = *(_QWORD *)(v6 + 24);
                v77 = *(_QWORD *)(v6 + 32);
                v26 = sub_15A9FE0(a4, v68);
                v27 = v68;
                v28 = 1;
                v29 = v26;
                while ( 2 )
                {
                  switch ( *(_BYTE *)(v27 + 8) )
                  {
                    case 1:
                      v37 = 16;
                      goto LABEL_56;
                    case 2:
                      v37 = 32;
                      goto LABEL_56;
                    case 3:
                    case 9:
                      v37 = 64;
                      goto LABEL_56;
                    case 4:
                      v37 = 80;
                      goto LABEL_56;
                    case 5:
                    case 6:
                      v37 = 128;
                      goto LABEL_56;
                    case 7:
                      v64 = v28;
                      v44 = 0;
                      v72 = v29;
                      goto LABEL_70;
                    case 0xB:
                      v37 = *(_DWORD *)(v27 + 8) >> 8;
                      goto LABEL_56;
                    case 0xD:
                      v65 = v28;
                      v73 = v29;
                      v46 = (_QWORD *)sub_15A9930(a4, v27);
                      v29 = v73;
                      v28 = v65;
                      v37 = 8LL * *v46;
                      goto LABEL_56;
                    case 0xE:
                      v55 = v28;
                      v57 = v29;
                      v59 = *(_QWORD *)(v27 + 24);
                      v74 = *(_QWORD *)(v27 + 32);
                      v66 = (unsigned int)sub_15A9FE0(a4, v59);
                      v48 = sub_127FA20(a4, v59);
                      v29 = v57;
                      v28 = v55;
                      v37 = 8 * v74 * v66 * ((v66 + ((unsigned __int64)(v48 + 7) >> 3) - 1) / v66);
                      goto LABEL_56;
                    case 0xF:
                      v64 = v28;
                      v72 = v29;
                      v44 = *(_DWORD *)(v27 + 8) >> 8;
LABEL_70:
                      v45 = sub_15A9520(a4, v44);
                      v29 = v72;
                      v28 = v64;
                      v37 = (unsigned int)(8 * v45);
LABEL_56:
                      v14 = 8 * v29 * v77 * ((v29 + ((unsigned __int64)(v37 * v28 + 7) >> 3) - 1) / v29);
                      goto LABEL_11;
                    case 0x10:
                      v47 = *(_QWORD *)(v27 + 32);
                      v27 = *(_QWORD *)(v27 + 24);
                      v28 *= v47;
                      continue;
                    default:
                      goto LABEL_5;
                  }
                }
              case 0xF:
                v14 = 8 * (unsigned int)sub_15A9520(a4, *(_DWORD *)(v6 + 8) >> 8);
LABEL_11:
                if ( ((v14 * v13 + 7) & 0xFFFFFFFFFFFFFFF8LL) == v12 )
                {
                  if ( a2 )
                    a3[2] = a2;
                  else
                    a3[2] = (unsigned int)sub_15A9FE0(a4, *a3);
                  v15 = a3[1];
                  v16 = 1;
                  while ( 2 )
                  {
                    switch ( *(_BYTE *)(v15 + 8) )
                    {
                      case 1:
                        v30 = 16;
                        goto LABEL_36;
                      case 2:
                        v30 = 32;
                        goto LABEL_36;
                      case 3:
                      case 9:
                        v30 = 64;
                        goto LABEL_36;
                      case 4:
                        v30 = 80;
                        goto LABEL_36;
                      case 5:
                      case 6:
                        v30 = 128;
                        goto LABEL_36;
                      case 7:
                        v30 = 8 * (unsigned int)sub_15A9520(a4, 0);
                        goto LABEL_36;
                      case 0xB:
                        v30 = *(_DWORD *)(v15 + 8) >> 8;
                        goto LABEL_36;
                      case 0xD:
                        v30 = 8LL * *(_QWORD *)sub_15A9930(a4, v15);
                        goto LABEL_36;
                      case 0xE:
                        v32 = *(_QWORD *)(v15 + 32);
                        v33 = 1;
                        v34 = *(_QWORD *)(v15 + 24);
                        v35 = (unsigned int)sub_15A9FE0(a4, v34);
                        while ( 2 )
                        {
                          switch ( *(_BYTE *)(v34 + 8) )
                          {
                            case 1:
                              v49 = 16;
                              goto LABEL_78;
                            case 2:
                              v49 = 32;
                              goto LABEL_78;
                            case 3:
                            case 9:
                              v49 = 64;
                              goto LABEL_78;
                            case 4:
                              v49 = 80;
                              goto LABEL_78;
                            case 5:
                            case 6:
                              v49 = 128;
                              goto LABEL_78;
                            case 7:
                              v80 = v35;
                              v53 = sub_15A9520(a4, 0);
                              v35 = v80;
                              v49 = (unsigned int)(8 * v53);
                              goto LABEL_78;
                            case 0xB:
                              v49 = *(_DWORD *)(v34 + 8) >> 8;
                              goto LABEL_78;
                            case 0xD:
                              v81 = v35;
                              v54 = (_QWORD *)sub_15A9930(a4, v34);
                              v35 = v81;
                              v49 = 8LL * *v54;
                              goto LABEL_78;
                            case 0xE:
                              v60 = v35;
                              v67 = *(_QWORD *)(v34 + 24);
                              v79 = *(_QWORD *)(v34 + 32);
                              v75 = (unsigned int)sub_15A9FE0(a4, v67);
                              v51 = sub_127FA20(a4, v67);
                              v35 = v60;
                              v49 = 8 * v75 * v79 * ((v75 + ((unsigned __int64)(v51 + 7) >> 3) - 1) / v75);
                              goto LABEL_78;
                            case 0xF:
                              v78 = v35;
                              v50 = sub_15A9520(a4, *(_DWORD *)(v34 + 8) >> 8);
                              v35 = v78;
                              v49 = (unsigned int)(8 * v50);
LABEL_78:
                              v30 = 8 * v35 * v32 * ((v35 + ((unsigned __int64)(v49 * v33 + 7) >> 3) - 1) / v35);
                              goto LABEL_36;
                            case 0x10:
                              v52 = *(_QWORD *)(v34 + 32);
                              v34 = *(_QWORD *)(v34 + 24);
                              v33 *= v52;
                              continue;
                            default:
                              goto LABEL_5;
                          }
                        }
                      case 0xF:
                        v30 = 8 * (unsigned int)sub_15A9520(a4, *(_DWORD *)(v15 + 8) >> 8);
LABEL_36:
                        a3[3] = (unsigned __int64)(v30 * v16 + 7) >> 3;
                        return 1;
                      case 0x10:
                        v31 = *(_QWORD *)(v15 + 32);
                        v15 = *(_QWORD *)(v15 + 24);
                        v16 *= v31;
                        continue;
                      default:
                        goto LABEL_5;
                    }
                  }
                }
                return 0;
              case 0x10:
                v25 = *(_QWORD *)(v6 + 32);
                v6 = *(_QWORD *)(v6 + 24);
                v13 *= v25;
                continue;
              default:
LABEL_5:
                BUG();
            }
          }
      }
    }
  }
  *a3 = 0;
  return 0;
}
