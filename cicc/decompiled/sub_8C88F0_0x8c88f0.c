// Function: sub_8C88F0
// Address: 0x8c88f0
//
__int64 __fastcall sub_8C88F0(__int64 *a1, int a2)
{
  __int64 *v2; // r15
  __int64 v3; // rdi
  __int64 result; // rax
  __int64 *v5; // r8
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rbx
  __int64 v9; // r14
  __int64 v10; // r12
  char v12; // r12
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // r15
  __int64 v16; // r13
  char v17; // al
  __int64 v18; // rdx
  __int64 v19; // rbx
  __int64 *v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rcx
  _UNKNOWN *__ptr32 *v23; // r8
  __int64 v24; // rsi
  __int64 v25; // rdi
  __int64 *v26; // rax
  __int64 v27; // rdx
  __int64 v28; // r12
  __int64 v29; // r10
  __int64 v30; // rdx
  __int64 v31; // r12
  __int64 v32; // rax
  __int64 v33; // rdi
  __int64 v34; // rsi
  int v35; // eax
  __int64 v36; // r12
  __int64 v37; // r13
  __int64 v38; // rbx
  _UNKNOWN *__ptr32 *v39; // r8
  __int64 v40; // r13
  __int64 v41; // rbx
  __int64 i; // r13
  __int64 v43; // r12
  _QWORD *v45; // rax
  __int64 v46; // rbx
  __int64 v47; // r12
  __int64 v48; // r15
  char v49; // dl
  __int64 v50; // rsi
  char v51; // al
  __int64 v52; // r13
  __int64 **v53; // rdi
  __int64 v54; // r14
  __int64 **v55; // rcx
  __int64 v56; // rbx
  __int64 *v57; // rdx
  __int64 v58; // rsi
  __int64 *v59; // rax
  __int64 v60; // rdi
  __int64 v61; // rcx
  _UNKNOWN *__ptr32 *v62; // r8
  __int64 v63; // rdi
  __int64 v64; // rsi
  __int64 *v65; // rax
  __int64 *v66; // [rsp+8h] [rbp-68h]
  __int64 v67; // [rsp+10h] [rbp-60h]
  char v69; // [rsp+1Eh] [rbp-52h]
  __int64 v70; // [rsp+20h] [rbp-50h]
  __int64 v71; // [rsp+28h] [rbp-48h]
  __int64 v72; // [rsp+28h] [rbp-48h]
  __int64 *v73; // [rsp+30h] [rbp-40h]
  __int64 v74; // [rsp+38h] [rbp-38h]
  char v75; // [rsp+38h] [rbp-38h]
  __int64 *v76; // [rsp+38h] [rbp-38h]

  v2 = a1;
  v3 = *a1;
  result = (unsigned int)*(unsigned __int8 *)(v3 + 80) - 19;
  if ( (unsigned __int8)(*(_BYTE *)(v3 + 80) - 19) <= 3u && (*(_BYTE *)(v3 + 81) & 0x40) == 0 && !v2[4] )
  {
    v5 = (__int64 *)v2[25];
    if ( v5 && v5 != v2 )
    {
      v6 = sub_8C9880(v2[25]);
      return sub_8CBB20(59, v2, v6);
    }
    if ( *(_DWORD *)(v3 + 40) != -1 )
    {
      result = sub_8C6B40(v3);
      if ( (_DWORD)result )
      {
        v74 = *v2;
        if ( (unsigned __int8)sub_877F80(*v2) == 7 )
        {
          if ( (unsigned int)*(unsigned __int8 *)(v74 + 80) - 19 > 3 )
            goto LABEL_145;
          v41 = **(_QWORD **)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v74 + 88) + 104LL) + 192LL) + 176LL);
          v73 = *(__int64 **)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v74 + 88) + 104LL) + 192LL) + 176LL);
          for ( i = *(_QWORD *)(*(_QWORD *)v41 + 32LL); i; i = *(_QWORD *)(i + 8) )
          {
            v43 = sub_880F80(v41);
            if ( *(_DWORD *)(i + 40) != -1
              && v43 != sub_880F80(i)
              && (unsigned int)sub_8C6B40(i)
              && !(a2 ? !sub_8C6230(i, v41) : (unsigned int)sub_8C7F70(i, v41) == 0)
              && *(_BYTE *)(i + 80) == 19 )
            {
              v45 = (_QWORD *)sub_8C8620(*v73, i);
              if ( v45 )
              {
                v46 = *(_QWORD *)(sub_8794A0(v45) + 216);
                if ( v46 )
                {
                  v47 = v74;
                  v76 = v2;
                  v48 = v46;
                  while ( 2 )
                  {
                    v49 = *(_BYTE *)(v47 + 80);
                    switch ( v49 )
                    {
                      case 4:
                      case 5:
                        v50 = *(_QWORD *)(*(_QWORD *)(v47 + 96) + 80LL);
                        goto LABEL_110;
                      case 6:
                        v50 = *(_QWORD *)(*(_QWORD *)(v47 + 96) + 32LL);
                        goto LABEL_113;
                      case 9:
                      case 10:
                        v50 = *(_QWORD *)(*(_QWORD *)(v47 + 96) + 56LL);
LABEL_113:
                        v51 = *(_BYTE *)(v48 + 80);
                        switch ( v51 )
                        {
                          case 4:
                          case 5:
LABEL_132:
                            v53 = *(__int64 ***)(v48 + 88);
                            v52 = *(_QWORD *)(*(_QWORD *)(v48 + 96) + 80LL);
                            break;
                          case 6:
LABEL_133:
                            v53 = *(__int64 ***)(v48 + 88);
                            v52 = *(_QWORD *)(*(_QWORD *)(v48 + 96) + 32LL);
                            v51 = 6;
                            break;
                          case 9:
LABEL_135:
                            v53 = *(__int64 ***)(v48 + 88);
                            v52 = *(_QWORD *)(*(_QWORD *)(v48 + 96) + 56LL);
                            v51 = 9;
                            break;
                          case 10:
LABEL_134:
                            v53 = *(__int64 ***)(v48 + 88);
                            v52 = *(_QWORD *)(*(_QWORD *)(v48 + 96) + 56LL);
                            v51 = 10;
                            break;
                          case 19:
                          case 20:
                          case 21:
                          case 22:
LABEL_114:
                            v52 = *(_QWORD *)(v48 + 88);
                            v53 = (__int64 **)v52;
                            break;
                          default:
LABEL_63:
                            BUG();
                        }
                        v54 = *(_QWORD *)(v50 + 176);
                        v55 = *(__int64 ***)(v47 + 88);
                        v56 = *(_QWORD *)(v52 + 176);
                        if ( v49 == 20 )
                        {
                          v58 = *v55[41];
                        }
                        else
                        {
                          if ( v49 == 21 )
                            v57 = v55[29];
                          else
                            v57 = v55[4];
                          v58 = *v57;
                        }
                        if ( v51 == 20 )
                        {
                          v60 = *v53[41];
                        }
                        else
                        {
                          if ( v51 == 21 )
                            v59 = v53[29];
                          else
                            v59 = v53[4];
                          v60 = *v59;
                        }
                        if ( (unsigned int)sub_89B3C0(v60, v58, 0, 0, (_DWORD *)(v47 + 48), 8u) )
                        {
                          v63 = *(_QWORD *)(v54 + 152);
                          v64 = *(_QWORD *)(v56 + 152);
                          if ( (v63 == v64 || (unsigned int)sub_8D97D0(v63, v64, 256, v61, v62))
                            && sub_89AB40(*(_QWORD *)(v54 + 240), *(_QWORD *)(v56 + 240), 0, v61, v62) )
                          {
                            v65 = *(__int64 **)(v52 + 104);
                            if ( v65 )
                            {
                              v2 = v76;
                              v27 = *v65;
                              goto LABEL_47;
                            }
                          }
                        }
                        v48 = *(_QWORD *)(v48 + 8);
                        if ( v48 )
                          continue;
                        v2 = v76;
                        break;
                      case 19:
                      case 20:
                      case 21:
                      case 22:
                        v50 = *(_QWORD *)(v47 + 88);
                        goto LABEL_110;
                      default:
                        v50 = 0;
LABEL_110:
                        v51 = *(_BYTE *)(v48 + 80);
                        switch ( v51 )
                        {
                          case 4:
                          case 5:
                            goto LABEL_132;
                          case 6:
                            goto LABEL_133;
                          case 9:
                            goto LABEL_135;
                          case 10:
                            goto LABEL_134;
                          case 19:
                          case 20:
                          case 21:
                          case 22:
                            goto LABEL_114;
                          default:
LABEL_146:
                            BUG();
                        }
                    }
                    break;
                  }
                }
                break;
              }
            }
          }
LABEL_25:
          sub_8C7090(59, (__int64)v2);
          return sub_8C99B0(*v2, 1);
        }
        v7 = v74;
        v69 = *(_BYTE *)(v74 + 80);
        v75 = (v69 - 19) & 0xFD;
        v8 = *(_QWORD *)(*(_QWORD *)v7 + 32LL);
        if ( !v8 )
          goto LABEL_25;
        v9 = v7;
        while ( 1 )
        {
          v10 = sub_880F80(v9);
          if ( *(_DWORD *)(v8 + 40) == -1 || v10 == sub_880F80(v8) || !(unsigned int)sub_8C6B40(v8) )
            goto LABEL_24;
          if ( a2 ? !sub_8C6230(v8, v9) : (unsigned int)sub_8C7F70(v8, v9) == 0 )
            goto LABEL_24;
          v12 = *(_BYTE *)(v8 + 80);
          if ( (unsigned __int8)(v12 - 19) > 3u )
            break;
          if ( (v69 == 19) != (v12 == 19) || (v69 == 21) != (v12 == 21) )
            goto LABEL_22;
          if ( v69 == 19 )
          {
            v26 = (__int64 *)sub_8C8620(v9, v8);
            goto LABEL_45;
          }
          if ( v69 != 21 )
          {
            switch ( *(_BYTE *)(v9 + 80) )
            {
              case 4:
              case 5:
                v13 = v8;
                goto LABEL_53;
              case 6:
                v13 = v8;
                goto LABEL_55;
              case 9:
              case 0xA:
                v13 = v8;
                goto LABEL_51;
              case 0x13:
              case 0x14:
              case 0x15:
              case 0x16:
                v13 = v8;
                goto LABEL_30;
              default:
                goto LABEL_63;
            }
          }
          switch ( *(_BYTE *)(v9 + 80) )
          {
            case 4:
            case 5:
              v29 = *(_QWORD *)(*(_QWORD *)(v9 + 96) + 80LL);
              break;
            case 6:
              v29 = *(_QWORD *)(*(_QWORD *)(v9 + 96) + 32LL);
              break;
            case 9:
            case 0xA:
              v29 = *(_QWORD *)(*(_QWORD *)(v9 + 96) + 56LL);
              break;
            case 0x13:
            case 0x14:
            case 0x15:
            case 0x16:
              v29 = *(_QWORD *)(v9 + 88);
              break;
            default:
              v29 = 0;
              break;
          }
          v30 = *(_QWORD *)(v29 + 152);
          v31 = *(_QWORD *)(v8 + 88);
          if ( v30 )
          {
            switch ( *(_BYTE *)(v30 + 80) )
            {
              case 4:
              case 5:
                v32 = *(_QWORD *)(*(_QWORD *)(v30 + 96) + 80LL);
                break;
              case 6:
                v32 = *(_QWORD *)(*(_QWORD *)(v30 + 96) + 32LL);
                break;
              case 9:
              case 0xA:
                v32 = *(_QWORD *)(*(_QWORD *)(v30 + 96) + 56LL);
                break;
              case 0x13:
              case 0x14:
              case 0x15:
              case 0x16:
                v32 = *(_QWORD *)(v30 + 88);
                break;
              default:
                BUG();
            }
            v33 = *(_QWORD *)(v32 + 104);
            v34 = *(_QWORD *)(v31 + 104);
            if ( v33 == v34
              || (v71 = v29, *qword_4D03FD0) && v33 && v34 && (v35 = sub_8C7EB0(v33, v34, 0x3Bu), v29 = v71, v35) )
            {
              v36 = *(_QWORD *)(v31 + 144);
              if ( v36 )
              {
                v72 = v8;
                v37 = v29;
                while ( 1 )
                {
                  switch ( *(_BYTE *)(v36 + 80) )
                  {
                    case 4:
                    case 5:
                      v38 = *(_QWORD *)(*(_QWORD *)(v36 + 96) + 80LL);
                      break;
                    case 6:
                      v38 = *(_QWORD *)(*(_QWORD *)(v36 + 96) + 32LL);
                      break;
                    case 9:
                    case 0xA:
                      v38 = *(_QWORD *)(*(_QWORD *)(v36 + 96) + 56LL);
                      break;
                    case 0x13:
                    case 0x14:
                    case 0x15:
                    case 0x16:
                      v38 = *(_QWORD *)(v36 + 88);
                      break;
                    default:
                      BUG();
                  }
                  if ( (unsigned int)sub_89B3C0(
                                       **(_QWORD **)(v38 + 32),
                                       **(_QWORD **)(v37 + 32),
                                       0,
                                       0,
                                       (_DWORD *)(v9 + 48),
                                       8u)
                    && sub_89AB40(
                         **(_QWORD **)(*(_QWORD *)(v37 + 192) + 216LL),
                         **(_QWORD **)(*(_QWORD *)(v38 + 192) + 216LL),
                         0,
                         *(_QWORD *)(*(_QWORD *)(v38 + 192) + 216LL),
                         v39) )
                  {
                    break;
                  }
                  v36 = *(_QWORD *)(v36 + 8);
                  if ( !v36 )
                  {
                    v8 = v72;
                    goto LABEL_24;
                  }
                }
                v40 = v38;
                v8 = v72;
                v26 = *(__int64 **)(v40 + 104);
LABEL_45:
                if ( !v26 )
                  goto LABEL_24;
                v27 = *v26;
LABEL_47:
                if ( v27 )
                {
                  if ( (unsigned int)*(unsigned __int8 *)(v27 + 80) - 19 <= 3 )
                  {
                    v28 = *(_QWORD *)(*(_QWORD *)(v27 + 88) + 104LL);
                    sub_8CBB20(59, v2, v28);
                    return sub_8CAA20(v2, v28);
                  }
LABEL_145:
                  sub_721090();
                }
                goto LABEL_25;
              }
            }
          }
          else if ( (unsigned int)sub_89B3C0(
                                    **(_QWORD **)(v31 + 32),
                                    **(_QWORD **)(v29 + 32),
                                    1,
                                    0,
                                    (_DWORD *)(v9 + 48),
                                    8u) )
          {
            v26 = *(__int64 **)(v31 + 104);
            goto LABEL_45;
          }
LABEL_24:
          v8 = *(_QWORD *)(v8 + 8);
          if ( !v8 )
            goto LABEL_25;
        }
        if ( v12 == 17 )
        {
          if ( v75 )
          {
            v13 = *(_QWORD *)(v8 + 88);
            switch ( *(_BYTE *)(v9 + 80) )
            {
              case 4:
              case 5:
LABEL_53:
                v14 = *(_QWORD *)(*(_QWORD *)(v9 + 96) + 80LL);
                goto LABEL_31;
              case 6:
LABEL_55:
                v14 = *(_QWORD *)(*(_QWORD *)(v9 + 96) + 32LL);
                goto LABEL_31;
              case 9:
              case 0xA:
LABEL_51:
                v14 = *(_QWORD *)(*(_QWORD *)(v9 + 96) + 56LL);
                goto LABEL_31;
              case 0x13:
              case 0x14:
              case 0x15:
              case 0x16:
LABEL_30:
                v14 = *(_QWORD *)(v9 + 88);
LABEL_31:
                v70 = *(_QWORD *)(v14 + 176);
                if ( !v13 )
                  goto LABEL_24;
                v66 = v2;
                v15 = v13;
                v67 = v8;
                break;
              default:
                goto LABEL_146;
            }
            while ( 1 )
            {
              if ( *(_BYTE *)(v15 + 80) == 20 )
              {
                v16 = *(_QWORD *)(v15 + 88);
                v17 = *(_BYTE *)(v9 + 80);
                v18 = *(_QWORD *)(v9 + 88);
                v19 = *(_QWORD *)(v16 + 176);
                if ( v17 == 20 )
                {
                  v21 = **(_QWORD **)(v18 + 328);
                }
                else
                {
                  v20 = v17 == 21 ? *(__int64 **)(v18 + 232) : *(__int64 **)(v18 + 32);
                  v21 = *v20;
                }
                if ( (unsigned int)sub_89B3C0(**(_QWORD **)(v16 + 328), v21, 0, 0, (_DWORD *)(v9 + 48), 8u) )
                {
                  v24 = *(_QWORD *)(v19 + 152);
                  v25 = *(_QWORD *)(v70 + 152);
                  if ( (v25 == v24 || (unsigned int)sub_8D97D0(v25, v24, 256, v22, v23))
                    && sub_89AB40(*(_QWORD *)(v70 + 240), *(_QWORD *)(v19 + 240), 0, v22, v23) )
                  {
                    break;
                  }
                }
              }
              if ( v12 == 17 )
              {
                v15 = *(_QWORD *)(v15 + 8);
                if ( v15 )
                  continue;
              }
              v8 = v67;
              v2 = v66;
              goto LABEL_24;
            }
            v8 = v67;
            v2 = v66;
            v26 = *(__int64 **)(v16 + 104);
            goto LABEL_45;
          }
          return sub_8C6700(v2, (unsigned int *)(v8 + 48), 0x42Au, 0x425u);
        }
LABEL_22:
        if ( !v75 || (unsigned __int8)(v12 - 10) > 1u )
          return sub_8C6700(v2, (unsigned int *)(v8 + 48), 0x42Au, 0x425u);
        goto LABEL_24;
      }
    }
  }
  return result;
}
