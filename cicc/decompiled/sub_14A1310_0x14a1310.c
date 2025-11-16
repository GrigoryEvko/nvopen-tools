// Function: sub_14A1310
// Address: 0x14a1310
//
_BOOL8 __fastcall sub_14A1310(__int64 *a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5)
{
  __int64 v9; // rsi
  unsigned __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r14
  __int64 v13; // r12
  unsigned __int64 v14; // r15
  unsigned __int64 v15; // r13
  __int64 v16; // rbx
  __int64 v17; // r8
  unsigned int v18; // eax
  __int64 v19; // r8
  __int64 v20; // rcx
  unsigned __int64 v21; // r10
  __int64 v22; // rax
  unsigned __int64 v23; // r10
  unsigned int v24; // eax
  char v25; // al
  _BOOL4 v26; // r12d
  __int64 v28; // rsi
  __int64 v29; // rax
  unsigned __int64 v30; // r10
  _QWORD *v31; // rbx
  __int64 v32; // rax
  unsigned __int64 v33; // r10
  __int64 v34; // rax
  __int64 v35; // rax
  __int64 v36; // rax
  int v37; // eax
  __int64 v38; // rdi
  unsigned int v39; // eax
  __int64 v40; // r9
  unsigned __int64 v41; // r11
  __int64 v42; // rdi
  _QWORD *v43; // rax
  __int64 v44; // rax
  __int64 v45; // rax
  int v46; // eax
  __int64 v47; // rdi
  _QWORD *v48; // rax
  unsigned int v49; // eax
  unsigned __int64 v50; // rax
  int v51; // eax
  __int64 v52; // rdi
  __int64 v53; // rax
  __int64 v54; // rax
  __int64 v55; // [rsp+8h] [rbp-E8h]
  __int64 v56; // [rsp+10h] [rbp-E0h]
  __int64 v57; // [rsp+18h] [rbp-D8h]
  __int64 v58; // [rsp+18h] [rbp-D8h]
  unsigned __int64 v59; // [rsp+20h] [rbp-D0h]
  __int64 v60; // [rsp+20h] [rbp-D0h]
  __int64 v61; // [rsp+28h] [rbp-C8h]
  __int64 v62; // [rsp+28h] [rbp-C8h]
  __int64 v63; // [rsp+28h] [rbp-C8h]
  __int64 v64; // [rsp+30h] [rbp-C0h]
  unsigned __int64 v65; // [rsp+30h] [rbp-C0h]
  unsigned __int64 v66; // [rsp+30h] [rbp-C0h]
  __int64 v67; // [rsp+38h] [rbp-B8h]
  __int64 v68; // [rsp+38h] [rbp-B8h]
  __int64 v69; // [rsp+38h] [rbp-B8h]
  __int64 v70; // [rsp+38h] [rbp-B8h]
  __int64 v71; // [rsp+40h] [rbp-B0h]
  unsigned __int64 v72; // [rsp+40h] [rbp-B0h]
  __int64 v73; // [rsp+48h] [rbp-A8h]
  unsigned __int64 v74; // [rsp+48h] [rbp-A8h]
  unsigned __int64 v75; // [rsp+48h] [rbp-A8h]
  __int64 v76; // [rsp+48h] [rbp-A8h]
  __int64 v77; // [rsp+50h] [rbp-A0h]
  unsigned int v78; // [rsp+5Ch] [rbp-94h]
  unsigned __int64 v79; // [rsp+60h] [rbp-90h]
  unsigned __int64 v80; // [rsp+68h] [rbp-88h]
  unsigned __int64 v81; // [rsp+68h] [rbp-88h]
  __int64 v82; // [rsp+68h] [rbp-88h]
  unsigned __int64 v83; // [rsp+68h] [rbp-88h]
  __int64 v84; // [rsp+68h] [rbp-88h]
  __int64 v85; // [rsp+68h] [rbp-88h]
  __int64 v86; // [rsp+68h] [rbp-88h]
  __int64 v87; // [rsp+70h] [rbp-80h]
  unsigned __int64 v88; // [rsp+70h] [rbp-80h]
  unsigned __int64 v89; // [rsp+70h] [rbp-80h]
  __int64 v90; // [rsp+70h] [rbp-80h]
  __int64 v91; // [rsp+70h] [rbp-80h]
  __int64 v92; // [rsp+70h] [rbp-80h]
  __int64 v93; // [rsp+70h] [rbp-80h]
  __int64 v94; // [rsp+78h] [rbp-78h]
  __int64 v96; // [rsp+88h] [rbp-68h]
  __int64 v97; // [rsp+90h] [rbp-60h] BYREF
  unsigned int v98; // [rsp+98h] [rbp-58h]
  _QWORD *v99; // [rsp+A0h] [rbp-50h] BYREF
  unsigned int v100; // [rsp+A8h] [rbp-48h]
  _QWORD *v101; // [rsp+B0h] [rbp-40h] BYREF
  unsigned int v102; // [rsp+B8h] [rbp-38h]

  if ( !a3 || (v77 = sub_1649C60(a3), *(_BYTE *)(v77 + 16) > 3u) )
    v77 = 0;
  v9 = *a3;
  v78 = sub_15A9570(*a1, *a3);
  v98 = v78;
  if ( v78 <= 0x40 )
  {
    v97 = 0;
    if ( a5 )
      goto LABEL_5;
LABEL_77:
    v26 = v77 != 0;
    goto LABEL_30;
  }
  v9 = 0;
  sub_16A4EF0(&v97, 0, 0);
  if ( !a5 )
    goto LABEL_77;
LABEL_5:
  v94 = a4 + 8 * a5;
  if ( a4 != v94 )
  {
    v79 = 0;
    v12 = a4 + 8;
    v11 = a2 | 4;
    while ( 1 )
    {
      v96 = v12;
      v13 = *(_QWORD *)(v12 - 8);
      v14 = v11 & 0xFFFFFFFFFFFFFFF8LL;
      v15 = v11 & 0xFFFFFFFFFFFFFFF8LL;
      v16 = (v11 >> 2) & 1;
      if ( ((v11 >> 2) & 1) != 0 )
        break;
      v28 = *(_QWORD *)(v12 - 8);
      v89 = v11 & 0xFFFFFFFFFFFFFFF8LL;
      sub_1643D30(v11 & 0xFFFFFFFFFFFFFFF8LL, v28);
      v13 = *(_QWORD *)(v12 - 8);
      v30 = v89;
      if ( *(_BYTE *)(v13 + 16) != 13 )
        goto LABEL_39;
      v17 = *a1;
      if ( !v14 )
      {
LABEL_51:
        v28 = *(_QWORD *)(v12 - 8);
        goto LABEL_52;
      }
LABEL_44:
      v31 = *(_QWORD **)(v13 + 24);
      if ( *(_DWORD *)(v13 + 32) > 0x40u )
        v31 = (_QWORD *)*v31;
      v88 = v30;
      v32 = sub_15A9930(v17, v14);
      sub_16A7490(&v97, *(_QWORD *)(v32 + 8LL * (unsigned int)v31 + 16));
      v33 = v88;
LABEL_47:
      v9 = *(_QWORD *)(v12 - 8);
      v15 = sub_1643D30(v33, v9);
LABEL_24:
      v25 = *(_BYTE *)(v15 + 8);
      if ( ((v25 - 14) & 0xFD) != 0 )
      {
        v11 = 0;
        if ( v25 == 13 )
          v11 = v15;
      }
      else
      {
        v11 = *(_QWORD *)(v15 + 24) | 4LL;
      }
      v12 += 8;
      if ( v94 == v96 )
        goto LABEL_27;
    }
    if ( v14 )
    {
      if ( *(_BYTE *)(v13 + 16) == 13
        || (v35 = sub_14C49D0(*(_QWORD *)(v12 - 8), v9), (v13 = v35) == 0)
        || *(_BYTE *)(v35 + 16) == 13 )
      {
        v17 = *a1;
      }
      else
      {
        v13 = 0;
LABEL_58:
        v17 = *a1;
        if ( !v14 )
        {
          v28 = *(_QWORD *)(v12 - 8);
LABEL_60:
          v30 = 0;
LABEL_52:
          v90 = v17;
          v34 = sub_1643D30(v30, v28);
          v17 = v90;
          v9 = v34;
LABEL_12:
          v87 = v17;
          v18 = sub_15A9FE0(v17, v9);
          v19 = v87;
          v20 = 1;
          v21 = v18;
          while ( 2 )
          {
            switch ( *(_BYTE *)(v9 + 8) )
            {
              case 0:
              case 8:
              case 0xA:
              case 0xC:
              case 0x10:
                v36 = *(_QWORD *)(v9 + 32);
                v9 = *(_QWORD *)(v9 + 24);
                v20 *= v36;
                continue;
              case 1:
                v22 = 16;
                goto LABEL_15;
              case 2:
                v22 = 32;
                goto LABEL_15;
              case 3:
              case 9:
                v22 = 64;
                goto LABEL_15;
              case 4:
                v22 = 80;
                goto LABEL_15;
              case 5:
              case 6:
                v22 = 128;
                goto LABEL_15;
              case 7:
                v81 = v21;
                v9 = 0;
                v91 = v20;
                goto LABEL_66;
              case 0xB:
                v22 = *(_DWORD *)(v9 + 8) >> 8;
                goto LABEL_15;
              case 0xD:
                v42 = v87;
                v83 = v21;
                v93 = v20;
                v43 = (_QWORD *)sub_15A9930(v42, v9);
                v20 = v93;
                v21 = v83;
                v22 = 8LL * *v43;
                goto LABEL_15;
              case 0xE:
                v38 = v87;
                v67 = v21;
                v71 = v20;
                v73 = *(_QWORD *)(v9 + 24);
                v82 = v87;
                v92 = *(_QWORD *)(v9 + 32);
                v39 = sub_15A9FE0(v38, v73);
                v21 = v67;
                v9 = v73;
                v40 = 1;
                v20 = v71;
                v19 = v82;
                v41 = v39;
                while ( 2 )
                {
                  switch ( *(_BYTE *)(v9 + 8) )
                  {
                    case 0:
                    case 8:
                    case 0xA:
                    case 0xC:
                    case 0x10:
                      v45 = *(_QWORD *)(v9 + 32);
                      v9 = *(_QWORD *)(v9 + 24);
                      v40 *= v45;
                      continue;
                    case 1:
                      v44 = 16;
                      goto LABEL_84;
                    case 2:
                      v44 = 32;
                      goto LABEL_84;
                    case 3:
                    case 9:
                      v44 = 64;
                      goto LABEL_84;
                    case 4:
                      v44 = 80;
                      goto LABEL_84;
                    case 5:
                    case 6:
                      v44 = 128;
                      goto LABEL_84;
                    case 7:
                      v9 = 0;
                      v74 = v41;
                      v84 = v40;
                      goto LABEL_89;
                    case 0xB:
                      v44 = *(_DWORD *)(v9 + 8) >> 8;
                      goto LABEL_84;
                    case 0xD:
                      v47 = v82;
                      v75 = v41;
                      v85 = v40;
                      v48 = (_QWORD *)sub_15A9930(v47, v9);
                      v40 = v85;
                      v41 = v75;
                      v20 = v71;
                      v21 = v67;
                      v44 = 8LL * *v48;
                      goto LABEL_84;
                    case 0xE:
                      v56 = v67;
                      v57 = v71;
                      v59 = v41;
                      v61 = v40;
                      v64 = *(_QWORD *)(v9 + 24);
                      v68 = v82;
                      v76 = *(_QWORD *)(v9 + 32);
                      v49 = sub_15A9FE0(v82, v64);
                      v21 = v56;
                      v86 = 1;
                      v20 = v71;
                      v41 = v59;
                      v72 = v49;
                      v9 = v64;
                      v40 = v61;
                      v19 = v68;
                      while ( 2 )
                      {
                        switch ( *(_BYTE *)(v9 + 8) )
                        {
                          case 0:
                          case 8:
                          case 0xA:
                          case 0xC:
                          case 0x10:
                            v54 = v86 * *(_QWORD *)(v9 + 32);
                            v9 = *(_QWORD *)(v9 + 24);
                            v86 = v54;
                            continue;
                          case 1:
                            v50 = 16;
                            goto LABEL_98;
                          case 2:
                            v50 = 32;
                            goto LABEL_98;
                          case 3:
                          case 9:
                            v50 = 64;
                            goto LABEL_98;
                          case 4:
                            v50 = 80;
                            goto LABEL_98;
                          case 5:
                          case 6:
                            v50 = 128;
                            goto LABEL_98;
                          case 7:
                            v60 = v56;
                            v9 = 0;
                            v62 = v57;
                            v65 = v41;
                            v69 = v40;
                            goto LABEL_103;
                          case 0xB:
                            v50 = *(_DWORD *)(v9 + 8) >> 8;
                            goto LABEL_98;
                          case 0xD:
                            JUMPOUT(0x14A1C3E);
                          case 0xE:
                            v52 = v68;
                            v58 = v61;
                            v70 = *(_QWORD *)(v9 + 32);
                            v55 = v20;
                            v63 = v19;
                            v9 = *(_QWORD *)(v9 + 24);
                            v66 = (unsigned int)sub_15A9FE0(v52, v9);
                            v53 = sub_127FA20(v63, v9);
                            v40 = v58;
                            v41 = v59;
                            v21 = v56;
                            v20 = v55;
                            v50 = 8 * v66 * v70 * ((v66 + ((unsigned __int64)(v53 + 7) >> 3) - 1) / v66);
                            goto LABEL_98;
                          case 0xF:
                            v60 = v56;
                            v62 = v57;
                            v65 = v41;
                            v9 = *(_DWORD *)(v9 + 8) >> 8;
                            v69 = v40;
LABEL_103:
                            v51 = sub_15A9520(v19, v9);
                            v40 = v69;
                            v41 = v65;
                            v20 = v62;
                            v21 = v60;
                            v50 = (unsigned int)(8 * v51);
LABEL_98:
                            v44 = 8 * v72 * v76 * ((v72 + ((v86 * v50 + 7) >> 3) - 1) / v72);
                            break;
                        }
                        goto LABEL_84;
                      }
                    case 0xF:
                      v74 = v41;
                      v9 = *(_DWORD *)(v9 + 8) >> 8;
                      v84 = v40;
LABEL_89:
                      v46 = sub_15A9520(v19, v9);
                      v40 = v84;
                      v41 = v74;
                      v20 = v71;
                      v21 = v67;
                      v44 = (unsigned int)(8 * v46);
LABEL_84:
                      v22 = 8 * v92 * v41 * ((v41 + ((unsigned __int64)(v40 * v44 + 7) >> 3) - 1) / v41);
                      break;
                  }
                  goto LABEL_15;
                }
              case 0xF:
                v81 = v21;
                v91 = v20;
                v9 = *(_DWORD *)(v9 + 8) >> 8;
LABEL_66:
                v37 = sub_15A9520(v19, v9);
                v20 = v91;
                v21 = v81;
                v22 = (unsigned int)(8 * v37);
LABEL_15:
                v10 = (unsigned __int64)(v22 * v20 + 7) >> 3;
                v23 = (v21 + v10 - 1) / v21 * v21;
                if ( v13 )
                {
                  v80 = v23;
                  sub_16A5D70(&v99, v13 + 24, v78, v10, v19);
                  sub_16A7A10(&v99, v80);
                  v24 = v100;
                  v9 = (__int64)&v101;
                  v100 = 0;
                  v102 = v24;
                  v101 = v99;
                  sub_16A7200(&v97, &v101);
                  if ( v102 > 0x40 && v101 )
                    j_j___libc_free_0_0(v101);
                  if ( v100 > 0x40 && v99 )
                    j_j___libc_free_0_0(v99);
                }
                else
                {
                  if ( v79 )
                    goto LABEL_29;
                  v79 = v23;
                }
                if ( (_BYTE)v16 )
                {
                  if ( v14 )
                    goto LABEL_24;
                  v33 = 0;
                }
                else
                {
                  v33 = v14;
                }
                break;
            }
            goto LABEL_47;
          }
        }
      }
      v9 = v14;
      goto LABEL_12;
    }
    sub_1643D30(0, *(_QWORD *)(v12 - 8));
    v13 = *(_QWORD *)(v12 - 8);
    v28 = v13;
    if ( *(_BYTE *)(v13 + 16) == 13 )
    {
      v17 = *a1;
      goto LABEL_60;
    }
LABEL_39:
    v29 = sub_14C49D0(v13, v28);
    v13 = v29;
    if ( v29 && *(_BYTE *)(v29 + 16) != 13 )
      v13 = 0;
    if ( (_BYTE)v16 )
      goto LABEL_58;
    v30 = v14;
    v17 = *a1;
    if ( !v14 )
      goto LABEL_51;
    goto LABEL_44;
  }
  v79 = 0;
LABEL_27:
  sub_16A5D70(&v101, &v97, 64, v10, v11);
  if ( v102 > 0x40 )
  {
    if ( *v101 | v77 )
    {
      v26 = 1;
    }
    else
    {
      v26 = v79 > 1;
      if ( !v101 )
        goto LABEL_30;
    }
    j_j___libc_free_0_0(v101);
  }
  else if ( ((__int64)((_QWORD)v101 << (64 - (unsigned __int8)v102)) >> (64 - (unsigned __int8)v102)) | v77
         || (v26 = 0, v79 > 1) )
  {
LABEL_29:
    v26 = 1;
  }
LABEL_30:
  if ( v98 > 0x40 && v97 )
    j_j___libc_free_0_0(v97);
  return v26;
}
