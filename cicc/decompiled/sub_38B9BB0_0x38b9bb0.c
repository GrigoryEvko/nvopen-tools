// Function: sub_38B9BB0
// Address: 0x38b9bb0
//
void __fastcall sub_38B9BB0(unsigned __int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v3; // r15
  __int64 v4; // rbx
  _BOOL4 v5; // r12d
  __int64 v6; // r13
  const char *v7; // rax
  __int64 v8; // rdx
  int v9; // ecx
  char v10; // r8
  __int64 v11; // rax
  unsigned __int16 v12; // ax
  unsigned int v13; // esi
  __int64 v14; // r8
  unsigned int v15; // r14d
  __int64 v16; // rdi
  __int64 *v17; // rax
  __int64 v18; // rcx
  int v19; // ecx
  _BYTE *v20; // rax
  __int64 v21; // rax
  int v22; // eax
  __int64 *v23; // r14
  __int64 *v24; // rdx
  __int64 v25; // r12
  __int64 v26; // rbx
  unsigned __int64 v27; // r8
  __int64 v28; // rax
  __int64 v29; // r12
  _BYTE *v30; // rax
  __int64 v31; // rax
  int v32; // eax
  unsigned int v33; // eax
  __int64 v34; // rsi
  __int64 v35; // r10
  unsigned __int64 v36; // r12
  _QWORD *v37; // rax
  int v38; // eax
  unsigned int v39; // r14d
  __int64 v40; // rax
  unsigned int v41; // esi
  int v42; // eax
  __int64 v43; // rax
  unsigned int v44; // eax
  __int64 v45; // rdx
  __int64 v46; // rsi
  unsigned __int64 v47; // rcx
  _QWORD *v48; // rax
  __int64 *v49; // rdx
  __int64 v50; // rax
  unsigned int v51; // esi
  int v52; // eax
  _QWORD *v53; // rax
  unsigned __int64 v54; // rax
  __int64 v55; // rax
  int v56; // r10d
  int v57; // eax
  int v58; // ecx
  int v59; // eax
  int v60; // edi
  __int64 v61; // rsi
  __int64 v62; // rax
  __int64 v63; // r8
  int v64; // r10d
  __int64 *v65; // r9
  int v66; // eax
  int v67; // eax
  __int64 v68; // rdi
  __int64 v69; // r11
  int v70; // r9d
  __int64 *v71; // r8
  __int64 v72; // rsi
  __int64 v73; // [rsp+0h] [rbp-D0h]
  __int64 v74; // [rsp+8h] [rbp-C8h]
  __int64 v75; // [rsp+8h] [rbp-C8h]
  __int64 v76; // [rsp+8h] [rbp-C8h]
  __int64 v77; // [rsp+10h] [rbp-C0h]
  __int64 v78; // [rsp+10h] [rbp-C0h]
  __int64 v79; // [rsp+10h] [rbp-C0h]
  unsigned __int64 v80; // [rsp+10h] [rbp-C0h]
  __int64 v81; // [rsp+18h] [rbp-B8h]
  unsigned __int64 v82; // [rsp+18h] [rbp-B8h]
  unsigned __int64 v83; // [rsp+18h] [rbp-B8h]
  __int64 v84; // [rsp+18h] [rbp-B8h]
  __int64 v85; // [rsp+20h] [rbp-B0h]
  __int64 v86; // [rsp+20h] [rbp-B0h]
  __int64 v87; // [rsp+20h] [rbp-B0h]
  __int64 v88; // [rsp+20h] [rbp-B0h]
  __int64 v89; // [rsp+20h] [rbp-B0h]
  __int64 v90; // [rsp+28h] [rbp-A8h]
  __int64 v91; // [rsp+28h] [rbp-A8h]
  __int64 v92; // [rsp+28h] [rbp-A8h]
  __int64 v93; // [rsp+28h] [rbp-A8h]
  unsigned __int64 v94; // [rsp+30h] [rbp-A0h]
  __int64 v95; // [rsp+30h] [rbp-A0h]
  unsigned __int64 v96; // [rsp+30h] [rbp-A0h]
  unsigned __int64 v97; // [rsp+30h] [rbp-A0h]
  __int64 *v98; // [rsp+38h] [rbp-98h]
  unsigned int v99; // [rsp+44h] [rbp-8Ch]
  __int64 v100; // [rsp+48h] [rbp-88h]
  const char *v101; // [rsp+50h] [rbp-80h] BYREF
  __int64 v102; // [rsp+58h] [rbp-78h]
  __int64 v103; // [rsp+60h] [rbp-70h]
  _QWORD v104[2]; // [rsp+80h] [rbp-50h] BYREF
  __int16 v105; // [rsp+90h] [rbp-40h]

  v3 = a1;
  v4 = a3;
  v100 = a2;
  v5 = (*(_BYTE *)(a3 + 32) & 0xF) == 0;
  v6 = sub_1632FA0(*(_QWORD *)(a3 + 40));
  if ( (*(_BYTE *)(v4 + 23) & 0x20) == 0 )
  {
    v13 = *(_DWORD *)(a1 + 24);
    if ( v13 )
    {
      v14 = *(_QWORD *)(a1 + 8);
      v15 = ((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4);
      v16 = (v13 - 1) & v15;
      v17 = (__int64 *)(v14 + 16 * v16);
      v18 = *v17;
      if ( *v17 == v4 )
      {
LABEL_17:
        v19 = *((_DWORD *)v17 + 2);
        if ( v19 )
        {
LABEL_18:
          LODWORD(v103) = v19;
          v104[0] = "__unnamed_";
          v105 = 2307;
          v104[1] = v103;
          switch ( *(_DWORD *)(v6 + 16) )
          {
            case 0:
            case 1:
            case 3:
            case 5:
              v10 = 0;
              break;
            case 2:
            case 4:
              v10 = 95;
              break;
          }
LABEL_20:
          sub_38B95A0(v100, (__int64)v104, v5, v6, v10);
          return;
        }
        v49 = v17;
LABEL_80:
        v19 = *(_DWORD *)(v3 + 16);
        *((_DWORD *)v49 + 2) = v19;
        goto LABEL_18;
      }
      v56 = 1;
      v49 = 0;
      while ( v18 != -8 )
      {
        if ( v18 == -16 && !v49 )
          v49 = v17;
        LODWORD(v16) = (v13 - 1) & (v56 + v16);
        v17 = (__int64 *)(v14 + 16LL * (unsigned int)v16);
        v18 = *v17;
        if ( *v17 == v4 )
          goto LABEL_17;
        ++v56;
      }
      if ( !v49 )
        v49 = v17;
      v57 = *(_DWORD *)(v3 + 16);
      ++*(_QWORD *)v3;
      v58 = v57 + 1;
      if ( 4 * (v57 + 1) < 3 * v13 )
      {
        if ( v13 - *(_DWORD *)(v3 + 20) - v58 > v13 >> 3 )
        {
LABEL_101:
          *(_DWORD *)(v3 + 16) = v58;
          if ( *v49 != -8 )
            --*(_DWORD *)(v3 + 20);
          *v49 = v4;
          *((_DWORD *)v49 + 2) = 0;
          goto LABEL_80;
        }
        sub_38B99F0(v3, v13);
        v66 = *(_DWORD *)(v3 + 24);
        if ( v66 )
        {
          v67 = v66 - 1;
          v68 = *(_QWORD *)(v3 + 8);
          LODWORD(v69) = v67 & v15;
          v70 = 1;
          v71 = 0;
          v58 = *(_DWORD *)(v3 + 16) + 1;
          v49 = (__int64 *)(v68 + 16LL * (v67 & v15));
          v72 = *v49;
          if ( *v49 != v4 )
          {
            while ( v72 != -8 )
            {
              if ( !v71 && v72 == -16 )
                v71 = v49;
              v69 = v67 & (unsigned int)(v69 + v70);
              v49 = (__int64 *)(v68 + 16 * v69);
              v72 = *v49;
              if ( *v49 == v4 )
                goto LABEL_101;
              ++v70;
            }
            if ( v71 )
              v49 = v71;
          }
          goto LABEL_101;
        }
LABEL_135:
        ++*(_DWORD *)(v3 + 16);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)a1;
    }
    sub_38B99F0(v3, 2 * v13);
    v59 = *(_DWORD *)(v3 + 24);
    if ( v59 )
    {
      v60 = v59 - 1;
      v61 = *(_QWORD *)(v3 + 8);
      LODWORD(v62) = (v59 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v58 = *(_DWORD *)(v3 + 16) + 1;
      v49 = (__int64 *)(v61 + 16LL * (unsigned int)v62);
      v63 = *v49;
      if ( *v49 != v4 )
      {
        v64 = 1;
        v65 = 0;
        while ( v63 != -8 )
        {
          if ( v63 == -16 && !v65 )
            v65 = v49;
          v62 = v60 & (unsigned int)(v62 + v64);
          v49 = (__int64 *)(v61 + 16 * v62);
          v63 = *v49;
          if ( *v49 == v4 )
            goto LABEL_101;
          ++v64;
        }
        if ( v65 )
          v49 = v65;
      }
      goto LABEL_101;
    }
    goto LABEL_135;
  }
  v7 = sub_1649960(v4);
  v102 = v8;
  v9 = *(_DWORD *)(v6 + 16);
  v101 = v7;
  switch ( v9 )
  {
    case 0:
    case 1:
    case 3:
    case 5:
      v10 = 0;
      v11 = v102;
      if ( !*(_BYTE *)(v4 + 16) )
        goto LABEL_4;
      goto LABEL_13;
    case 2:
    case 4:
      v11 = v102;
      v10 = 95;
      if ( *(_BYTE *)(v4 + 16) )
      {
LABEL_13:
        if ( !v11 )
          goto LABEL_11;
        v4 = 0;
      }
      else
      {
LABEL_4:
        if ( !v11 )
          goto LABEL_9;
      }
      if ( *v101 == 1 || (unsigned int)(v9 - 3) <= 1 && *v101 == 63 || !v4 )
        goto LABEL_11;
LABEL_9:
      v12 = (*(_WORD *)(v4 + 18) >> 4) & 0x3FF;
      if ( v9 != 4 )
      {
        if ( v12 != 80 )
        {
LABEL_11:
          v105 = 261;
          v104[0] = &v101;
          goto LABEL_20;
        }
LABEL_23:
        v105 = 261;
        v104[0] = &v101;
        sub_38B95A0(a2, (__int64)v104, v5, v6, 0);
        v20 = *(_BYTE **)(a2 + 24);
        if ( (unsigned __int64)v20 >= *(_QWORD *)(a2 + 16) )
        {
          a2 = 64;
          sub_16E7DE0(v100, 64);
        }
        else
        {
          *(_QWORD *)(a2 + 24) = v20 + 1;
          *v20 = 64;
        }
        goto LABEL_25;
      }
      v39 = v12;
      if ( v12 == 65 )
      {
        a2 = (__int64)v104;
        v105 = 261;
        v104[0] = &v101;
        sub_38B95A0(v100, (__int64)v104, v5, v6, 64);
        goto LABEL_25;
      }
      if ( v12 == 80 )
        goto LABEL_23;
      a2 = (__int64)v104;
      v105 = 261;
      v104[0] = &v101;
      sub_38B95A0(v100, (__int64)v104, v5, v6, v10);
      if ( v39 > 0x41 || v39 <= 0x3F )
        return;
LABEL_25:
      v21 = *(_QWORD *)(v4 + 24);
      if ( !(*(_DWORD *)(v21 + 8) >> 8)
        || (v22 = *(_DWORD *)(v21 + 12), v22 == 1)
        || v22 == 2
        && ((a2 = 0, (unsigned __int8)sub_1560290((_QWORD *)(v4 + 112), 0, 53))
         || (a2 = 1, (unsigned __int8)sub_1560290((_QWORD *)(v4 + 112), 1, 53))) )
      {
        if ( (*(_BYTE *)(v4 + 18) & 1) != 0 )
        {
          sub_15E08E0(v4, a2);
          v23 = *(__int64 **)(v4 + 88);
          if ( (*(_BYTE *)(v4 + 18) & 1) != 0 )
            sub_15E08E0(v4, a2);
          v24 = *(__int64 **)(v4 + 88);
        }
        else
        {
          v23 = *(__int64 **)(v4 + 88);
          v24 = v23;
        }
        v98 = &v24[5 * *(_QWORD *)(v4 + 96)];
        if ( v23 == v98 )
        {
          v29 = 0;
        }
        else
        {
          v99 = 0;
          while ( 2 )
          {
            v25 = *v23;
            if ( (unsigned __int8)sub_15E0300((__int64)v23) )
              v25 = *(_QWORD *)(v25 + 24);
            v26 = 1;
            v3 = (unsigned int)sub_15A9520(v6, 0);
            v27 = (unsigned int)sub_15A9FE0(v6, v25);
LABEL_37:
            switch ( *(_BYTE *)(v25 + 8) )
            {
              case 1:
                v28 = 16;
                goto LABEL_39;
              case 2:
                v28 = 32;
                goto LABEL_39;
              case 3:
              case 9:
                v28 = 64;
                goto LABEL_39;
              case 4:
                v28 = 80;
                goto LABEL_39;
              case 5:
              case 6:
                v28 = 128;
                goto LABEL_39;
              case 7:
                v97 = v27;
                v38 = sub_15A9520(v6, 0);
                v27 = v97;
                v28 = (unsigned int)(8 * v38);
                goto LABEL_39;
              case 0xB:
                v28 = *(_DWORD *)(v25 + 8) >> 8;
                goto LABEL_39;
              case 0xD:
                v96 = v27;
                v37 = (_QWORD *)sub_15A9930(v6, v25);
                v27 = v96;
                v28 = 8LL * *v37;
                goto LABEL_39;
              case 0xE:
                v85 = v27;
                v90 = *(_QWORD *)(v25 + 24);
                v95 = *(_QWORD *)(v25 + 32);
                v33 = sub_15A9FE0(v6, v90);
                v27 = v85;
                v34 = v90;
                v35 = 1;
                v36 = v33;
                while ( 2 )
                {
                  switch ( *(_BYTE *)(v34 + 8) )
                  {
                    case 1:
                      v40 = 16;
                      goto LABEL_66;
                    case 2:
                      v40 = 32;
                      goto LABEL_66;
                    case 3:
                    case 9:
                      v40 = 64;
                      goto LABEL_66;
                    case 4:
                      v40 = 80;
                      goto LABEL_66;
                    case 5:
                    case 6:
                      v40 = 128;
                      goto LABEL_66;
                    case 7:
                      v41 = 0;
                      v91 = v35;
                      goto LABEL_70;
                    case 0xB:
                      v40 = *(_DWORD *)(v34 + 8) >> 8;
                      goto LABEL_66;
                    case 0xD:
                      v93 = v35;
                      v48 = (_QWORD *)sub_15A9930(v6, v34);
                      v35 = v93;
                      v27 = v85;
                      v40 = 8LL * *v48;
                      goto LABEL_66;
                    case 0xE:
                      v77 = v85;
                      v81 = v35;
                      v86 = *(_QWORD *)(v34 + 24);
                      v92 = *(_QWORD *)(v34 + 32);
                      v44 = sub_15A9FE0(v6, v86);
                      v27 = v77;
                      v45 = 1;
                      v46 = v86;
                      v35 = v81;
                      v47 = v44;
                      while ( 2 )
                      {
                        switch ( *(_BYTE *)(v46 + 8) )
                        {
                          case 1:
                            v50 = 16;
                            goto LABEL_82;
                          case 2:
                            v50 = 32;
                            goto LABEL_82;
                          case 3:
                          case 9:
                            v50 = 64;
                            goto LABEL_82;
                          case 4:
                            v50 = 80;
                            goto LABEL_82;
                          case 5:
                          case 6:
                            v50 = 128;
                            goto LABEL_82;
                          case 7:
                            v74 = v77;
                            v51 = 0;
                            v78 = v81;
                            v82 = v47;
                            v87 = v45;
                            goto LABEL_85;
                          case 0xB:
                            v50 = *(_DWORD *)(v46 + 8) >> 8;
                            goto LABEL_82;
                          case 0xD:
                            v75 = v77;
                            v79 = v81;
                            v83 = v47;
                            v88 = v45;
                            v53 = (_QWORD *)sub_15A9930(v6, v46);
                            v45 = v88;
                            v47 = v83;
                            v35 = v79;
                            v27 = v75;
                            v50 = 8LL * *v53;
                            goto LABEL_82;
                          case 0xE:
                            v73 = v77;
                            v76 = v81;
                            v80 = v47;
                            v84 = v45;
                            v89 = *(_QWORD *)(v46 + 32);
                            v54 = sub_12BE0A0(v6, *(_QWORD *)(v46 + 24));
                            v45 = v84;
                            v47 = v80;
                            v35 = v76;
                            v27 = v73;
                            v50 = 8 * v89 * v54;
                            goto LABEL_82;
                          case 0xF:
                            v74 = v77;
                            v78 = v81;
                            v82 = v47;
                            v51 = *(_DWORD *)(v46 + 8) >> 8;
                            v87 = v45;
LABEL_85:
                            v52 = sub_15A9520(v6, v51);
                            v45 = v87;
                            v47 = v82;
                            v35 = v78;
                            v27 = v74;
                            v50 = (unsigned int)(8 * v52);
LABEL_82:
                            v40 = 8 * v47 * v92 * ((v47 + ((unsigned __int64)(v50 * v45 + 7) >> 3) - 1) / v47);
                            goto LABEL_66;
                          case 0x10:
                            v55 = *(_QWORD *)(v46 + 32);
                            v46 = *(_QWORD *)(v46 + 24);
                            v45 *= v55;
                            continue;
                          default:
                            goto LABEL_135;
                        }
                      }
                    case 0xF:
                      v91 = v35;
                      v41 = *(_DWORD *)(v34 + 8) >> 8;
LABEL_70:
                      v42 = sub_15A9520(v6, v41);
                      v35 = v91;
                      v27 = v85;
                      v40 = (unsigned int)(8 * v42);
LABEL_66:
                      v28 = 8 * v36 * v95 * ((v36 + ((unsigned __int64)(v40 * v35 + 7) >> 3) - 1) / v36);
                      goto LABEL_39;
                    case 0x10:
                      v43 = *(_QWORD *)(v34 + 32);
                      v34 = *(_QWORD *)(v34 + 24);
                      v35 *= v43;
                      continue;
                    default:
                      goto LABEL_135;
                  }
                }
              case 0xF:
                v94 = v27;
                v32 = sub_15A9520(v6, *(_DWORD *)(v25 + 8) >> 8);
                v27 = v94;
                v28 = (unsigned int)(8 * v32);
LABEL_39:
                v23 += 5;
                v99 += v3 * ((v3 + v27 * ((v27 + ((unsigned __int64)(v28 * v26 + 7) >> 3) - 1) / v27) - 1) / v3);
                if ( v23 != v98 )
                  continue;
                v29 = v99;
                break;
              case 0x10:
                v31 = *(_QWORD *)(v25 + 32);
                v25 = *(_QWORD *)(v25 + 24);
                v26 *= v31;
                goto LABEL_37;
              default:
                goto LABEL_135;
            }
            break;
          }
        }
        v30 = *(_BYTE **)(v100 + 24);
        if ( (unsigned __int64)v30 >= *(_QWORD *)(v100 + 16) )
        {
          v100 = sub_16E7DE0(v100, 64);
        }
        else
        {
          *(_QWORD *)(v100 + 24) = v30 + 1;
          *v30 = 64;
        }
        sub_16E7A90(v100, v29);
      }
      break;
    default:
      goto LABEL_135;
  }
}
