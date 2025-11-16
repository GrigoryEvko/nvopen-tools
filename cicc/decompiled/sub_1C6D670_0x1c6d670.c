// Function: sub_1C6D670
// Address: 0x1c6d670
//
__int64 __fastcall sub_1C6D670(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r12
  int v4; // ebx
  char *v6; // rax
  char v7; // al
  __int64 *v8; // r13
  __int64 *v9; // rdx
  unsigned int v10; // ebx
  __int64 v11; // r14
  __int64 v12; // r15
  __int64 v13; // rsi
  unsigned __int64 v14; // r14
  __int64 v15; // rax
  __int64 v16; // r13
  unsigned __int64 v17; // rax
  __int64 v18; // rax
  unsigned int v19; // eax
  __int64 v20; // rsi
  __int64 v21; // rdx
  unsigned __int64 v22; // r10
  unsigned int v23; // eax
  __int64 v24; // r9
  unsigned __int64 v25; // r15
  __int64 v26; // rax
  int v27; // eax
  __int64 v28; // rax
  int v29; // eax
  unsigned int v30; // eax
  __int64 v31; // rsi
  __int64 v32; // rdx
  unsigned __int64 v33; // r14
  _QWORD *v34; // rax
  unsigned __int64 v35; // rdx
  __int64 v36; // rcx
  const char *v37; // r9
  size_t v38; // r8
  char *v39; // rcx
  unsigned __int64 v40; // rax
  const char *v41; // rdx
  __int64 v42; // rdx
  __int64 v43; // rbx
  unsigned int v44; // eax
  unsigned int v45; // ecx
  unsigned __int64 v46; // rsi
  __int64 v47; // rax
  unsigned int v48; // esi
  int v49; // eax
  _QWORD *v50; // rax
  unsigned __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rax
  unsigned int v55; // esi
  int v56; // eax
  _QWORD *v57; // rax
  __int64 v58; // rax
  unsigned __int64 v59; // rax
  unsigned __int64 v60; // rax
  _QWORD *v61; // rax
  int v62; // eax
  __int64 v63; // rdx
  int v64; // eax
  const char *v65; // rax
  char *v66; // rdi
  unsigned __int64 v67; // [rsp+8h] [rbp-D8h]
  __int64 v68; // [rsp+8h] [rbp-D8h]
  __int64 v69; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v70; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v71; // [rsp+10h] [rbp-D0h]
  __int64 v72; // [rsp+10h] [rbp-D0h]
  __int64 v73; // [rsp+10h] [rbp-D0h]
  __int64 v74; // [rsp+18h] [rbp-C8h]
  __int64 v75; // [rsp+18h] [rbp-C8h]
  __int64 v76; // [rsp+18h] [rbp-C8h]
  __int64 v77; // [rsp+18h] [rbp-C8h]
  __int64 v78; // [rsp+18h] [rbp-C8h]
  __int64 v79; // [rsp+18h] [rbp-C8h]
  __int64 v80; // [rsp+18h] [rbp-C8h]
  __int64 v81; // [rsp+18h] [rbp-C8h]
  __int64 v82; // [rsp+28h] [rbp-B8h]
  size_t n; // [rsp+38h] [rbp-A8h]
  size_t na; // [rsp+38h] [rbp-A8h]
  void *src; // [rsp+40h] [rbp-A0h]
  void *srcb; // [rsp+40h] [rbp-A0h]
  void *srcc; // [rsp+40h] [rbp-A0h]
  void *srca; // [rsp+40h] [rbp-A0h]
  void *srcd; // [rsp+40h] [rbp-A0h]
  unsigned int srce; // [rsp+40h] [rbp-A0h]
  const char *srcf; // [rsp+40h] [rbp-A0h]
  __int64 *v92; // [rsp+48h] [rbp-98h]
  __int64 v93; // [rsp+48h] [rbp-98h]
  unsigned int v94; // [rsp+48h] [rbp-98h]
  unsigned int v95; // [rsp+48h] [rbp-98h]
  unsigned int v96; // [rsp+48h] [rbp-98h]
  unsigned __int8 v97; // [rsp+50h] [rbp-90h]
  __int64 v98; // [rsp+58h] [rbp-88h]
  unsigned __int64 v99; // [rsp+68h] [rbp-78h] BYREF
  _QWORD *v100; // [rsp+70h] [rbp-70h] BYREF
  __int64 v101; // [rsp+78h] [rbp-68h]
  _QWORD v102[2]; // [rsp+80h] [rbp-60h] BYREF
  char *v103; // [rsp+90h] [rbp-50h] BYREF
  unsigned __int64 v104; // [rsp+98h] [rbp-48h]
  char v105[64]; // [rsp+A0h] [rbp-40h] BYREF

  v97 = 0;
  if ( *(_QWORD *)(a2 + 32) != a2 + 24 )
  {
    v98 = a2 + 24;
    v2 = *(_QWORD *)(a2 + 32);
    while ( 1 )
    {
      v3 = v2 - 56;
      if ( !v2 )
        v3 = 0;
      if ( sub_15E4F60(v3) )
        goto LABEL_12;
      v103 = *(char **)(v3 + 112);
      if ( (unsigned __int8)sub_1560180((__int64)&v103, 35) )
        goto LABEL_12;
      v4 = 1;
      if ( (unsigned __int8)sub_1C2F070(v3) )
        goto LABEL_15;
      if ( !(unsigned __int8)sub_1C2E690(v3, "wroimage", 8u, &v103)
        && !(unsigned __int8)sub_1C2E690(v3, "rdoimage", 8u, &v103)
        && !(unsigned __int8)sub_1C2E690(v3, "sampler", 7u, &v103) )
      {
        break;
      }
      v103 = *(char **)(v3 + 112);
      if ( (unsigned __int8)sub_1560180((__int64)&v103, 26) )
      {
        v4 = 2;
        goto LABEL_16;
      }
LABEL_10:
      *(_WORD *)(v3 + 32) = *(_WORD *)(v3 + 32) & 0xBFC0 | 0x4007;
LABEL_11:
      sub_15E0D50(v3, -1, 3);
      v97 = 1;
LABEL_12:
      v2 = *(_QWORD *)(v2 + 8);
      if ( v98 == v2 )
        return v97;
    }
    v103 = *(char **)(v3 + 112);
    if ( (unsigned __int8)sub_1560180((__int64)&v103, 26) )
      goto LABEL_12;
    n = sub_1632FA0(*(_QWORD *)(v3 + 40));
    if ( !n )
      goto LABEL_12;
    if ( (*(_BYTE *)(v3 + 18) & 1) != 0 )
    {
      sub_15E08E0(v3, 26);
      v8 = *(__int64 **)(v3 + 88);
      if ( (*(_BYTE *)(v3 + 18) & 1) != 0 )
        sub_15E08E0(v3, 26);
      v9 = *(__int64 **)(v3 + 88);
    }
    else
    {
      v8 = *(__int64 **)(v3 + 88);
      v9 = v8;
    }
    v92 = &v9[5 * *(_QWORD *)(v3 + 96)];
    if ( v92 == v8 )
    {
LABEL_38:
      v16 = **(_QWORD **)(*(_QWORD *)(v3 + 24) + 16LL);
      v17 = *(unsigned __int8 *)(v16 + 8);
      if ( (unsigned __int8)v17 > 0xFu || (v42 = 35454, !_bittest64(&v42, v17)) )
      {
        if ( (unsigned int)(v17 - 13) > 1 && (_DWORD)v17 != 16
          || !sub_16435F0(**(_QWORD **)(*(_QWORD *)(v3 + 24) + 16LL), 0) )
        {
          goto LABEL_12;
        }
      }
      v43 = 1;
      v44 = sub_15A9FE0(n, v16);
      v45 = v44;
      while ( 2 )
      {
        switch ( *(_BYTE *)(v16 + 8) )
        {
          case 1:
            v54 = 16;
            goto LABEL_117;
          case 2:
            v54 = 32;
            goto LABEL_117;
          case 3:
          case 9:
            v54 = 64;
            goto LABEL_117;
          case 4:
            v54 = 80;
            goto LABEL_117;
          case 5:
          case 6:
            v54 = 128;
            goto LABEL_117;
          case 7:
            v95 = v44;
            v62 = sub_15A9520(n, 0);
            v45 = v95;
            v54 = (unsigned int)(8 * v62);
            goto LABEL_117;
          case 0xB:
            v54 = *(_DWORD *)(v16 + 8) >> 8;
            goto LABEL_117;
          case 0xD:
            v94 = v44;
            v61 = (_QWORD *)sub_15A9930(n, v16);
            v45 = v94;
            v54 = 8LL * *v61;
            goto LABEL_117;
          case 0xE:
            srce = v44;
            v93 = *(_QWORD *)(v16 + 32);
            v60 = sub_12BE0A0(n, *(_QWORD *)(v16 + 24));
            v45 = srce;
            v54 = 8 * v93 * v60;
            goto LABEL_117;
          case 0xF:
            v96 = v44;
            v64 = sub_15A9520(n, *(_DWORD *)(v16 + 8) >> 8);
            v45 = v96;
            v54 = (unsigned int)(8 * v64);
LABEL_117:
            if ( v45 * ((v45 + ((unsigned __int64)(v43 * v54 + 7) >> 3) - 1) / v45) <= 0x90 )
              goto LABEL_12;
            v4 = 8;
            break;
          case 0x10:
            v63 = *(_QWORD *)(v16 + 32);
            v16 = *(_QWORD *)(v16 + 24);
            v43 *= v63;
            continue;
          default:
            goto LABEL_145;
        }
        break;
      }
    }
    else
    {
      v82 = v2;
      v10 = 0;
      while ( 2 )
      {
        v11 = *v8;
        if ( (unsigned __int8)sub_15E0450((__int64)v8) )
        {
          if ( *(_BYTE *)(v11 + 8) == 15 )
          {
            v12 = 1;
            v13 = *(_QWORD *)(v11 + 24);
            v14 = (unsigned int)sub_15A9FE0(n, v13);
            while ( 2 )
            {
              switch ( *(_BYTE *)(v13 + 8) )
              {
                case 0:
                case 8:
                case 0xA:
                case 0xC:
                case 0x10:
                  v18 = *(_QWORD *)(v13 + 32);
                  v13 = *(_QWORD *)(v13 + 24);
                  v12 *= v18;
                  continue;
                case 1:
                  v15 = 16;
                  goto LABEL_35;
                case 2:
                  v15 = 32;
                  goto LABEL_35;
                case 3:
                case 9:
                  v15 = 64;
                  goto LABEL_35;
                case 4:
                  v15 = 80;
                  goto LABEL_35;
                case 5:
                case 6:
                  v15 = 128;
                  goto LABEL_35;
                case 7:
                  v15 = 8 * (unsigned int)sub_15A9520(n, 0);
                  goto LABEL_35;
                case 0xB:
                  v15 = *(_DWORD *)(v13 + 8) >> 8;
                  goto LABEL_35;
                case 0xD:
                  v15 = 8LL * *(_QWORD *)sub_15A9930(n, v13);
                  goto LABEL_35;
                case 0xE:
                  v74 = *(_QWORD *)(v13 + 24);
                  src = *(void **)(v13 + 32);
                  v19 = sub_15A9FE0(n, v74);
                  v20 = v74;
                  v21 = 1;
                  v22 = v19;
                  while ( 2 )
                  {
                    switch ( *(_BYTE *)(v20 + 8) )
                    {
                      case 0:
                      case 8:
                      case 0xA:
                      case 0xC:
                      case 0x10:
                        v52 = *(_QWORD *)(v20 + 32);
                        v20 = *(_QWORD *)(v20 + 24);
                        v21 *= v52;
                        continue;
                      case 1:
                        v47 = 16;
                        goto LABEL_101;
                      case 2:
                        v47 = 32;
                        goto LABEL_101;
                      case 3:
                      case 9:
                        v47 = 64;
                        goto LABEL_101;
                      case 4:
                        v47 = 80;
                        goto LABEL_101;
                      case 5:
                      case 6:
                        v47 = 128;
                        goto LABEL_101;
                      case 7:
                        v70 = v22;
                        v48 = 0;
                        v76 = v21;
                        goto LABEL_104;
                      case 0xB:
                        v47 = *(_DWORD *)(v20 + 8) >> 8;
                        goto LABEL_101;
                      case 0xD:
                        v71 = v22;
                        v77 = v21;
                        v50 = (_QWORD *)sub_15A9930(n, v20);
                        v21 = v77;
                        v22 = v71;
                        v47 = 8LL * *v50;
                        goto LABEL_101;
                      case 0xE:
                        v67 = v22;
                        v72 = v21;
                        v78 = *(_QWORD *)(v20 + 32);
                        v51 = sub_12BE0A0(n, *(_QWORD *)(v20 + 24));
                        v21 = v72;
                        v22 = v67;
                        v47 = 8 * v78 * v51;
                        goto LABEL_101;
                      case 0xF:
                        v70 = v22;
                        v76 = v21;
                        v48 = *(_DWORD *)(v20 + 8) >> 8;
LABEL_104:
                        v49 = sub_15A9520(n, v48);
                        v21 = v76;
                        v22 = v70;
                        v47 = (unsigned int)(8 * v49);
LABEL_101:
                        v15 = 8 * (_QWORD)src * v22 * ((v22 + ((unsigned __int64)(v47 * v21 + 7) >> 3) - 1) / v22);
                        goto LABEL_35;
                      default:
                        goto LABEL_145;
                    }
                  }
                case 0xF:
                  v15 = 8 * (unsigned int)sub_15A9520(n, *(_DWORD *)(v13 + 8) >> 8);
LABEL_35:
                  LODWORD(v15) = v14 * ((v14 + ((unsigned __int64)(v12 * v15 + 7) >> 3) - 1) / v14);
                  goto LABEL_36;
                default:
                  goto LABEL_145;
              }
            }
          }
          BUG();
        }
        v23 = sub_15A9FE0(n, v11);
        v24 = 1;
        v25 = v23;
LABEL_58:
        switch ( *(_BYTE *)(v11 + 8) )
        {
          case 1:
            v26 = 16;
            goto LABEL_60;
          case 2:
            v26 = 32;
            goto LABEL_60;
          case 3:
          case 9:
            v26 = 64;
            goto LABEL_60;
          case 4:
            v26 = 80;
            goto LABEL_60;
          case 5:
          case 6:
            v26 = 128;
            goto LABEL_60;
          case 7:
            srcb = (void *)v24;
            v27 = sub_15A9520(n, 0);
            v24 = (__int64)srcb;
            v26 = (unsigned int)(8 * v27);
            goto LABEL_60;
          case 0xB:
            v26 = *(_DWORD *)(v11 + 8) >> 8;
            goto LABEL_60;
          case 0xD:
            srcd = (void *)v24;
            v34 = (_QWORD *)sub_15A9930(n, v11);
            v24 = (__int64)srcd;
            v26 = 8LL * *v34;
            goto LABEL_60;
          case 0xE:
            v69 = v24;
            v75 = *(_QWORD *)(v11 + 24);
            srca = *(void **)(v11 + 32);
            v30 = sub_15A9FE0(n, v75);
            v31 = v75;
            v32 = 1;
            v24 = v69;
            v33 = v30;
            while ( 2 )
            {
              switch ( *(_BYTE *)(v31 + 8) )
              {
                case 1:
                  v53 = 16;
                  goto LABEL_114;
                case 2:
                  v53 = 32;
                  goto LABEL_114;
                case 3:
                case 9:
                  v53 = 64;
                  goto LABEL_114;
                case 4:
                  v53 = 80;
                  goto LABEL_114;
                case 5:
                case 6:
                  v53 = 128;
                  goto LABEL_114;
                case 7:
                  v55 = 0;
                  v79 = v32;
                  goto LABEL_123;
                case 0xB:
                  v53 = *(_DWORD *)(v31 + 8) >> 8;
                  goto LABEL_114;
                case 0xD:
                  v80 = v32;
                  v57 = (_QWORD *)sub_15A9930(n, v31);
                  v32 = v80;
                  v24 = v69;
                  v53 = 8LL * *v57;
                  goto LABEL_114;
                case 0xE:
                  v68 = v69;
                  v73 = v32;
                  v81 = *(_QWORD *)(v31 + 32);
                  v59 = sub_12BE0A0(n, *(_QWORD *)(v31 + 24));
                  v32 = v73;
                  v24 = v68;
                  v53 = 8 * v81 * v59;
                  goto LABEL_114;
                case 0xF:
                  v79 = v32;
                  v55 = *(_DWORD *)(v31 + 8) >> 8;
LABEL_123:
                  v56 = sub_15A9520(n, v55);
                  v32 = v79;
                  v24 = v69;
                  v53 = (unsigned int)(8 * v56);
LABEL_114:
                  v26 = 8 * v33 * (_QWORD)srca * ((v33 + ((unsigned __int64)(v53 * v32 + 7) >> 3) - 1) / v33);
                  goto LABEL_60;
                case 0x10:
                  v58 = *(_QWORD *)(v31 + 32);
                  v31 = *(_QWORD *)(v31 + 24);
                  v32 *= v58;
                  continue;
                default:
                  goto LABEL_145;
              }
            }
          case 0xF:
            srcc = (void *)v24;
            v29 = sub_15A9520(n, *(_DWORD *)(v11 + 8) >> 8);
            v24 = (__int64)srcc;
            v26 = (unsigned int)(8 * v29);
LABEL_60:
            v15 = v25 * ((v25 + ((unsigned __int64)(v26 * v24 + 7) >> 3) - 1) / v25);
            if ( (unsigned int)v15 < 4 )
              LODWORD(v15) = 4;
LABEL_36:
            v10 += v15;
            v8 += 5;
            if ( v8 != v92 )
              continue;
            v2 = v82;
            if ( v10 <= 0x180 )
              goto LABEL_38;
            v4 = 7;
            break;
          case 0x10:
            v28 = *(_QWORD *)(v11 + 32);
            v11 = *(_QWORD *)(v11 + 24);
            v24 *= v28;
            goto LABEL_58;
          default:
LABEL_145:
            BUG();
        }
        break;
      }
    }
LABEL_15:
    v103 = *(char **)(v3 + 112);
    if ( !(unsigned __int8)sub_1560180((__int64)&v103, 26) )
      goto LABEL_11;
LABEL_16:
    v6 = (char *)sub_16D40F0((__int64)qword_4FBB4D0);
    if ( v6 )
      v7 = *v6;
    else
      v7 = qword_4FBB4D0[2];
    if ( v7 )
    {
LABEL_19:
      sub_15E0E50(v3, -1, 26);
      if ( v4 != 2 )
        goto LABEL_11;
      goto LABEL_10;
    }
    LOBYTE(v102[0]) = 0;
    v100 = v102;
    v101 = 0;
    sub_1C30B20((__int64)&v103, v3);
    sub_2241490(&v100, v103, v104);
    if ( v103 != v105 )
      j_j___libc_free_0(v103, *(_QWORD *)v105 + 1LL);
    if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v101) <= 0xA
      || (sub_2241490(&v100, ": Warning: ", 11), (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v101) <= 8) )
    {
LABEL_141:
      sub_4262D8((__int64)"basic_string::append");
    }
    sub_2241490(&v100, "Function ", 9);
    v37 = sub_1649960(v3);
    v38 = v35;
    if ( !v37 )
    {
      v104 = 0;
      v105[0] = 0;
      v103 = v105;
      sub_2241490(&v100, v105, 0, v36, v35);
      goto LABEL_87;
    }
    v39 = v105;
    v99 = v35;
    v40 = v35;
    v103 = v105;
    if ( v35 > 0xF )
    {
      na = v35;
      srcf = v37;
      v65 = (const char *)sub_22409D0(&v103, &v99, 0);
      v37 = srcf;
      v38 = na;
      v103 = (char *)v65;
      v66 = (char *)v65;
      *(_QWORD *)v105 = v99;
    }
    else
    {
      if ( v35 == 1 )
      {
        v105[0] = *v37;
        v41 = v105;
LABEL_81:
        v104 = v40;
        v41[v40] = 0;
        sub_2241490(&v100, v103, v104, v39, v38);
LABEL_87:
        if ( v103 != v105 )
          j_j___libc_free_0(v103, *(_QWORD *)v105 + 1LL);
        v46 = 0x3FFFFFFFFFFFFFFFLL - v101;
        switch ( v4 )
        {
          case 1:
            if ( v46 <= 0xC )
              goto LABEL_141;
            sub_2241490(&v100, " is a kernel,", 13);
            break;
          case 2:
            if ( v46 <= 0x16 )
              goto LABEL_141;
            sub_2241490(&v100, " has an image argument,", 23);
            break;
          default:
            break;
        }
        if ( (unsigned __int64)(0x3FFFFFFFFFFFFFFFLL - v101) <= 0x4A )
          goto LABEL_141;
        sub_2241490(&v100, " so overriding noinline attribute. The function may be inlined when called.", 75);
        sub_1C3EFD0((__int64)&v100, 1);
        if ( v100 != v102 )
          j_j___libc_free_0(v100, v102[0] + 1LL);
        goto LABEL_19;
      }
      if ( !v35 )
      {
        v41 = v105;
        goto LABEL_81;
      }
      v66 = v105;
    }
    memcpy(v66, v37, v38);
    v40 = v99;
    v41 = v103;
    goto LABEL_81;
  }
  return v97;
}
