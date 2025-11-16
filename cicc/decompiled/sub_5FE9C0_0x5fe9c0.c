// Function: sub_5FE9C0
// Address: 0x5fe9c0
//
_DWORD *__fastcall sub_5FE9C0(__int64 *a1, __int64 a2)
{
  __int64 *v2; // rax
  __int64 *v3; // r14
  _QWORD *i; // r13
  __int64 v5; // rax
  bool v6; // sf
  __int64 v7; // r12
  __int64 v8; // rax
  char v9; // al
  __int64 v10; // rbx
  char v11; // r15
  char v12; // al
  __int64 v13; // r13
  __int64 v14; // r12
  char v15; // al
  char v16; // al
  __int16 v17; // dx
  int v18; // eax
  char v19; // si
  char v20; // di
  __int64 ***v21; // rax
  __int64 **v22; // r15
  __int64 *j; // rax
  int v24; // eax
  __int64 **v25; // rdx
  __int64 *k; // rax
  int v27; // eax
  __int64 v28; // rbx
  __int64 v29; // r15
  __int64 v30; // rbx
  __int64 v31; // r15
  __int64 v32; // rdi
  __int64 m; // rax
  __int64 v34; // rbx
  __int64 v35; // rcx
  char v36; // dl
  bool v37; // bl
  char v38; // cl
  char v39; // r15
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // r9
  __int64 v44; // rbx
  char v45; // al
  int v46; // eax
  __int64 v47; // rdx
  __int64 n; // rax
  __int64 v49; // rax
  __int64 v50; // rax
  char ii; // cl
  char v52; // si
  _QWORD *v53; // rax
  __int64 v54; // rbx
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // r15
  __int64 v58; // rax
  __int64 v59; // rdi
  __int64 v60; // rax
  __int64 v61; // rdx
  __int64 v62; // r15
  __int64 v63; // rbx
  char v64; // al
  char v65; // al
  __int64 v66; // rax
  _BYTE *v67; // rax
  char v68; // al
  char v69; // al
  _DWORD *result; // rax
  __int64 v71; // r13
  __int64 v72; // rdi
  __int64 v73; // rax
  __int64 v74; // rax
  char v75; // dl
  __int64 v76; // r15
  __int64 jj; // rbx
  __int64 v78; // rax
  __int64 v79; // rdx
  __int64 v80; // rbx
  __int64 v81; // rax
  __int64 v82; // r15
  __int64 v83; // rbx
  __int64 v84; // rax
  __int64 *v85; // r12
  _QWORD *v86; // r14
  bool v87; // r13
  _QWORD *v88; // rbx
  _BYTE *v89; // rsi
  int v90; // eax
  char v91; // al
  __int64 v92; // rdi
  char v93; // al
  bool v94; // [rsp+Fh] [rbp-371h]
  int v95; // [rsp+10h] [rbp-370h]
  __int64 v96; // [rsp+10h] [rbp-370h]
  int v97; // [rsp+18h] [rbp-368h]
  char v98; // [rsp+20h] [rbp-360h]
  __int64 v99; // [rsp+20h] [rbp-360h]
  __int64 v100; // [rsp+28h] [rbp-358h]
  __int64 v101; // [rsp+28h] [rbp-358h]
  __int64 v102; // [rsp+28h] [rbp-358h]
  __int64 v103; // [rsp+28h] [rbp-358h]
  __int64 v104; // [rsp+28h] [rbp-358h]
  __int64 v105; // [rsp+28h] [rbp-358h]
  _QWORD *v106; // [rsp+30h] [rbp-350h]
  bool v107; // [rsp+30h] [rbp-350h]
  unsigned __int8 v108; // [rsp+38h] [rbp-348h]
  int v109; // [rsp+44h] [rbp-33Ch] BYREF
  int v110; // [rsp+48h] [rbp-338h] BYREF
  char v111[4]; // [rsp+4Ch] [rbp-334h] BYREF
  __int64 v112; // [rsp+50h] [rbp-330h] BYREF
  __int64 v113; // [rsp+58h] [rbp-328h]
  __int64 v114; // [rsp+60h] [rbp-320h]
  __int64 v115; // [rsp+68h] [rbp-318h]
  __int64 v116; // [rsp+70h] [rbp-310h]
  __int64 v117; // [rsp+78h] [rbp-308h]
  __int64 v118; // [rsp+80h] [rbp-300h]
  __int64 v119; // [rsp+88h] [rbp-2F8h]
  _DWORD v120[28]; // [rsp+90h] [rbp-2F0h] BYREF
  _QWORD v121[70]; // [rsp+100h] [rbp-280h] BYREF
  char v122; // [rsp+330h] [rbp-50h]

  v2 = a1;
  v3 = a1;
  for ( i = (_QWORD *)a2; *((_BYTE *)v2 + 140) == 12; v2 = (__int64 *)v2[20] )
    ;
  v5 = *v2;
  v6 = *(char *)(a2 + 9) < 0;
  v112 = 0;
  v113 = 0;
  v114 = 0;
  v7 = *(_QWORD *)(v5 + 96);
  v8 = a1[21];
  v115 = 0;
  v116 = 0;
  v100 = v8;
  v117 = 0;
  v118 = 0;
  v119 = 0;
  if ( v6 )
    HIDWORD(v115) = 1;
  if ( dword_4D04434 )
  {
    v9 = *(_BYTE *)(v7 + 182);
    if ( (v9 & 8) != 0 )
      LODWORD(v113) = 1;
    if ( (v9 & 0x10) != 0 )
      HIDWORD(v113) = 1;
    if ( (v9 & 0x20) != 0 )
      LODWORD(v114) = 1;
    if ( (v9 & 0x40) != 0 )
      HIDWORD(v115) = 1;
    if ( v9 >= 0 )
    {
      if ( (*(_BYTE *)(v7 + 183) & 1) == 0 )
        goto LABEL_16;
      goto LABEL_170;
    }
    HIDWORD(v114) = 1;
    if ( (*(_BYTE *)(v7 + 183) & 1) != 0 )
LABEL_170:
      LODWORD(v115) = 1;
  }
LABEL_16:
  v108 = 0;
  v95 = sub_5F04F0(*(_QWORD *)(v7 + 32), unk_4D04470, &v110, &v109);
  if ( v95 )
  {
    v108 = (*(_BYTE *)(v100 + 109) & 0x20) == 0;
    v95 = v108;
  }
  v10 = *(_QWORD *)(v7 + 32);
  if ( v10 )
  {
    v11 = *(_BYTE *)(v10 + 80);
    v12 = v11;
    if ( v11 == 17 )
    {
      v10 = *(_QWORD *)(v10 + 88);
      if ( v10 )
      {
        v106 = (_QWORD *)a2;
        v12 = *(_BYTE *)(v10 + 80);
        v13 = v7;
        goto LABEL_24;
      }
    }
    else
    {
      v106 = (_QWORD *)a2;
      v13 = v7;
      while ( 1 )
      {
LABEL_24:
        if ( v12 == 10 )
        {
          v14 = *(_QWORD *)(v10 + 88);
          if ( (*(_BYTE *)(v14 + 193) & 0x10) == 0 )
          {
            if ( (unsigned int)sub_72F850(*(_QWORD *)(v10 + 88)) )
            {
              v15 = *(_BYTE *)(v13 + 177);
              *(_BYTE *)(v13 + 177) = v15 | 4;
              if ( (*(_BYTE *)(v14 + 206) & 0x18) == 0 )
              {
                v7 = v13;
                i = v106;
                *(_BYTE *)(v7 + 177) = v15 | 0xC;
                goto LABEL_29;
              }
            }
          }
        }
        if ( v11 != 17 )
          break;
        v10 = *(_QWORD *)(v10 + 8);
        if ( !v10 )
          break;
        v12 = *(_BYTE *)(v10 + 80);
      }
      v7 = v13;
      i = v106;
    }
  }
LABEL_29:
  if ( (*(_BYTE *)(v7 + 177) & 8) != 0 || v110 )
    *((_BYTE *)i + 8) |= 4u;
  if ( dword_4D0446C )
  {
    v16 = *(_BYTE *)(v7 + 176);
    v17 = *(_WORD *)(v7 + 176) & 0x440;
    if ( (v16 & 8) != 0 )
    {
      v18 = v16 & 0x40;
      if ( !v18 )
        LODWORD(v114) = 1;
      v19 = *(_BYTE *)(v7 + 177);
      v20 = v19 & 4;
      if ( (v19 & 4) == 0 )
        LODWORD(v115) = 1;
      if ( !v17 )
        goto LABEL_38;
    }
    else
    {
      if ( !v17 )
        goto LABEL_38;
      v19 = *(_BYTE *)(v7 + 177);
      LOBYTE(v18) = v16 & 0x40;
      v20 = v19 & 4;
      if ( !dword_4F077BC || (unsigned __int64)(qword_4F077A8 - 40600LL) > 0x63 )
        HIDWORD(v113) = 1;
    }
    if ( !(_BYTE)v18 )
      LODWORD(v114) = 1;
    if ( !v108 )
    {
      if ( dword_4F077BC && (unsigned __int64)(qword_4F077A8 - 40600LL) <= 0x63 )
      {
        if ( v20 )
        {
LABEL_330:
          if ( (v19 & 4) == 0 && (v3[22] & 0x10) != 0 )
            LODWORD(v115) = 1;
          goto LABEL_40;
        }
LABEL_169:
        LODWORD(v115) = 1;
LABEL_38:
        if ( !dword_4F077BC || qword_4F077A8 > 0x9F5Fu )
          goto LABEL_40;
        v19 = *(_BYTE *)(v7 + 177);
        goto LABEL_330;
      }
      HIDWORD(v114) = 1;
    }
    if ( v20 )
      goto LABEL_38;
    goto LABEL_169;
  }
LABEL_40:
  v21 = (__int64 ***)v3[21];
  LODWORD(v121[0]) = 1;
  v22 = *v21;
  if ( *v21 )
  {
    while ( 1 )
    {
      if ( ((_BYTE)v22[12] & 3) != 0 )
      {
        for ( j = v22[5]; *((_BYTE *)j + 140) == 12; j = (__int64 *)j[20] )
          ;
        if ( (unsigned int)sub_5F04F0(*(_QWORD *)(*(_QWORD *)(*j + 96) + 32LL), 0, 0, (int *)v121) )
        {
          v24 = v121[0];
          if ( !LODWORD(v121[0]) )
            break;
        }
      }
      v22 = (__int64 **)*v22;
      if ( !v22 )
        goto LABEL_57;
    }
  }
  else
  {
LABEL_57:
    v28 = **(_QWORD **)(*v3 + 96);
    if ( v28 )
    {
      while ( 1 )
      {
        if ( *(_BYTE *)(v28 + 80) != 8 )
          goto LABEL_59;
        v29 = *(_QWORD *)(*(_QWORD *)(v28 + 88) + 120LL);
        if ( (unsigned int)sub_8D3410(v29) )
          v29 = sub_8D40F0(v29);
        if ( (unsigned int)sub_8D3A70(v29) )
        {
          while ( *(_BYTE *)(v29 + 140) == 12 )
            v29 = *(_QWORD *)(v29 + 160);
          if ( (unsigned int)sub_5F04F0(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)v29 + 96LL) + 32LL), 0, 0, (int *)v121) )
          {
            v24 = v121[0];
            if ( !LODWORD(v121[0]) )
              goto LABEL_48;
          }
LABEL_59:
          v28 = *(_QWORD *)(v28 + 16);
          if ( !v28 )
            break;
        }
        else
        {
          v28 = *(_QWORD *)(v28 + 16);
          if ( !v28 )
            break;
        }
      }
    }
    v24 = v121[0];
  }
LABEL_48:
  HIDWORD(v112) = v24 != 0;
  v25 = *(__int64 ***)v3[21];
  if ( v25 )
  {
    while ( 1 )
    {
      if ( ((_BYTE)v25[12] & 3) != 0 )
      {
        for ( k = v25[5]; *((_BYTE *)k + 140) == 12; k = (__int64 *)k[20] )
          ;
        if ( (*(_BYTE *)(*(_QWORD *)(*k + 96) + 176LL) & 0x18) == 8 )
          break;
      }
      v25 = (__int64 **)*v25;
      if ( !v25 )
        goto LABEL_71;
    }
  }
  else
  {
LABEL_71:
    v30 = v3[20];
    if ( !v30 )
    {
LABEL_81:
      v27 = 1;
      goto LABEL_82;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v31 = *(_QWORD *)(v30 + 120);
        if ( (unsigned int)sub_8D3410(v31) )
          v31 = sub_8D40F0(v31);
        if ( (unsigned int)sub_8D3A70(v31) )
          break;
        v30 = *(_QWORD *)(v30 + 112);
        if ( !v30 )
          goto LABEL_81;
      }
      while ( *(_BYTE *)(v31 + 140) == 12 )
        v31 = *(_QWORD *)(v31 + 160);
      if ( (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v31 + 96LL) + 176LL) & 0x18) == 8 )
        break;
      v30 = *(_QWORD *)(v30 + 112);
      if ( !v30 )
        goto LABEL_81;
    }
  }
  v27 = 0;
LABEL_82:
  v32 = *i;
  LODWORD(v112) = v27;
  for ( m = v32; *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
    ;
  v34 = *(_QWORD *)(*(_QWORD *)m + 96LL);
  v35 = *(_QWORD *)(v34 + 16);
  if ( !v35 )
  {
LABEL_118:
    if ( *(_QWORD *)(v34 + 8) )
      goto LABEL_119;
    goto LABEL_297;
  }
  if ( (*(_BYTE *)(*(_QWORD *)(v35 + 88) + 206LL) & 0x10) != 0 )
  {
LABEL_86:
    v97 = 0;
    goto LABEL_87;
  }
  if ( dword_4F077C4 == 2 )
  {
    if ( unk_4F07778 > 201102 || dword_4F07774 )
    {
      v99 = *(_QWORD *)(v34 + 16);
      sub_5E88A0(v32, 1, (__int64)&v112);
      v35 = v99;
      if ( dword_4F077C4 != 2 )
        goto LABEL_117;
      if ( unk_4F07778 > 201102 )
        goto LABEL_354;
    }
    if ( dword_4F07774 )
    {
LABEL_354:
      if ( (_DWORD)v113 )
        *(_BYTE *)(*(_QWORD *)(v35 + 88) + 206LL) |= 0x10u;
    }
  }
LABEL_117:
  if ( (*((_BYTE *)i + 9) & 0x10) == 0 )
    goto LABEL_118;
  *(_BYTE *)(v34 + 176) |= 1u;
  *(_QWORD *)(v34 + 16) = 0;
  *(_BYTE *)(*(_QWORD *)(v35 + 88) + 194LL) &= ~2u;
  if ( *(_QWORD *)(v34 + 8) )
  {
LABEL_119:
    if ( (*(_BYTE *)(v34 + 176) & 2) == 0 )
      *((_BYTE *)i + 8) |= 4u;
    goto LABEL_86;
  }
LABEL_297:
  if ( (i[1] & 0x200004) == 0 && (*(_BYTE *)(v34 + 177) & 0x40) != 0 )
  {
    v75 = *((_BYTE *)i + 10);
    if ( (v75 & 2) == 0
      && (!dword_4D04464
       || (dword_4F077BC && !dword_4F077B4 && qword_4F077A8 <= 0x9FC3u || (i[1] & 0x20) == 0)
       && !__PAIR64__(v114, HIDWORD(v113)))
      && v75 >= 0 )
    {
      goto LABEL_86;
    }
  }
  if ( (*(_BYTE *)(*(_QWORD *)(v32 + 168) + 109LL) & 0x20) != 0 && *(_QWORD *)(v34 + 16) )
    goto LABEL_86;
  v97 = 1;
  if ( dword_4F077C4 == 2 && (unk_4F07778 > 201102 || dword_4F07774) )
  {
    sub_5E88A0(v32, 1, (__int64)&v112);
    v97 = 1;
  }
LABEL_87:
  v107 = 0;
  if ( !v108 && (*(_BYTE *)(v100 + 109) & 0x20) == 0 )
  {
    v107 = 1;
    if ( qword_4D0495C )
    {
      if ( (*((_BYTE *)i + 10) & 0x40) == 0 )
        v107 = *(_QWORD *)(v7 + 32) == 0;
    }
  }
  v98 = 0;
  v36 = *(_BYTE *)(v7 + 177);
  if ( dword_4D0446C && (v36 & 4) == 0 && !((unsigned int)v115 | v95) && !*(_QWORD *)(v7 + 24) )
    v98 = ((*(_BYTE *)(v100 + 109) >> 5) ^ 1) & 1;
  v37 = 1;
  if ( (v36 & 0xC0) == 0x40 )
    v37 = (*((_BYTE *)i + 10) & 2) != 0;
  v38 = *(_BYTE *)(v7 + 176);
  if ( (v38 & 8) != 0 )
  {
    v39 = 0;
    goto LABEL_122;
  }
  if ( HIDWORD(v113) )
  {
    v39 = 1;
    if ( !dword_4D0446C || v108 )
      goto LABEL_97;
    goto LABEL_173;
  }
  v39 = 1;
  if ( (*(_BYTE *)(v100 + 109) & 0x20) != 0 || *(_QWORD *)(v7 + 8) || (i[1] & 0xA01000) != 0 )
    goto LABEL_122;
  if ( !dword_4D04464 )
  {
    v39 = v37;
    goto LABEL_122;
  }
  if ( (*(_BYTE *)(v7 + 178) & 0x20) != 0 )
  {
LABEL_122:
    if ( !dword_4D0446C || v108 )
    {
LABEL_97:
      if ( HIDWORD(v115) )
        goto LABEL_99;
      goto LABEL_98;
    }
    goto LABEL_173;
  }
  if ( !HIDWORD(v115) )
  {
    if ( dword_4D0446C )
    {
      v39 = v37;
      if ( v108 )
        goto LABEL_98;
LABEL_173:
      if ( v36 & 4 | v38 & 0x48 )
        goto LABEL_97;
      goto LABEL_174;
    }
    v39 = v37;
    if ( (*((_BYTE *)i + 9) & 0x60) == 0 )
    {
      v37 = 0;
      goto LABEL_127;
    }
LABEL_99:
    v37 = 0;
    if ( *(_QWORD *)(v7 + 24) )
      goto LABEL_100;
    goto LABEL_295;
  }
  if ( !dword_4D0446C )
  {
    v108 = *(_QWORD *)(v7 + 24) == 0;
    v37 = 0;
    goto LABEL_128;
  }
  if ( v108 )
  {
    v37 = 0;
    v39 = v108;
    if ( !*(_QWORD *)(v7 + 24) )
      goto LABEL_103;
    goto LABEL_100;
  }
  v39 = 1;
  if ( v38 & 0x48 | v36 & 4 )
    goto LABEL_99;
LABEL_174:
  if ( *(_QWORD *)(v7 + 24) )
  {
    if ( HIDWORD(v115) )
    {
LABEL_176:
      v37 = 0;
      goto LABEL_100;
    }
LABEL_98:
    if ( (*((_BYTE *)i + 9) & 0x60) != 0 )
      goto LABEL_99;
    goto LABEL_176;
  }
  if ( (*(_BYTE *)(v100 + 109) & 0x20) == 0 && !*(_QWORD *)(v7 + 8) && (i[1] & 0xA01000) == 0 )
  {
    if ( dword_4D04464 )
    {
      if ( (*(_BYTE *)(v7 + 178) & 0x20) == 0 )
      {
        if ( HIDWORD(v115) )
        {
LABEL_294:
          v37 = 1;
          goto LABEL_295;
        }
        if ( !v37 )
          goto LABEL_98;
        goto LABEL_293;
      }
    }
    else if ( !v37 )
    {
      if ( !HIDWORD(v115) )
        goto LABEL_98;
LABEL_295:
      if ( !dword_4D0446C )
      {
        v108 = 1;
        goto LABEL_128;
      }
LABEL_337:
      v108 = 1;
      goto LABEL_103;
    }
  }
  if ( HIDWORD(v115) )
  {
    v37 = 1;
    goto LABEL_337;
  }
LABEL_293:
  v37 = 1;
  if ( (*((_BYTE *)i + 9) & 0x60) != 0 )
    goto LABEL_294;
LABEL_100:
  if ( !dword_4D0446C )
  {
LABEL_127:
    v108 = 0;
    goto LABEL_128;
  }
  v108 = v39 | v107;
  if ( (unsigned __int8)v39 | v107 )
  {
    v108 = 0;
    goto LABEL_103;
  }
  v39 = 0;
  if ( (*((_BYTE *)i + 11) & 1) == 0 )
  {
LABEL_128:
    if ( *(_DWORD *)&word_4D04898 )
    {
      sub_5E7FB0((__int64)v3, (__int64)&v112);
      v113 = 0;
      v114 = 0;
      v115 = 0;
      v116 = 0;
      LODWORD(v117) = 0;
    }
    goto LABEL_130;
  }
LABEL_103:
  sub_5E7FB0((__int64)v3, (__int64)&v112);
  if ( (*((_BYTE *)i + 11) & 1) == 0 )
    goto LABEL_130;
  v101 = v3[21];
  if ( !*(_QWORD *)(*(_QWORD *)(v101 + 152) + 144LL) )
  {
    if ( unk_4D041C0 )
    {
      v53 = *(_QWORD **)(v101 + 136);
      if ( v53 )
        goto LABEL_360;
    }
    goto LABEL_195;
  }
  v94 = v37;
  v44 = *(_QWORD *)(*(_QWORD *)(v101 + 152) + 144LL);
  do
  {
    v45 = *(_BYTE *)(v44 + 206);
    if ( (v45 & 8) == 0 )
      goto LABEL_107;
    v40 = *(unsigned __int8 *)(v44 + 174);
    if ( (_BYTE)v40 == 1 )
    {
      if ( (unsigned int)sub_72F310(v44, 1, v40, v41, v42, v43) )
      {
        if ( !(_DWORD)v113 )
          goto LABEL_107;
LABEL_370:
        *(_BYTE *)(v44 + 206) |= 0x10u;
        *(_BYTE *)(v44 + 193) |= 0x20u;
        goto LABEL_107;
      }
      if ( !(unsigned int)sub_72F500(v44, 0, v121, 0, 1) )
      {
        if ( !(unsigned int)sub_72F570(v44) )
          goto LABEL_107;
        if ( (_DWORD)v114 )
          goto LABEL_370;
        goto LABEL_406;
      }
      if ( HIDWORD(v113) )
        goto LABEL_370;
    }
    else if ( (_BYTE)v40 == 5 )
    {
      v46 = *(unsigned __int8 *)(v44 + 176);
      if ( (_BYTE)v46 == 15 )
      {
        if ( (unsigned int)sub_5F04A0(*(_QWORD *)v44, 0, (__int64)v120, (__int64)v111, (__int64)v121) )
        {
          if ( HIDWORD(v114) )
            goto LABEL_370;
        }
        else if ( (unsigned int)sub_72F850(v44) )
        {
          if ( (_DWORD)v115 )
            goto LABEL_370;
LABEL_406:
          sub_5F9380(v44, 0, v40, v41, v42);
        }
      }
      else if ( (_BYTE)v46 == 30 )
      {
        sub_691E30(v3, v44, v40, v41, v42, v43);
      }
      else
      {
        v40 = (unsigned int)(v46 - 31);
        if ( (unsigned __int8)(v46 - 31) <= 2u || (unsigned __int8)(v46 - 16) <= 1u )
          sub_692200(v3, v44, v40, v41, v42, v43);
      }
    }
    else if ( (_BYTE)v40 == 2 && HIDWORD(v115) )
    {
      *(_BYTE *)(v44 + 193) |= 0x20u;
      *(_BYTE *)(v44 + 206) = v45 | 0x10;
    }
LABEL_107:
    v44 = *(_QWORD *)(v44 + 112);
  }
  while ( v44 );
  v37 = v94;
  if ( unk_4D041C0 )
  {
    v53 = *(_QWORD **)(v101 + 136);
    if ( v53 )
    {
LABEL_360:
      v103 = v7;
      v85 = v3;
      v86 = i;
      v87 = v37;
      v88 = v53;
      do
      {
        v89 = (_BYTE *)v88[1];
        if ( (v89[206] & 8) != 0 && v89[174] == 5 )
        {
          v90 = (unsigned __int8)v89[176];
          if ( (_BYTE)v90 == 30 )
          {
            sub_691E30(v85, v89, v40, v41, v42, v43);
          }
          else
          {
            v41 = (unsigned int)(v90 - 31);
            if ( (unsigned __int8)(v90 - 31) <= 2u || (unsigned __int8)(v90 - 16) <= 1u )
              sub_692200(v85, v89, v40, v41, v42, v43);
          }
        }
        v88 = (_QWORD *)*v88;
      }
      while ( v88 );
      v37 = v87;
      i = v86;
      v3 = v85;
      v7 = v103;
    }
  }
LABEL_130:
  v47 = *(_QWORD *)(*(_QWORD *)(v3[21] + 152) + 144LL);
  if ( v47 )
  {
    while ( (*(_BYTE *)(v47 + 206) & 8) == 0 )
    {
LABEL_132:
      v47 = *(_QWORD *)(v47 + 112);
      if ( !v47 )
        goto LABEL_195;
    }
    for ( n = *(_QWORD *)(v47 + 152); *(_BYTE *)(n + 140) == 12; n = *(_QWORD *)(n + 160) )
      ;
    v49 = **(_QWORD **)(n + 168);
    if ( !v49 )
    {
LABEL_137:
      if ( dword_4F068EC
        && *(char *)(v47 + 192) < 0
        && (*(_BYTE *)(v47 + 206) & 0x10) == 0
        && (*(_BYTE *)(v47 + 193) & 4) == 0 )
      {
        v102 = v47;
        sub_89A080(v47);
        v47 = v102;
      }
      goto LABEL_132;
    }
    v50 = *(_QWORD *)(v49 + 8);
    for ( ii = *(_BYTE *)(v50 + 140); ii == 12; ii = *(_BYTE *)(v50 + 140) )
      v50 = *(_QWORD *)(v50 + 160);
    v52 = *(_BYTE *)(v47 + 174);
    if ( v52 == 1 )
    {
      if ( ii != 6 || (*(_BYTE *)(v50 + 168) & 1) == 0 )
        goto LABEL_137;
      if ( (*(_BYTE *)(v50 + 168) & 2) != 0 )
      {
        if ( !*(_DWORD *)&word_4D04898 || (_DWORD)v118 )
          goto LABEL_137;
      }
      else
      {
        v72 = *(_QWORD *)(v50 + 160);
        if ( (*(_BYTE *)(v72 + 140) & 0xFB) == 8 )
        {
          v104 = v47;
          v91 = sub_8D4C10(v72, dword_4F077C4 != 2);
          v47 = v104;
          if ( (v91 & 1) != 0 && (v112 & 1) == 0 )
          {
            if ( dword_4F077C4 == 2 && unk_4F07778 > 202001
              || dword_4F077BC
              && !dword_4F077B4
              && qword_4F077A8 > 0x15F8Fu
              && dword_4F077C4 == 2
              && (unk_4F07778 > 201102 || dword_4F07774) )
            {
              *(_BYTE *)(v104 + 206) |= 0x10u;
            }
            else
            {
              sub_6851C0(2456, v104 + 64);
              v47 = v104;
            }
          }
        }
        if ( !*(_DWORD *)&word_4D04898 || HIDWORD(v117) )
          goto LABEL_137;
      }
    }
    else
    {
      if ( v52 != 5 || *(_BYTE *)(v47 + 176) != 15 || ii != 6 || (*(_BYTE *)(v50 + 168) & 1) == 0 )
        goto LABEL_137;
      if ( (*(_BYTE *)(v50 + 168) & 2) != 0 )
      {
        if ( !*(_DWORD *)&word_4D04898 || (_DWORD)v119 )
          goto LABEL_137;
      }
      else
      {
        v92 = *(_QWORD *)(v50 + 160);
        if ( (*(_BYTE *)(v92 + 140) & 0xFB) == 8 )
        {
          v105 = v47;
          v93 = sub_8D4C10(v92, dword_4F077C4 != 2);
          v47 = v105;
          if ( (v93 & 1) != 0 && (v112 & 0x100000000LL) == 0 )
          {
            if ( dword_4F077C4 == 2 && unk_4F07778 > 202001 )
            {
              *(_BYTE *)(v105 + 206) |= 0x10u;
            }
            else
            {
              sub_6851C0(2457, v105 + 64);
              v47 = v105;
            }
          }
        }
        if ( !*(_DWORD *)&word_4D04898 || HIDWORD(v118) )
          goto LABEL_137;
      }
    }
    if ( (v3[22] & 0x10) == 0 )
      *(_BYTE *)(v47 + 193) |= 2u;
    goto LABEL_137;
  }
LABEL_195:
  if ( v97 && (sub_5FE730(i, v113), !*(_QWORD *)(v7 + 8)) && (v74 = *(_QWORD *)(v7 + 16)) != 0 )
  {
    if ( v37 || v39 || *((char *)i + 10) < 0 )
    {
      *(_QWORD *)(v7 + 8) = v74;
      goto LABEL_196;
    }
  }
  else
  {
LABEL_196:
    if ( v37 && !(_DWORD)v114 )
    {
      v83 = *i;
      v84 = sub_72D6A0(*i);
      v96 = sub_724EF0(v84);
      sub_5E4C60((__int64)v121, (_QWORD *)(v83 + 64));
      v122 |= 2u;
      sub_87E3B0(v120);
      sub_5FE480(i, (__int64)v121, (__int64)v120, v96);
      if ( *(_DWORD *)&word_4D04898 )
      {
        if ( !(_DWORD)v118 && (*(_BYTE *)(v83 + 176) & 0x10) == 0 )
          *(_BYTE *)(*(_QWORD *)(v121[0] + 88LL) + 193LL) |= 2u;
      }
    }
    if ( v39 )
    {
      if ( dword_4F077BC && (unsigned __int64)(qword_4F077A8 - 40600LL) <= 0x63 && (*(_BYTE *)(v7 + 176) & 0x40) != 0 )
      {
        *((_BYTE *)i + 10) |= 4u;
      }
      else
      {
        v54 = *i;
        v55 = sub_73C570(*i, (unsigned int)v112, -1);
        v56 = sub_72D600(v55);
        v57 = sub_724EF0(v56);
        sub_5E4C60((__int64)v121, (_QWORD *)(v54 + 64));
        v122 |= 2u;
        sub_87E3B0(v120);
        sub_5FE480(i, (__int64)v121, (__int64)v120, v57);
        if ( HIDWORD(v113) && dword_4D04464 )
        {
          v78 = v121[0];
          v79 = *(_QWORD *)(v121[0] + 88LL);
          *(_BYTE *)(v121[0] + 81LL) |= 2u;
          *(_BYTE *)(v79 + 206) |= 0x10u;
          *(_BYTE *)(*(_QWORD *)(v78 + 88) + 193LL) |= 0x20u;
        }
        if ( *(_DWORD *)&word_4D04898 && !HIDWORD(v117) && (*(_BYTE *)(v54 + 176) & 0x10) == 0 )
          *(_BYTE *)(*(_QWORD *)(v121[0] + 88LL) + 193LL) |= 2u;
      }
    }
  }
  if ( v108 )
  {
    sub_5FE7E0(i, (__int64)&v112);
  }
  else
  {
    v58 = *(_QWORD *)(v7 + 24);
    if ( !v58 )
      goto LABEL_263;
    sub_5F93D0(*(_QWORD *)(v58 + 88), (__int64 *)(*(_QWORD *)(v58 + 88) + 152LL));
    v59 = *(_QWORD *)(v7 + 24);
    if ( (*(_BYTE *)(v59 + 104) & 1) != 0 )
    {
      if ( (unsigned int)sub_8796F0(v59) )
      {
        v59 = *(_QWORD *)(v7 + 24);
        goto LABEL_320;
      }
    }
    else
    {
      v60 = *(_QWORD *)(v59 + 88);
      v61 = v60;
      if ( *(_BYTE *)(v59 + 80) == 20 )
        v61 = *(_QWORD *)(v60 + 176);
      if ( (*(_BYTE *)(v61 + 208) & 4) == 0 )
      {
LABEL_212:
        if ( (*(_BYTE *)(v60 + 206) & 0x18) != 0 && (*(_BYTE *)(v60 + 192) & 2) == 0 && (*((_BYTE *)i + 9) & 0x60) == 0 )
          goto LABEL_263;
        goto LABEL_214;
      }
LABEL_320:
      v121[0] = 0;
      v121[1] = 0;
      v76 = sub_67DA80((unsigned int)((*(_BYTE *)(v59 + 82) & 4) == 0) + 3127, dword_4F07508, v3);
      for ( jj = *(_QWORD *)(*(_QWORD *)(v3[21] + 152) + 144LL); jj; jj = *(_QWORD *)(jj + 112) )
      {
        while ( *(_BYTE *)(jj + 174) != 2 || (*(_BYTE *)(*(_QWORD *)jj + 82LL) & 4) == 0 )
        {
          jj = *(_QWORD *)(jj + 112);
          if ( !jj )
            goto LABEL_326;
        }
        sub_6855B0(3129, jj + 64, v121);
      }
LABEL_326:
      sub_67E370(v76, v121);
      sub_685910(v76);
    }
  }
  v73 = *(_QWORD *)(v7 + 24);
  if ( v73 )
  {
    v60 = *(_QWORD *)(v73 + 88);
    goto LABEL_212;
  }
LABEL_263:
  *(_BYTE *)(v7 + 177) |= 2u;
LABEL_214:
  if ( v107 )
  {
    if ( dword_4F077BC && (unsigned __int64)(qword_4F077A8 - 40600LL) <= 0x63 && (*(_BYTE *)(v7 + 177) & 4) != 0 )
      *((_BYTE *)i + 10) |= 0x10u;
    else
      sub_5FE8B0(i, (unsigned int *)&v112);
  }
  if ( v98 )
  {
    if ( !(_DWORD)v115 )
    {
      v80 = *i;
      sub_5E4C60((__int64)v121, (_QWORD *)(*i + 64LL));
      v81 = sub_72D6A0(v80);
      v82 = sub_724EF0(v81);
      sub_87E3B0(v120);
      sub_5FE480(i, (__int64)v121, (__int64)v120, v82);
      if ( dword_4D0488C )
      {
        if ( !(_DWORD)v119 && (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)v80 + 96LL) + 183LL) & 4) == 0 )
          *(_BYTE *)(*(_QWORD *)(v121[0] + 88LL) + 193LL) |= 2u;
      }
    }
  }
  v62 = *(_QWORD *)(*(_QWORD *)*i + 96LL);
  if ( (*(_DWORD *)(v62 + 176) & 0x78000) != 0x78000 )
  {
    v63 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*i + 168LL) + 152LL) + 144LL);
    if ( v63 )
    {
      while ( 2 )
      {
        if ( (*(_BYTE *)(v63 + 193) & 0x10) == 0 && (*(_BYTE *)(v63 + 206) & 0x18) == 0 )
          goto LABEL_223;
        v120[0] = 0;
        LODWORD(v121[0]) = 0;
        v64 = *(_BYTE *)(v63 + 174);
        if ( v64 != 1 )
        {
LABEL_227:
          if ( v64 != 5 || *(_BYTE *)(v63 + 176) != 15 || !(unsigned int)sub_72F790(v63, v120, v121) )
            goto LABEL_223;
          v65 = *(_BYTE *)(v62 + 178);
          if ( LODWORD(v121[0]) )
          {
            if ( (v65 & 4) != 0 )
            {
LABEL_232:
              if ( (*(_BYTE *)(v63 + 206) & 0x10) != 0 )
              {
                if ( dword_4F077BC && qword_4F077A8 <= 0x9EFBu )
                  *(_BYTE *)(v63 + 194) &= ~4u;
                *(_BYTE *)(v62 + 177) |= 0x10u;
              }
              goto LABEL_223;
            }
          }
          else if ( (v65 & 2) != 0 )
          {
            goto LABEL_232;
          }
          *(_DWORD *)(v63 + 192) = *(_DWORD *)(v63 + 192) & 0xFFFBFDFF | (word_4D04898 << 9) & 0x200 | 0x40000;
          goto LABEL_232;
        }
        if ( !(unsigned int)sub_72F500(v63, 0, v120, 1, 1) )
        {
          v64 = *(_BYTE *)(v63 + 174);
          goto LABEL_227;
        }
        LODWORD(v121[0]) = sub_72F530(v63);
        if ( LODWORD(v121[0]) )
        {
          if ( (*(_BYTE *)(v62 + 178) & 1) == 0 )
            goto LABEL_282;
        }
        else if ( *(char *)(v62 + 177) >= 0 )
        {
LABEL_282:
          *(_DWORD *)(v63 + 192) = *(_DWORD *)(v63 + 192) & 0xFFFBFDFF | (word_4D04898 << 9) & 0x200 | 0x40000;
        }
        if ( (*(_BYTE *)(v63 + 206) & 0x10) != 0 )
        {
          if ( dword_4F077BC && qword_4F077A8 <= 0x9EFBu )
            *(_BYTE *)(v63 + 194) &= ~4u;
          *(_BYTE *)(v62 + 177) |= 1u;
        }
LABEL_223:
        v63 = *(_QWORD *)(v63 + 112);
        if ( !v63 )
          break;
        continue;
      }
    }
  }
  v66 = *(_QWORD *)(v62 + 24);
  if ( v66 )
  {
    if ( (*(_BYTE *)(v62 + 177) & 2) != 0 )
    {
      v67 = *(_BYTE **)(v66 + 88);
      if ( (v67[193] & 0x10) != 0 || (v67[206] & 0x18) != 0 )
        v67[194] |= 8u;
    }
  }
  v68 = *(_BYTE *)(v7 + 176);
  if ( (v68 & 0x20) != 0 )
    *(_BYTE *)(v7 + 177) = *(_BYTE *)(v7 + 177) & 0x3F | 0x80;
  if ( v68 < 0 )
    *(_DWORD *)(v7 + 176) = *(_DWORD *)(v7 + 176) & 0xFFFEBFFF | 0x10000;
  if ( v110 )
    *(_BYTE *)(v7 + 178) |= 2u;
  if ( (*(_BYTE *)(v7 + 177) & 8) != 0 )
    *(_BYTE *)(v7 + 178) |= 4u;
  if ( (*(_DWORD *)(v7 + 176) & 0x20800) != 0 )
    *(_BYTE *)(v7 + 177) &= ~0x20u;
  v69 = *((_BYTE *)i + 10);
  if ( (v69 & 4) != 0 )
  {
    *(_BYTE *)(v7 + 177) &= ~0x40u;
    v69 = *((_BYTE *)i + 10);
  }
  if ( (v69 & 0x10) != 0 )
    *(_BYTE *)(v7 + 177) &= ~0x20u;
  result = &dword_4D04278;
  if ( dword_4D04278 | unk_4F076E0 )
  {
    v71 = *v3;
    if ( *(_QWORD *)(v7 + 8) )
    {
      if ( *(_QWORD *)(v7 + 24) )
        return result;
      return (_DWORD *)sub_5F2480(v71, 0);
    }
    result = (_DWORD *)sub_5F1EF0(*v3);
    if ( !*(_QWORD *)(v7 + 24) )
      return (_DWORD *)sub_5F2480(v71, 0);
  }
  return result;
}
