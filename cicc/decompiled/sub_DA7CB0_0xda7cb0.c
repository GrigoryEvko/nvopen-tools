// Function: sub_DA7CB0
// Address: 0xda7cb0
//
__int64 __fastcall sub_DA7CB0(__int64 a1, __int64 a2, __int64 a3, unsigned __int8 a4)
{
  unsigned __int8 v4; // cl
  __int64 v7; // rsi
  __int64 v8; // rdi
  _QWORD *v9; // rax
  _QWORD *v10; // rdx
  __int64 *v11; // rax
  __int64 v12; // rbx
  __int64 v13; // r14
  __int64 v14; // rbx
  __int64 v15; // rdx
  __int64 v16; // r13
  _BYTE *v17; // rax
  int v18; // r11d
  __int64 *v19; // rcx
  unsigned int v20; // r8d
  __int64 *v21; // rdx
  __int64 v22; // rdi
  _QWORD *v23; // rdx
  __int64 v24; // rax
  unsigned int v25; // eax
  _BYTE *v26; // rdx
  unsigned __int8 *v27; // rax
  unsigned __int8 v28; // dl
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 *v31; // r11
  __int64 v32; // rsi
  __int64 v33; // rcx
  __int64 *v34; // r15
  int v35; // edx
  __int64 v36; // rdi
  int v37; // edx
  unsigned int v38; // eax
  __int64 v39; // r12
  int v41; // edi
  unsigned int v42; // edx
  __int64 v43; // r12
  int v44; // r9d
  __int64 *v45; // r8
  __int64 *v46; // rsi
  int v47; // r8d
  unsigned int v48; // r12d
  __int64 v49; // r9
  __int64 *v50; // rax
  __int64 *v51; // r13
  __int64 v52; // r12
  __int64 *v53; // rbx
  __int64 v54; // rax
  __int64 *v55; // rbx
  int v56; // r10d
  _QWORD *v57; // rdx
  unsigned int v58; // r8d
  _QWORD *v59; // rax
  __int64 v60; // rdi
  unsigned __int8 **v61; // r13
  __int64 v62; // r12
  __int64 v63; // rcx
  int v64; // eax
  __int64 v65; // rsi
  __int64 v66; // rsi
  __int64 v67; // rax
  __int64 v68; // rax
  unsigned __int8 *v69; // rdi
  unsigned __int8 v70; // al
  int v71; // edi
  _QWORD *v72; // rsi
  __int64 v73; // r13
  __int64 v74; // r8
  int v75; // r10d
  _QWORD *v76; // r9
  int i; // edi
  _QWORD *v78; // rax
  __int64 v79; // rax
  __int64 *v80; // [rsp+8h] [rbp-108h]
  __int64 *v81; // [rsp+8h] [rbp-108h]
  _BYTE *v82; // [rsp+10h] [rbp-100h]
  int v83; // [rsp+18h] [rbp-F8h]
  _BYTE *v84; // [rsp+18h] [rbp-F8h]
  _BYTE *v85; // [rsp+18h] [rbp-F8h]
  unsigned __int8 *v86; // [rsp+20h] [rbp-F0h]
  __int64 v88; // [rsp+30h] [rbp-E0h]
  _BYTE *v89; // [rsp+38h] [rbp-D8h]
  unsigned int v90; // [rsp+38h] [rbp-D8h]
  __int64 v93; // [rsp+50h] [rbp-C0h] BYREF
  __int64 *v94; // [rsp+58h] [rbp-B8h]
  __int64 v95; // [rsp+60h] [rbp-B0h]
  unsigned int v96; // [rsp+68h] [rbp-A8h]
  __int64 v97; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v98; // [rsp+78h] [rbp-98h]
  __int64 v99; // [rsp+80h] [rbp-90h]
  unsigned int v100; // [rsp+88h] [rbp-88h]
  __int64 *v101; // [rsp+90h] [rbp-80h] BYREF
  __int64 v102; // [rsp+98h] [rbp-78h]
  __int64 v103; // [rsp+A0h] [rbp-70h] BYREF
  unsigned int v104; // [rsp+A8h] [rbp-68h]

  v4 = *(_BYTE *)a3;
  if ( *(_BYTE *)a3 <= 0x1Cu )
    return sub_D970F0(a1);
  v7 = *(_QWORD *)(a3 + 40);
  v8 = a2;
  if ( *(_BYTE *)(a2 + 84) )
  {
    v9 = *(_QWORD **)(a2 + 64);
    v10 = &v9[*(unsigned int *)(v8 + 76)];
    if ( v9 == v10 )
      return sub_D970F0(a1);
    while ( v7 != *v9 )
    {
      if ( v10 == ++v9 )
        return sub_D970F0(a1);
    }
  }
  else
  {
    if ( !sub_C8CA60(a2 + 56, v7) )
      return sub_D970F0(a1);
    v4 = *(_BYTE *)a3;
  }
  if ( v4 == 84 )
  {
    if ( **(_QWORD **)(a2 + 32) != *(_QWORD *)(a3 + 40) )
      return sub_D970F0(a1);
    goto LABEL_9;
  }
  if ( !sub_D90BC0((unsigned __int8 *)a3) )
    return sub_D970F0(a1);
  if ( *(_BYTE *)a3 == 84 )
  {
LABEL_9:
    v89 = (_BYTE *)a3;
    goto LABEL_10;
  }
  v101 = 0;
  v102 = 0;
  v103 = 0;
  v104 = 0;
  v89 = sub_DA6F50(a3, a2, (__int64)&v101, 0);
  sub_C7D6A0(v102, 16LL * v104, 8);
  if ( !v89 )
    return sub_D970F0(a1);
LABEL_10:
  if ( (*((_DWORD *)v89 + 1) & 0x7FFFFFF) != 2 )
    return sub_D970F0(a1);
  v93 = 0;
  v94 = 0;
  v95 = 0;
  v11 = *(__int64 **)(a2 + 32);
  v96 = 0;
  v12 = *v11;
  v88 = *v11;
  v13 = sub_D47930(a2);
  v14 = sub_AA5930(v12);
  v16 = v15;
LABEL_12:
  if ( v16 != v14 )
  {
    while ( 1 )
    {
      v17 = sub_D90990(v14, v13);
      if ( !v17 )
        goto LABEL_18;
      if ( !v96 )
        break;
      v18 = 1;
      v19 = 0;
      v20 = (v96 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v21 = &v94[2 * v20];
      v22 = *v21;
      if ( v14 != *v21 )
      {
        while ( v22 != -4096 )
        {
          if ( !v19 && v22 == -8192 )
            v19 = v21;
          v20 = (v96 - 1) & (v18 + v20);
          v21 = &v94[2 * v20];
          v22 = *v21;
          if ( v14 == *v21 )
            goto LABEL_16;
          ++v18;
        }
        if ( !v19 )
          v19 = v21;
        ++v93;
        v41 = v95 + 1;
        if ( 4 * ((int)v95 + 1) < 3 * v96 )
        {
          if ( v96 - HIDWORD(v95) - v41 <= v96 >> 3 )
          {
            v85 = v17;
            sub_DA7360((__int64)&v93, v96);
            if ( !v96 )
            {
LABEL_160:
              LODWORD(v95) = v95 + 1;
              BUG();
            }
            v46 = 0;
            v47 = 1;
            v48 = (v96 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
            v41 = v95 + 1;
            v17 = v85;
            v19 = &v94[2 * v48];
            v49 = *v19;
            if ( v14 != *v19 )
            {
              while ( v49 != -4096 )
              {
                if ( v49 == -8192 && !v46 )
                  v46 = v19;
                v48 = (v96 - 1) & (v47 + v48);
                v19 = &v94[2 * v48];
                v49 = *v19;
                if ( v14 == *v19 )
                  goto LABEL_55;
                ++v47;
              }
              if ( v46 )
                v19 = v46;
            }
          }
          goto LABEL_55;
        }
LABEL_59:
        v84 = v17;
        sub_DA7360((__int64)&v93, 2 * v96);
        if ( !v96 )
          goto LABEL_160;
        v42 = (v96 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
        v41 = v95 + 1;
        v17 = v84;
        v19 = &v94[2 * v42];
        v43 = *v19;
        if ( v14 != *v19 )
        {
          v44 = 1;
          v45 = 0;
          while ( v43 != -4096 )
          {
            if ( !v45 && v43 == -8192 )
              v45 = v19;
            v42 = (v96 - 1) & (v44 + v42);
            v19 = &v94[2 * v42];
            v43 = *v19;
            if ( v14 == *v19 )
              goto LABEL_55;
            ++v44;
          }
          if ( v45 )
            v19 = v45;
        }
LABEL_55:
        LODWORD(v95) = v41;
        if ( *v19 != -4096 )
          --HIDWORD(v95);
        *v19 = v14;
        v23 = v19 + 1;
        v19[1] = 0;
        goto LABEL_17;
      }
LABEL_16:
      v23 = v21 + 1;
LABEL_17:
      *v23 = v17;
LABEL_18:
      if ( !v14 )
        BUG();
      v24 = *(_QWORD *)(v14 + 32);
      if ( !v24 )
        BUG();
      v14 = 0;
      if ( *(_BYTE *)(v24 - 24) != 84 )
        goto LABEL_12;
      v14 = v24 - 24;
      if ( v16 == v24 - 24 )
        goto LABEL_22;
    }
    ++v93;
    goto LABEL_59;
  }
LABEL_22:
  if ( !v96 )
    goto LABEL_34;
  v25 = (v96 - 1) & (((unsigned int)v89 >> 9) ^ ((unsigned int)v89 >> 4));
  v26 = (_BYTE *)v94[2 * v25];
  if ( v26 != v89 )
  {
    for ( i = 1; ; ++i )
    {
      if ( v26 == (_BYTE *)-4096LL )
        goto LABEL_34;
      v25 = (v96 - 1) & (i + v25);
      v26 = (_BYTE *)v94[2 * v25];
      if ( v26 == v89 )
        break;
    }
  }
  v83 = dword_4F89AE8;
  v82 = *(_BYTE **)(a1 + 8);
  if ( !dword_4F89AE8 )
    goto LABEL_34;
  v90 = 0;
  v86 = (unsigned __int8 *)a3;
LABEL_26:
  v27 = v86;
  v28 = *v86;
  if ( *v86 > 0x15u )
  {
    if ( v28 <= 0x1Cu )
      goto LABEL_34;
    v27 = (unsigned __int8 *)sub_DA7540((__int64)v86, a2, (__int64)&v93, v82, *(__int64 **)(a1 + 24));
    if ( !v27 )
      goto LABEL_34;
    v28 = *v27;
  }
  if ( v28 != 17 )
    goto LABEL_34;
  if ( !sub_D94970((__int64)(v27 + 24), (_QWORD *)a4) )
  {
    v31 = &v103;
    v97 = 0;
    v102 = 0x800000000LL;
    v98 = 0;
    v99 = 0;
    v100 = 0;
    v101 = &v103;
    if ( !(_DWORD)v95 )
      goto LABEL_30;
    v50 = v94;
    v51 = &v94[2 * v96];
    if ( v94 == v51 )
      goto LABEL_30;
    while ( 1 )
    {
      v52 = *v50;
      v53 = v50;
      if ( *v50 != -8192 && v52 != -4096 )
        break;
      v50 += 2;
      if ( v51 == v50 )
        goto LABEL_30;
    }
    if ( v50 == v51 )
    {
LABEL_30:
      v32 = 0;
      v33 = 0;
      v34 = &v103;
      goto LABEL_31;
    }
    v54 = 0;
    do
    {
      if ( *(_BYTE *)v52 == 84 && v88 == *(_QWORD *)(v52 + 40) )
      {
        if ( v54 + 1 > (unsigned __int64)HIDWORD(v102) )
        {
          v81 = v31;
          sub_C8D5F0((__int64)&v101, v31, v54 + 1, 8u, v29, v30);
          v54 = (unsigned int)v102;
          v31 = v81;
        }
        v101[v54] = v52;
        v54 = (unsigned int)(v102 + 1);
        LODWORD(v102) = v102 + 1;
      }
      v53 += 2;
      if ( v53 == v51 )
        break;
      while ( 1 )
      {
        v52 = *v53;
        if ( *v53 != -4096 && v52 != -8192 )
          break;
        v53 += 2;
        if ( v51 == v53 )
          goto LABEL_85;
      }
    }
    while ( v53 != v51 );
LABEL_85:
    v55 = v101;
    v33 = v98;
    v32 = v100;
    v34 = &v101[v54];
    if ( v34 == v101 )
      goto LABEL_31;
    v80 = v31;
    while ( 1 )
    {
      v62 = *v55;
      if ( !(_DWORD)v32 )
      {
        ++v97;
        goto LABEL_92;
      }
      v56 = 1;
      v57 = 0;
      v58 = (v32 - 1) & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
      v59 = (_QWORD *)(v33 + 16LL * v58);
      v60 = *v59;
      if ( v62 != *v59 )
        break;
LABEL_88:
      v61 = (unsigned __int8 **)(v59 + 1);
      if ( !v59[1] )
        goto LABEL_97;
LABEL_89:
      ++v55;
      v33 = v98;
      v32 = v100;
      if ( v34 == v55 )
      {
        v31 = v80;
        v34 = v101;
LABEL_31:
        v35 = v99;
        v36 = (__int64)v94;
        ++v93;
        LODWORD(v99) = v95;
        LODWORD(v95) = v35;
        v37 = HIDWORD(v99);
        HIDWORD(v99) = HIDWORD(v95);
        v38 = v96;
        ++v97;
        v94 = (__int64 *)v33;
        v98 = v36;
        HIDWORD(v95) = v37;
        v96 = v32;
        v100 = v38;
        if ( v34 != v31 )
        {
          _libc_free(v34, v32);
          v38 = v100;
          v36 = v98;
        }
        sub_C7D6A0(v36, 16LL * v38, 8);
        if ( v83 == ++v90 )
        {
LABEL_34:
          v39 = sub_D970F0(a1);
          goto LABEL_35;
        }
        goto LABEL_26;
      }
    }
    while ( v60 != -4096 )
    {
      if ( v60 == -8192 && !v57 )
        v57 = v59;
      v58 = (v32 - 1) & (v56 + v58);
      v59 = (_QWORD *)(v33 + 16LL * v58);
      v60 = *v59;
      if ( v62 == *v59 )
        goto LABEL_88;
      ++v56;
    }
    if ( !v57 )
      v57 = v59;
    ++v97;
    v64 = v99 + 1;
    if ( 4 * ((int)v99 + 1) < (unsigned int)(3 * v32) )
    {
      if ( (int)v32 - (v64 + HIDWORD(v99)) <= (unsigned int)v32 >> 3 )
      {
        sub_DA7360((__int64)&v97, v32);
        if ( !v100 )
        {
LABEL_163:
          LODWORD(v99) = v99 + 1;
          BUG();
        }
        v71 = 1;
        v72 = 0;
        LODWORD(v73) = (v100 - 1) & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
        v64 = v99 + 1;
        v57 = (_QWORD *)(v98 + 16LL * (unsigned int)v73);
        v74 = *v57;
        if ( v62 != *v57 )
        {
          while ( v74 != -4096 )
          {
            if ( v74 == -8192 && !v72 )
              v72 = v57;
            v73 = (v100 - 1) & ((_DWORD)v73 + v71);
            v57 = (_QWORD *)(v98 + 16 * v73);
            v74 = *v57;
            if ( v62 == *v57 )
              goto LABEL_94;
            ++v71;
          }
          if ( v72 )
            v57 = v72;
        }
      }
      goto LABEL_94;
    }
LABEL_92:
    sub_DA7360((__int64)&v97, 2 * v32);
    if ( !v100 )
      goto LABEL_163;
    LODWORD(v63) = (v100 - 1) & (((unsigned int)v62 >> 9) ^ ((unsigned int)v62 >> 4));
    v64 = v99 + 1;
    v57 = (_QWORD *)(v98 + 16LL * (unsigned int)v63);
    v65 = *v57;
    if ( v62 != *v57 )
    {
      v75 = 1;
      v76 = 0;
      while ( v65 != -4096 )
      {
        if ( v65 == -8192 && !v76 )
          v76 = v57;
        v63 = (v100 - 1) & ((_DWORD)v63 + v75);
        v57 = (_QWORD *)(v98 + 16 * v63);
        v65 = *v57;
        if ( v62 == *v57 )
          goto LABEL_94;
        ++v75;
      }
      if ( v76 )
        v57 = v76;
    }
LABEL_94:
    LODWORD(v99) = v64;
    if ( *v57 != -4096 )
      --HIDWORD(v99);
    *v57 = v62;
    v61 = (unsigned __int8 **)(v57 + 1);
    v57[1] = 0;
LABEL_97:
    v66 = *(_QWORD *)(v62 - 8);
    v67 = 0x1FFFFFFFE0LL;
    if ( (*(_DWORD *)(v62 + 4) & 0x7FFFFFF) != 0 )
    {
      v68 = 0;
      do
      {
        if ( v13 == *(_QWORD *)(v66 + 32LL * *(unsigned int *)(v62 + 72) + 8 * v68) )
        {
          v67 = 32 * v68;
          goto LABEL_102;
        }
        ++v68;
      }
      while ( (*(_DWORD *)(v62 + 4) & 0x7FFFFFF) != (_DWORD)v68 );
      v69 = *(unsigned __int8 **)(v66 + 0x1FFFFFFFE0LL);
      v70 = *v69;
      if ( *v69 <= 0x15u )
        goto LABEL_103;
    }
    else
    {
LABEL_102:
      v69 = *(unsigned __int8 **)(v66 + v67);
      v70 = *v69;
      if ( *v69 <= 0x15u )
      {
LABEL_103:
        *v61 = v69;
        goto LABEL_89;
      }
    }
    if ( v70 <= 0x1Cu )
      v69 = 0;
    else
      v69 = (unsigned __int8 *)sub_DA7540((__int64)v69, a2, (__int64)&v93, v82, *(__int64 **)(a1 + 24));
    goto LABEL_103;
  }
  v78 = (_QWORD *)sub_B2BE50(*(_QWORD *)a1);
  v79 = sub_BCB2D0(v78);
  v39 = (__int64)sub_DA2C50(a1, v79, v90, 0);
LABEL_35:
  sub_C7D6A0((__int64)v94, 16LL * v96, 8);
  return v39;
}
