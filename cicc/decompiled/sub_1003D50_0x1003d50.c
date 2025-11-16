// Function: sub_1003D50
// Address: 0x1003d50
//
__int64 __fastcall sub_1003D50(
        __int64 a1,
        unsigned __int8 *a2,
        _DWORD *a3,
        __int64 a4,
        __int64 **a5,
        __int64 a6,
        int a7)
{
  __int64 v9; // r13
  _DWORD *v10; // r12
  __int64 v12; // r9
  __int64 v13; // rsi
  __int64 v14; // rcx
  __int64 v15; // r11
  _DWORD *v16; // rcx
  __int64 v17; // r10
  __int64 v18; // r8
  unsigned int *v19; // rdx
  unsigned int *v20; // rax
  char v21; // cl
  unsigned int v22; // edx
  int *v23; // rax
  int v24; // edx
  int v25; // edi
  unsigned __int8 *v26; // rax
  __int64 v27; // r8
  _BYTE *v28; // r10
  __int64 v29; // rdx
  _DWORD *v30; // rcx
  __int64 v31; // rdx
  _DWORD *v32; // rax
  _DWORD *v33; // rdx
  __int64 v34; // r12
  __int64 v35; // rdx
  __int64 i; // rcx
  __int64 v37; // rax
  int v38; // edi
  __int64 v39; // rcx
  unsigned int *v41; // rdi
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  char v45; // al
  __int64 v46; // rax
  int v47; // eax
  bool v48; // r11
  __int64 v49; // rdx
  __int64 v50; // r12
  __int64 v51; // rax
  _QWORD *v52; // rdx
  __int64 v53; // rcx
  char *v54; // rax
  char *v55; // rdi
  __int64 v56; // rcx
  int v57; // ecx
  int v58; // ecx
  int v59; // ecx
  __int64 *v60; // rax
  __int64 v61; // rbx
  __int64 v62; // rax
  signed __int64 v63; // rcx
  __int64 *v64; // rax
  __int64 *v65; // rdx
  int v66; // [rsp+Ch] [rbp-184h]
  __int64 v67; // [rsp+10h] [rbp-180h]
  int v68; // [rsp+10h] [rbp-180h]
  int v69; // [rsp+10h] [rbp-180h]
  size_t na; // [rsp+18h] [rbp-178h]
  int nb; // [rsp+18h] [rbp-178h]
  int nc; // [rsp+18h] [rbp-178h]
  int nd; // [rsp+18h] [rbp-178h]
  char n; // [rsp+18h] [rbp-178h]
  __int64 v75; // [rsp+20h] [rbp-170h]
  __int64 v76; // [rsp+20h] [rbp-170h]
  __int64 v77; // [rsp+20h] [rbp-170h]
  __int64 v78; // [rsp+20h] [rbp-170h]
  __int64 v79; // [rsp+20h] [rbp-170h]
  __int64 **v80; // [rsp+20h] [rbp-170h]
  __int64 v82; // [rsp+28h] [rbp-168h]
  char v83; // [rsp+3Ch] [rbp-154h]
  int v84; // [rsp+3Ch] [rbp-154h]
  unsigned int *v85; // [rsp+40h] [rbp-150h] BYREF
  __int64 v86; // [rsp+48h] [rbp-148h]
  _BYTE v87[128]; // [rsp+50h] [rbp-140h] BYREF
  __int64 *v88; // [rsp+D0h] [rbp-C0h] BYREF
  __int64 v89; // [rsp+D8h] [rbp-B8h]
  _QWORD v90[22]; // [rsp+E0h] [rbp-B0h] BYREF

  v9 = a1;
  v10 = a3;
  v12 = 4 * a4;
  v13 = (__int64)&a3[a4];
  v14 = (4 * a4) >> 4;
  v15 = v12 >> 2;
  if ( v14 > 0 )
  {
    v16 = &a3[4 * v14];
    while ( *a3 == -1 )
    {
      if ( a3[1] != -1 )
      {
        ++a3;
        goto LABEL_8;
      }
      if ( a3[2] != -1 )
      {
        a3 += 2;
        goto LABEL_8;
      }
      if ( a3[3] != -1 )
      {
        a3 += 3;
        goto LABEL_8;
      }
      a3 += 4;
      if ( v16 == a3 )
      {
        v39 = (v13 - (__int64)a3) >> 2;
        goto LABEL_61;
      }
    }
    goto LABEL_8;
  }
  v39 = v12 >> 2;
LABEL_61:
  if ( v39 == 2 )
    goto LABEL_92;
  if ( v39 == 3 )
  {
    if ( *a3 != -1 )
      goto LABEL_8;
    ++a3;
LABEL_92:
    if ( *a3 != -1 )
      goto LABEL_8;
    ++a3;
    goto LABEL_64;
  }
  if ( v39 != 1 )
    return sub_ACADE0(a5);
LABEL_64:
  if ( *a3 == -1 )
    return sub_ACADE0(a5);
LABEL_8:
  if ( (_DWORD *)v13 == a3 )
    return sub_ACADE0(a5);
  v17 = *(_QWORD *)(a1 + 8);
  v18 = *(unsigned int *)(v17 + 32);
  v83 = *(_BYTE *)(v17 + 8);
  v85 = (unsigned int *)v87;
  v86 = 0x2000000000LL;
  if ( (unsigned __int64)v12 > 0x80 )
  {
    v66 = v18;
    v67 = v17;
    na = v12;
    v75 = v12 >> 2;
    sub_C8D5F0((__int64)&v85, v87, v12 >> 2, 4u, v18, v12);
    LODWORD(v15) = v75;
    v12 = na;
    v17 = v67;
    LODWORD(v18) = v66;
    v41 = &v85[(unsigned int)v86];
  }
  else
  {
    if ( !v12 )
      goto LABEL_11;
    v41 = (unsigned int *)v87;
  }
  v13 = (__int64)v10;
  v68 = v15;
  nb = v18;
  v76 = v17;
  memcpy(v41, v10, v12);
  v12 = (unsigned int)v86;
  LODWORD(v15) = v68;
  LODWORD(v18) = nb;
  v17 = v76;
LABEL_11:
  LODWORD(v86) = v12 + v15;
  if ( v83 == 18 )
    goto LABEL_23;
  if ( !(_DWORD)a4 )
  {
    nc = v18;
    v77 = v17;
    v42 = sub_ACADE0((__int64 **)v17);
    v17 = v77;
    LODWORD(v18) = nc;
    v9 = v42;
LABEL_74:
    nd = v18;
    v78 = v17;
    v43 = sub_ACADE0((__int64 **)v17);
    LODWORD(v18) = nd;
    v17 = v78;
    a2 = (unsigned __int8 *)v43;
    goto LABEL_23;
  }
  v19 = v85;
  v13 = 0;
  v20 = v85 + 1;
  v21 = 0;
  while ( 1 )
  {
    v22 = *v19;
    if ( v22 != -1 )
    {
      if ( v22 >= (unsigned int)v18 )
        v21 = 1;
      else
        v13 = 1;
    }
    v19 = v20;
    if ( v20 == &v85[(unsigned int)(a4 - 1) + 1] )
      break;
    ++v20;
  }
  if ( !(_BYTE)v13 )
  {
    v69 = v18;
    n = v21;
    v79 = v17;
    v44 = sub_ACADE0((__int64 **)v17);
    LODWORD(v18) = v69;
    v21 = n;
    v17 = v79;
    v9 = v44;
  }
  if ( !v21 )
    goto LABEL_74;
LABEL_23:
  if ( *(_BYTE *)v9 > 0x15u )
  {
    if ( v83 != 18 )
      goto LABEL_33;
  }
  else
  {
    if ( *a2 <= 0x15u )
    {
      v13 = (__int64)a2;
      v34 = sub_AD5CE0(v9, (__int64)a2, v10, a4, 0);
LABEL_69:
      v28 = v85;
      goto LABEL_70;
    }
    if ( v83 != 18 )
    {
      v23 = (int *)v85;
      v13 = (__int64)&v85[(unsigned int)v86];
      if ( v85 != (unsigned int *)v13 )
      {
        do
        {
          v24 = *v23;
          if ( *v23 != -1 )
          {
            v25 = v18 + v24;
            if ( v24 >= (int)v18 )
              v25 = v24 - v18;
            *v23 = v25;
          }
          ++v23;
        }
        while ( (int *)v13 != v23 );
      }
      v26 = (unsigned __int8 *)v9;
      v9 = (__int64)a2;
      a2 = v26;
LABEL_33:
      if ( *(_BYTE *)v9 != 91 )
      {
        if ( *(_BYTE *)v9 != 92 )
        {
LABEL_35:
          v27 = (__int64)v85;
          goto LABEL_36;
        }
        goto LABEL_80;
      }
      if ( (*(_BYTE *)(v9 + 7) & 0x40) != 0 )
      {
        v49 = *(_QWORD *)(v9 - 8);
        v50 = *(_QWORD *)(v49 + 32);
        if ( *(_BYTE *)v50 > 0x15u )
          goto LABEL_35;
      }
      else
      {
        v49 = v9 - 32LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF);
        v50 = *(_QWORD *)(v49 + 32);
        if ( *(_BYTE *)v50 > 0x15u )
          goto LABEL_35;
      }
      v51 = *(_QWORD *)(v49 + 64);
      if ( *(_BYTE *)v51 != 17 )
        goto LABEL_35;
      v52 = *(_QWORD **)(v51 + 24);
      if ( *(_DWORD *)(v51 + 32) > 0x40u )
        v52 = (_QWORD *)*v52;
      v27 = (__int64)v85;
      v53 = (unsigned int)v86;
      v54 = (char *)v85;
      v55 = (char *)&v85[v53];
      v56 = (v53 * 4) >> 4;
      if ( v56 )
      {
        v13 = (__int64)&v85[4 * v56];
        while ( (_DWORD)v52 == *(_DWORD *)v54 || *(_DWORD *)v54 == -1 )
        {
          v57 = *((_DWORD *)v54 + 1);
          if ( (_DWORD)v52 != v57 && v57 != -1 )
          {
            v54 += 4;
            break;
          }
          v58 = *((_DWORD *)v54 + 2);
          if ( (_DWORD)v52 != v58 && v58 != -1 )
          {
            v54 += 8;
            break;
          }
          v59 = *((_DWORD *)v54 + 3);
          if ( (_DWORD)v52 != v59 && v59 != -1 )
          {
            v54 += 12;
            break;
          }
          v54 += 16;
          if ( (char *)v13 == v54 )
            goto LABEL_139;
        }
LABEL_110:
        if ( v55 != v54 )
          goto LABEL_36;
        goto LABEL_111;
      }
LABEL_139:
      v63 = v55 - v54;
      if ( v55 - v54 != 8 )
      {
        if ( v63 != 12 )
        {
          if ( v63 != 4 )
            goto LABEL_111;
          goto LABEL_142;
        }
        if ( (_DWORD)v52 != *(_DWORD *)v54 && *(_DWORD *)v54 != -1 )
          goto LABEL_110;
        v54 += 4;
      }
      if ( (_DWORD)v52 != *(_DWORD *)v54 && *(_DWORD *)v54 != -1 )
        goto LABEL_110;
      v54 += 4;
LABEL_142:
      if ( (_DWORD)v52 != *(_DWORD *)v54 && *(_DWORD *)v54 != -1 )
        goto LABEL_110;
LABEL_111:
      v88 = v90;
      v89 = 0x1000000000LL;
      if ( (unsigned int)a4 > 0x10 )
      {
        sub_C8D5F0((__int64)&v88, v90, (unsigned int)a4, 8u, (__int64)v85, v12);
        v64 = v88;
        v65 = &v88[(unsigned int)a4];
        do
          *v64++ = v50;
        while ( v65 != v64 );
      }
      else if ( (_DWORD)a4 )
      {
        v60 = v90;
        do
          *v60++ = v50;
        while ( &v90[(unsigned int)a4] != v60 );
      }
      LODWORD(v89) = a4;
      v61 = 0;
      if ( (_DWORD)a4 )
      {
        do
        {
          if ( v85[v61] == -1 )
          {
            v62 = sub_ACADE0(*(__int64 ***)(v50 + 8));
            v88[v61] = v62;
          }
          ++v61;
        }
        while ( (unsigned int)a4 != v61 );
      }
      v13 = (unsigned int)v89;
      v34 = sub_AD3730(v88, (unsigned int)v89);
      if ( v88 != v90 )
        _libc_free(v88, v13);
      goto LABEL_69;
    }
  }
  if ( *(_BYTE *)v9 != 92 )
  {
    v28 = v85;
    v34 = 0;
    goto LABEL_70;
  }
LABEL_80:
  v13 = (__int64)a2;
  v80 = (__int64 **)v17;
  v45 = sub_1003090(a6, a2);
  v27 = (__int64)v85;
  if ( v80 == a5 )
  {
    if ( v45 )
    {
      v82 = (__int64)v85;
      v46 = 4LL * *(unsigned int *)(v9 + 80);
      if ( !v46
        || v46 == 4
        || (v13 = *(_QWORD *)(v9 + 72),
            v47 = memcmp((const void *)(v13 + 4), (const void *)v13, v46 - 4),
            v27 = v82,
            !v47) )
      {
        v28 = v85;
        v34 = v9;
        goto LABEL_70;
      }
    }
  }
LABEL_36:
  v28 = (_BYTE *)v27;
  if ( v83 == 18 )
  {
LABEL_59:
    v34 = 0;
    goto LABEL_70;
  }
  v29 = 4LL * (unsigned int)v86;
  v30 = (_DWORD *)(v27 + v29);
  v13 = v29 >> 2;
  v31 = v29 >> 4;
  if ( !v31 )
  {
    v32 = (_DWORD *)v27;
LABEL_126:
    if ( v13 != 2 )
    {
      if ( v13 != 3 )
      {
        if ( v13 != 1 )
          goto LABEL_45;
        goto LABEL_129;
      }
      if ( *v32 == -1 )
        goto LABEL_44;
      ++v32;
    }
    if ( *v32 == -1 )
      goto LABEL_44;
    ++v32;
LABEL_129:
    if ( *v32 == -1 )
      goto LABEL_44;
LABEL_45:
    if ( (_DWORD)a4 )
    {
      v35 = 0;
      for ( i = 0; ; i = v34 )
      {
        v37 = *(unsigned int *)(v27 + 4 * v35);
        if ( (_DWORD)v37 == -1 || !a7 )
          break;
        v13 = (__int64)a2;
        v34 = v9;
        v84 = a7 - 1;
        while ( 1 )
        {
          v38 = *(_DWORD *)(*(_QWORD *)(v34 + 8) + 32LL);
          if ( v38 <= (int)v37 )
          {
            v37 = (unsigned int)(v37 - v38);
            v34 = v13;
          }
          if ( *(_BYTE *)v34 != 92 )
            break;
          LODWORD(v37) = *(_DWORD *)(*(_QWORD *)(v34 + 72) + 4 * v37);
          v13 = *(_QWORD *)(v34 - 32);
          v48 = v84 == 0;
          v34 = *(_QWORD *)(v34 - 64);
          --v84;
          if ( (_DWORD)v37 == -1 || v48 )
            goto LABEL_59;
        }
        if ( i && i != v34 || (_DWORD)v37 != (_DWORD)v35 || a5 != *(__int64 ***)(v34 + 8) )
          break;
        if ( (unsigned int)a4 == ++v35 )
          goto LABEL_70;
      }
    }
    goto LABEL_59;
  }
  v32 = (_DWORD *)v27;
  v33 = (_DWORD *)(v27 + 16 * v31);
  while ( *v32 != -1 )
  {
    if ( v32[1] == -1 )
    {
      ++v32;
      break;
    }
    if ( v32[2] == -1 )
    {
      v32 += 2;
      break;
    }
    if ( v32[3] == -1 )
    {
      v32 += 3;
      break;
    }
    v32 += 4;
    if ( v33 == v32 )
    {
      v13 = v30 - v32;
      goto LABEL_126;
    }
  }
LABEL_44:
  v34 = 0;
  if ( v30 == v32 )
    goto LABEL_45;
LABEL_70:
  if ( v28 != v87 )
    _libc_free(v28, v13);
  return v34;
}
