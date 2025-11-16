// Function: sub_39F9D50
// Address: 0x39f9d50
//
unsigned int *__fastcall sub_39F9D50(__int64 a1, unsigned __int64 a2)
{
  char v3; // dl
  __int64 v4; // rbx
  unsigned __int16 v5; // ax
  __int16 v6; // ax
  unsigned __int8 v7; // dl
  char *v8; // rcx
  char *v9; // r14
  unsigned __int64 v10; // rbx
  unsigned __int64 v11; // r12
  __int64 v12; // rbp
  unsigned __int64 v13; // r13
  char *v14; // rax
  __int64 v15; // r14
  __int64 v17; // rbx
  char **v18; // rbp
  char *v19; // rsi
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // rcx
  unsigned __int64 v23; // rax
  unsigned __int64 v24; // rsi
  unsigned __int64 v25; // r15
  unsigned __int64 v26; // r14
  unsigned __int64 v27; // rbp
  __int64 v28; // r13
  char v29; // al
  char v30; // bl
  char v31; // di
  unsigned __int8 v32; // al
  char *v33; // rsi
  char *v34; // rax
  unsigned int **v35; // rbp
  unsigned int *j; // rsi
  unsigned int *v37; // rax
  _QWORD *v38; // rax
  __int64 v39; // rax
  char **v40; // r14
  char *v41; // rdx
  __int64 (__fastcall *v42)(__int64, __int64, __int64); // rbp
  __int64 v43; // r15
  _QWORD *v44; // rbx
  __int64 *v45; // r12
  __int64 *v46; // r14
  __int64 v47; // rax
  __int64 v48; // rsi
  __int64 v49; // rdx
  __int64 i; // rax
  __int64 v51; // rcx
  __int64 v52; // r10
  __int64 v53; // rax
  _QWORD *v54; // r12
  __int64 v55; // r14
  __int64 v56; // r15
  __int64 v57; // rbx
  __int64 v58; // rbp
  _QWORD *v59; // r14
  _QWORD *v60; // rax
  char v61; // [rsp+0h] [rbp-88h]
  __int64 v62; // [rsp+0h] [rbp-88h]
  __int64 v63; // [rsp+0h] [rbp-88h]
  __int64 (__fastcall *v64)(__int64, __int64, __int64); // [rsp+0h] [rbp-88h]
  char v65; // [rsp+8h] [rbp-80h]
  __int64 v67; // [rsp+8h] [rbp-80h]
  __int64 v68; // [rsp+8h] [rbp-80h]
  __int64 v69; // [rsp+10h] [rbp-78h]
  unsigned __int64 v70; // [rsp+10h] [rbp-78h]
  __int64 v71; // [rsp+18h] [rbp-70h]
  _QWORD *v72; // [rsp+18h] [rbp-70h]
  unsigned __int64 v73; // [rsp+18h] [rbp-70h]
  unsigned __int64 v74; // [rsp+20h] [rbp-68h]
  _QWORD *v75; // [rsp+20h] [rbp-68h]
  unsigned __int64 v76; // [rsp+28h] [rbp-60h]
  unsigned __int64 v77; // [rsp+38h] [rbp-50h] BYREF
  _QWORD *v78; // [rsp+40h] [rbp-48h] BYREF
  __int64 v79; // [rsp+48h] [rbp-40h]

  v3 = *(_BYTE *)(a1 + 32);
  if ( (v3 & 1) != 0 )
  {
LABEL_2:
    v4 = *(_QWORD *)(a1 + 24);
    if ( (v3 & 4) == 0 )
    {
      v5 = *(_WORD *)(a1 + 32);
      if ( (v5 & 0x7F8) == 0 )
      {
        v21 = *(_QWORD *)(v4 + 8);
        v22 = 0;
        while ( v21 > v22 )
        {
          v23 = (v21 + v22) >> 1;
          v15 = *(_QWORD *)(v4 + 8 * v23 + 16);
          v24 = *(_QWORD *)(v15 + 8);
          if ( a2 >= v24 )
          {
            if ( a2 < *(_QWORD *)(v15 + 16) + v24 )
              return (unsigned int *)v15;
            v22 = v23 + 1;
          }
          else
          {
            v21 = (v21 + v22) >> 1;
          }
        }
        return 0;
      }
      v6 = v5 >> 3;
      v61 = v6;
      if ( (_BYTE)v6 == 0xFF )
        goto LABEL_59;
      v7 = v6 & 0x70;
      if ( (v6 & 0x70) == 0x20 )
      {
        v8 = *(char **)(a1 + 8);
        goto LABEL_9;
      }
      if ( v7 <= 0x20u )
      {
        if ( (v6 & 0x60) == 0 )
          goto LABEL_59;
      }
      else
      {
        if ( v7 == 48 )
        {
          v8 = *(char **)(a1 + 16);
LABEL_9:
          if ( !*(_QWORD *)(v4 + 8) )
            return 0;
          v71 = *(_QWORD *)(a1 + 24);
          v65 = v6 & 0xF;
          v9 = v8;
          v10 = *(_QWORD *)(v4 + 8);
          v74 = a2;
          v11 = 0;
          while ( 1 )
          {
            while ( 1 )
            {
              v12 = *(_QWORD *)(v71 + 8 * ((v10 + v11) >> 1) + 16);
              v13 = (v10 + v11) >> 1;
              v14 = sub_39F8BA0(v61, v9, (char *)(v12 + 8), &v77);
              sub_39F8BA0(v65, 0, v14, (unsigned __int64 *)&v78);
              if ( v74 >= v77 )
                break;
              v10 = (v10 + v11) >> 1;
              if ( v11 >= v13 )
                return 0;
            }
            if ( v74 < (unsigned __int64)v78 + v77 )
              break;
            v11 = v13 + 1;
            if ( v13 + 1 >= v10 )
              return 0;
          }
          return (unsigned int *)v12;
        }
        if ( v7 == 80 )
        {
LABEL_59:
          v8 = 0;
          goto LABEL_9;
        }
      }
      goto LABEL_115;
    }
    v25 = *(_QWORD *)(v4 + 8);
    if ( !v25 )
      return 0;
    v62 = *(_QWORD *)(a1 + 24);
    v26 = 0;
    while ( 1 )
    {
      v27 = (v25 + v26) >> 1;
      v28 = *(_QWORD *)(v62 + 8 * v27 + 16);
      v29 = sub_39F8CF0(v28 + 4 - *(int *)(v28 + 4));
      v30 = v29;
      v31 = v29;
      if ( v29 == -1 )
        goto LABEL_54;
      v32 = v29 & 0x70;
      if ( v32 != 32 )
        break;
      v33 = *(char **)(a1 + 8);
LABEL_39:
      v34 = sub_39F8BA0(v31, v33, (char *)(v28 + 8), &v77);
      sub_39F8BA0(v30 & 0xF, 0, v34, (unsigned __int64 *)&v78);
      if ( a2 >= v77 )
      {
        if ( a2 < (unsigned __int64)v78 + v77 )
          return (unsigned int *)v28;
        v26 = v27 + 1;
      }
      else
      {
        v25 = (v25 + v26) >> 1;
      }
      if ( v26 >= v25 )
        return 0;
    }
    if ( v32 <= 0x20u )
    {
      if ( (v30 & 0x60) != 0 )
        goto LABEL_115;
    }
    else
    {
      if ( v32 == 48 )
      {
        v33 = *(char **)(a1 + 16);
        goto LABEL_39;
      }
      if ( v32 != 80 )
        goto LABEL_115;
    }
LABEL_54:
    v33 = 0;
    goto LABEL_39;
  }
  v17 = *(_DWORD *)(a1 + 32) >> 11;
  v69 = v17;
  if ( !(*(_DWORD *)(a1 + 32) >> 11) )
  {
    v18 = *(char ***)(a1 + 24);
    if ( (*(_BYTE *)(a1 + 32) & 2) != 0 )
    {
      v19 = *v18;
      if ( !*v18 )
        goto LABEL_45;
      do
      {
        v20 = sub_39F8E20(a1, v19);
        if ( v20 == -1 )
          goto LABEL_23;
        v19 = v18[1];
        ++v18;
        v17 += v20;
      }
      while ( v19 );
      v69 = v17;
    }
    else
    {
      v69 = sub_39F8E20(a1, *(char **)(a1 + 24));
      if ( v69 == -1 )
      {
LABEL_23:
        *(_QWORD *)(a1 + 32) = 2040;
        *(_QWORD *)(a1 + 24) = &unk_4535488;
        goto LABEL_45;
      }
    }
    if ( (v69 & 0xFFFFFFFFFFE00000LL) != 0 )
      *(_DWORD *)(a1 + 32) &= 0x7FFu;
    else
      *(_DWORD *)(a1 + 32) = ((_DWORD)v69 << 11) | *(_DWORD *)(a1 + 32) & 0x7FF;
    if ( !v69 )
      goto LABEL_45;
  }
  v38 = (_QWORD *)malloc(8 * v69 + 16);
  v78 = v38;
  if ( !v38 )
    goto LABEL_45;
  v38[1] = 0;
  v39 = malloc(8 * v69 + 16);
  v79 = v39;
  if ( v39 )
    *(_QWORD *)(v39 + 8) = 0;
  v40 = *(char ***)(a1 + 24);
  if ( (*(_BYTE *)(a1 + 32) & 2) != 0 )
  {
    v41 = *v40;
    if ( !*v40 )
      goto LABEL_115;
    do
    {
      ++v40;
      sub_39F91E0(a1, (__int64 *)&v78, v41);
      v41 = *v40;
    }
    while ( *v40 );
  }
  else
  {
    sub_39F91E0(a1, (__int64 *)&v78, *(char **)(a1 + 24));
  }
  v72 = v78;
  v75 = v78;
  if ( v78 && v78[1] != v69 )
    goto LABEL_115;
  v42 = sub_39F90C0;
  if ( (*(_BYTE *)(a1 + 32) & 4) == 0 )
  {
    v42 = sub_39F8A10;
    if ( (*(_WORD *)(a1 + 32) & 0x7F8) != 0 )
      v42 = sub_39F9010;
  }
  v43 = v79;
  if ( !v79 )
  {
    sub_39F8AF0(a1, v42, (__int64)v78);
    goto LABEL_103;
  }
  v67 = v78[1];
  if ( v67 )
  {
    v63 = 0;
    v76 = a2;
    v44 = v78 + 2;
    v45 = (__int64 *)&unk_5057720;
    v46 = v78 + 2;
    while ( 1 )
    {
      *(_QWORD *)(v43 + 8 * v63++ + 16) = v45;
      if ( v67 == v63 )
        break;
      if ( v46 == (__int64 *)&unk_5057720 )
      {
LABEL_84:
        v45 = (__int64 *)&unk_5057720;
        ++v46;
      }
      else
      {
        v45 = v46;
        while ( (int)v42(a1, v46[1], *v45) < 0 )
        {
          v47 = v45 - v44 + 2;
          v45 = *(__int64 **)(v43 + 8 * v47);
          *(_QWORD *)(v43 + 8 * v47) = 0;
          if ( v45 == (__int64 *)&unk_5057720 )
            goto LABEL_84;
        }
        ++v46;
      }
    }
    a2 = v76;
    v48 = 0;
    v49 = 0;
    for ( i = 0; i != v67; ++i )
    {
      v51 = v72[i + 2];
      if ( *(_QWORD *)(v43 + 8 * i + 16) )
        v72[v49++ + 2] = v51;
      else
        *(_QWORD *)(v43 + 8 * v48++ + 16) = v51;
    }
    v52 = v79;
    v72 = v78;
  }
  else
  {
    v52 = v79;
    v48 = 0;
    v49 = 0;
  }
  v75[1] = v49;
  *(_QWORD *)(v43 + 8) = v48;
  if ( v72[1] + *(_QWORD *)(v52 + 8) != v69 )
LABEL_115:
    abort();
  sub_39F8AF0(a1, v42, v52);
  v70 = v79;
  v53 = *(_QWORD *)(v79 + 8);
  if ( !v53 )
    goto LABEL_102;
  v64 = v42;
  v73 = a2;
  v54 = v78;
  v55 = v78[1];
  while ( 1 )
  {
    v68 = v53 - 1;
    v56 = *(_QWORD *)(v70 + 8 * (v53 + 1));
    if ( !v55 )
      break;
    v57 = v55 + v53 - 1;
    v58 = v55;
    while ( 1 )
    {
      v55 = v58--;
      if ( (int)v64(a1, v54[v58 + 2], v56) <= 0 )
        break;
      v54[v57-- + 2] = v54[v58 + 2];
      if ( !v58 )
      {
        v57 = v68;
        v55 = 0;
        break;
      }
    }
LABEL_98:
    v54[v57 + 2] = v56;
    if ( !v68 )
      goto LABEL_101;
    v53 = v68;
  }
  v54[v53 + 1] = v56;
  if ( v53 != 1 )
  {
    v57 = v53 - 2;
    v68 = v53 - 2;
    v56 = *(_QWORD *)(v70 + 8 * v53);
    goto LABEL_98;
  }
LABEL_101:
  v59 = v54;
  a2 = v73;
  v59[1] += *(_QWORD *)(v70 + 8);
  v70 = v79;
LABEL_102:
  _libc_free(v70);
LABEL_103:
  v60 = v78;
  *v78 = *(_QWORD *)(a1 + 24);
  *(_BYTE *)(a1 + 32) |= 1u;
  *(_QWORD *)(a1 + 24) = v60;
LABEL_45:
  if ( *(_QWORD *)a1 > a2 )
    return 0;
  v3 = *(_BYTE *)(a1 + 32);
  v35 = *(unsigned int ***)(a1 + 24);
  if ( (v3 & 1) != 0 )
    goto LABEL_2;
  if ( (*(_BYTE *)(a1 + 32) & 2) != 0 )
  {
    for ( j = *v35; j; ++v35 )
    {
      v37 = sub_39F9490(a1, j, a2);
      if ( v37 )
        return v37;
      j = v35[1];
    }
    return 0;
  }
  return sub_39F9490(a1, *(unsigned int **)(a1 + 24), a2);
}
