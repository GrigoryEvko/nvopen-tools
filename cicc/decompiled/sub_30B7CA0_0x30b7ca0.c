// Function: sub_30B7CA0
// Address: 0x30b7ca0
//
void __fastcall sub_30B7CA0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  char *v5; // r12
  __int64 v6; // rax
  unsigned int v7; // eax
  __int64 v8; // rcx
  _QWORD *v9; // r8
  _QWORD *v10; // rsi
  __int64 v11; // rdi
  unsigned __int16 v12; // dx
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 *v17; // rbx
  __int64 *v18; // r14
  __int64 v19; // r15
  __int64 *v20; // rax
  __int64 v21; // rax
  unsigned __int64 v22; // rdx
  unsigned __int64 v23; // rdx
  char *v24; // r12
  __int64 v25; // rax
  char *v26; // rsi
  char *v27; // rbx
  __int64 v28; // rcx
  char *v29; // rax
  __int64 v30; // rax
  __int64 v31; // rbx
  char *v32; // r13
  unsigned __int64 v33; // rax
  char *v34; // rbx
  __int64 v35; // rdi
  char *i; // rax
  int v37; // r8d
  int v38; // esi
  __int64 v39; // rdx
  __int16 v40; // cx
  char *v41; // rbx
  __int64 v42; // r13
  __int64 v43; // rcx
  __int64 v44; // r8
  char *v45; // r9
  __int64 v46; // r13
  __int64 **v47; // rbx
  __int64 v48; // rax
  unsigned __int64 v49; // rdx
  __int64 *v50; // r12
  __int16 v51; // ax
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // r13
  __int64 **v55; // r14
  __int64 v56; // rbx
  __int64 v57; // r12
  __int64 v58; // rax
  unsigned __int64 v59; // rdx
  __int64 v60; // r8
  __int64 v61; // r9
  __int64 v62; // rax
  __int64 *v63; // rdi
  _QWORD *v64; // rdx
  size_t v65; // rdx
  __int64 v69; // [rsp+20h] [rbp-130h]
  __int64 v70; // [rsp+20h] [rbp-130h]
  _BYTE *v72; // [rsp+40h] [rbp-110h] BYREF
  __int64 v73; // [rsp+48h] [rbp-108h]
  _BYTE v74[16]; // [rsp+50h] [rbp-100h] BYREF
  __int64 *v75; // [rsp+60h] [rbp-F0h] BYREF
  __int64 v76; // [rsp+68h] [rbp-E8h] BYREF
  __int64 v77; // [rsp+70h] [rbp-E0h] BYREF
  _QWORD v78[8]; // [rsp+78h] [rbp-D8h] BYREF
  __int64 v79; // [rsp+B8h] [rbp-98h] BYREF
  __int64 *v80; // [rsp+C0h] [rbp-90h]
  __int64 v81; // [rsp+C8h] [rbp-88h]
  int v82; // [rsp+D0h] [rbp-80h]
  char v83; // [rsp+D4h] [rbp-7Ch]
  __int64 v84; // [rsp+D8h] [rbp-78h] BYREF

  v4 = *(unsigned int *)(a2 + 8);
  if ( !a4 || !(_DWORD)v4 )
    return;
  v5 = *(char **)a2;
  v69 = *(_QWORD *)a2 + 8 * v4;
  while ( 1 )
  {
    v6 = *(_QWORD *)v5;
    v83 = 1;
    LOBYTE(v72) = 0;
    v75 = (__int64 *)&v72;
    v77 = 0x800000000LL;
    v76 = (__int64)v78;
    v80 = &v84;
    v81 = 0x100000008LL;
    v82 = 0;
    v84 = v6;
    v79 = 1;
    if ( *(_WORD *)(v6 + 24) == 15 )
    {
      LOBYTE(v72) = 1;
      v7 = 0;
    }
    else
    {
      LODWORD(v77) = 1;
      v78[0] = v6;
      v7 = 1;
    }
    v8 = (__int64)&v72;
    v9 = v78;
LABEL_7:
    while ( 1 )
    {
      v10 = &v9[v7];
      if ( !v7 )
        break;
      while ( 1 )
      {
        if ( *(_BYTE *)v8 )
          goto LABEL_12;
        v11 = *(v10 - 1);
        LODWORD(v77) = --v7;
        v12 = *(_WORD *)(v11 + 24);
        if ( v12 > 0xEu )
        {
          if ( v12 != 15 )
            BUG();
          goto LABEL_11;
        }
        if ( v12 > 1u )
          break;
LABEL_11:
        --v10;
        if ( !v7 )
          goto LABEL_12;
      }
      v13 = sub_D960E0(v11);
      v17 = (__int64 *)(v13 + 8 * v14);
      v18 = (__int64 *)v13;
      if ( (__int64 *)v13 != v17 )
      {
        while ( 1 )
        {
          v19 = *v18;
          if ( !v83 )
            goto LABEL_30;
          v20 = v80;
          v8 = HIDWORD(v81);
          v14 = (__int64)&v80[HIDWORD(v81)];
          if ( v80 != (__int64 *)v14 )
          {
            while ( v19 != *v20 )
            {
              if ( (__int64 *)v14 == ++v20 )
                goto LABEL_35;
            }
            goto LABEL_27;
          }
LABEL_35:
          if ( HIDWORD(v81) < (unsigned int)v81 )
          {
            ++HIDWORD(v81);
            *(_QWORD *)v14 = v19;
            ++v79;
            if ( *(_WORD *)(v19 + 24) != 15 )
            {
LABEL_32:
              v21 = (unsigned int)v77;
              v22 = (unsigned int)v77 + 1LL;
              if ( v22 > HIDWORD(v77) )
              {
                sub_C8D5F0((__int64)&v76, v78, v22, 8u, v15, v16);
                v21 = (unsigned int)v77;
              }
              v14 = v76;
              *(_QWORD *)(v76 + 8 * v21) = v19;
              LODWORD(v77) = v77 + 1;
              goto LABEL_27;
            }
          }
          else
          {
LABEL_30:
            sub_C8CC70((__int64)&v79, *v18, v14, v8, v15, v16);
            if ( !(_BYTE)v14 )
              goto LABEL_27;
            if ( *(_WORD *)(v19 + 24) != 15 )
              goto LABEL_32;
          }
          *(_BYTE *)v75 = 1;
LABEL_27:
          v8 = (__int64)v75;
          if ( !*(_BYTE *)v75 && v17 != ++v18 )
            continue;
          v9 = (_QWORD *)v76;
          v7 = v77;
          goto LABEL_7;
        }
      }
      v8 = (__int64)v75;
      v9 = (_QWORD *)v76;
      v7 = v77;
    }
LABEL_12:
    if ( !v83 )
    {
      _libc_free((unsigned __int64)v80);
      v9 = (_QWORD *)v76;
    }
    if ( v9 != v78 )
      _libc_free((unsigned __int64)v9);
    if ( (_BYTE)v72 )
      break;
    v5 += 8;
    if ( (char *)v69 == v5 )
      return;
  }
  v23 = *(unsigned int *)(a2 + 8);
  v24 = *(char **)a2;
  v25 = 8 * v23;
  if ( v23 <= 1 )
  {
    v26 = &v24[v25];
    if ( &v24[v25] != v24 )
      goto LABEL_42;
LABEL_101:
    *(_DWORD *)(a2 + 8) = 0;
    goto LABEL_102;
  }
  qsort(*(void **)a2, v25 >> 3, 8u, (__compar_fn_t)sub_284F380);
  v24 = *(char **)a2;
  v26 = (char *)(*(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8));
  if ( v26 == *(char **)a2 )
    goto LABEL_101;
LABEL_42:
  v27 = v24;
  while ( 1 )
  {
    v29 = v27;
    v27 += 8;
    if ( v26 == v27 )
      break;
    v28 = *((_QWORD *)v27 - 1);
    if ( v28 == *(_QWORD *)v27 )
    {
      if ( v26 == v29 )
      {
        v27 = v26;
      }
      else
      {
        v64 = v29 + 16;
        if ( v29 + 16 != v26 )
        {
          while ( 1 )
          {
            if ( *v64 != v28 )
            {
              *((_QWORD *)v29 + 1) = *v64;
              v29 += 8;
            }
            if ( v26 == (char *)++v64 )
              break;
            v28 = *(_QWORD *)v29;
          }
          v24 = *(char **)a2;
          v65 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8) - (_QWORD)v26;
          v27 = &v29[v65 + 8];
          if ( v26 != (char *)(*(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8)) )
          {
            memmove(v29 + 8, v26, v65);
            v24 = *(char **)a2;
          }
        }
      }
      break;
    }
  }
  v30 = (v27 - v24) >> 3;
  *(_DWORD *)(a2 + 8) = v30;
  v31 = 8LL * (unsigned int)v30;
  v32 = &v24[v31];
  if ( &v24[v31] == v24 )
    goto LABEL_102;
  _BitScanReverse64(&v33, v31 >> 3);
  sub_30B5F20(v24, (__int64 *)&v24[v31], 2LL * (int)(63 - (v33 ^ 0x3F)));
  if ( (unsigned __int64)v31 <= 0x80 )
  {
    sub_30B61E0(v24, &v24[v31]);
    goto LABEL_59;
  }
  v34 = v24 + 128;
  sub_30B61E0(v24, v24 + 128);
  if ( v32 != v24 + 128 )
  {
    do
    {
      v35 = *(_QWORD *)v34;
      for ( i = v34; ; i -= 8 )
      {
        v39 = *((_QWORD *)i - 1);
        v40 = *(_WORD *)(v39 + 24);
        if ( *(_WORD *)(v35 + 24) == 6 )
        {
          v37 = *(_DWORD *)(v35 + 40);
          v38 = 1;
          if ( v40 != 6 )
            goto LABEL_50;
          goto LABEL_55;
        }
        if ( v40 != 6 )
          break;
        v37 = 1;
LABEL_55:
        v38 = *(_DWORD *)(v39 + 40);
LABEL_50:
        if ( v37 <= v38 )
          break;
        *(_QWORD *)i = v39;
      }
      v34 += 8;
      *(_QWORD *)i = v35;
    }
    while ( v32 != v34 );
  }
LABEL_59:
  v41 = *(char **)a2;
  v42 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 == v42 )
    goto LABEL_102;
  do
  {
    sub_310BF50(a1, *(_QWORD *)v41, a4, &v72, &v75);
    if ( !sub_D968A0((__int64)v72) )
      *(_QWORD *)v41 = v72;
    v41 += 8;
  }
  while ( (char *)v42 != v41 );
  v75 = &v77;
  v45 = *(char **)a2;
  v46 = *(_QWORD *)a2 + 8LL * *(unsigned int *)(a2 + 8);
  v76 = 0x400000000LL;
  if ( v45 == (char *)v46 )
  {
LABEL_102:
    *(_DWORD *)(a3 + 8) = 0;
    return;
  }
  v47 = (__int64 **)v45;
  do
  {
    while ( 1 )
    {
      v50 = *v47;
      v51 = *((_WORD *)*v47 + 12);
      if ( v51 )
        break;
LABEL_68:
      if ( (__int64 **)v46 == ++v47 )
        goto LABEL_83;
    }
    if ( v51 != 6 )
      goto LABEL_65;
    v73 = 0x200000000LL;
    v72 = v74;
    v52 = v50[4];
    v53 = v50[5];
    if ( v52 != v52 + 8 * v53 )
    {
      v70 = v46;
      v54 = v52 + 8 * v53;
      v55 = v47;
      v56 = v50[4];
      do
      {
        while ( 1 )
        {
          v57 = *(_QWORD *)v56;
          if ( *(_WORD *)(*(_QWORD *)v56 + 24LL) )
            break;
          v56 += 8;
          if ( v54 == v56 )
            goto LABEL_78;
        }
        v58 = (unsigned int)v73;
        v59 = (unsigned int)v73 + 1LL;
        if ( v59 > HIDWORD(v73) )
        {
          sub_C8D5F0((__int64)&v72, v74, v59, 8u, v44, (__int64)v45);
          v58 = (unsigned int)v73;
        }
        v56 += 8;
        *(_QWORD *)&v72[8 * v58] = v57;
        LODWORD(v73) = v73 + 1;
      }
      while ( v54 != v56 );
LABEL_78:
      v46 = v70;
      v47 = v55;
    }
    v50 = sub_DC8BD0(a1, (__int64)&v72, 0, 0);
    if ( v72 != v74 )
      _libc_free((unsigned __int64)v72);
    if ( v50 )
    {
LABEL_65:
      v48 = (unsigned int)v76;
      v43 = HIDWORD(v76);
      v49 = (unsigned int)v76 + 1LL;
      if ( v49 > HIDWORD(v76) )
      {
        sub_C8D5F0((__int64)&v75, &v77, v49, 8u, v44, (__int64)v45);
        v48 = (unsigned int)v76;
      }
      v75[v48] = (__int64)v50;
      LODWORD(v76) = v76 + 1;
      goto LABEL_68;
    }
    ++v47;
  }
  while ( (__int64 **)v46 != v47 );
LABEL_83:
  if ( (_DWORD)v76 && sub_30B62C0(a1, (__int64)&v75, a3, v43, v44) )
  {
    v62 = *(unsigned int *)(a3 + 8);
    if ( v62 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
    {
      sub_C8D5F0(a3, (const void *)(a3 + 16), v62 + 1, 8u, v60, v61);
      v62 = *(unsigned int *)(a3 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a3 + 8 * v62) = a4;
    ++*(_DWORD *)(a3 + 8);
    v63 = v75;
    if ( v75 != &v77 )
LABEL_88:
      _libc_free((unsigned __int64)v63);
  }
  else
  {
    v63 = v75;
    *(_DWORD *)(a3 + 8) = 0;
    if ( v63 != &v77 )
      goto LABEL_88;
  }
}
