// Function: sub_E21F00
// Address: 0xe21f00
//
unsigned __int64 __fastcall sub_E21F00(__int64 a1, size_t *a2)
{
  _QWORD *v4; // rdx
  __int64 v5; // rax
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // r12
  __int64 v8; // rsi
  char v9; // r14
  char *v10; // rdx
  char v11; // cl
  char v13; // dl
  unsigned __int64 v14; // rcx
  unsigned __int64 v15; // r15
  unsigned __int64 v16; // rax
  _BYTE *v17; // rax
  char *v18; // rdi
  size_t v19; // rax
  unsigned __int64 v20; // rcx
  size_t v21; // rax
  _BYTE *v22; // rdx
  char v23; // dl
  __int64 *v24; // rax
  __int64 *v25; // r14
  __int64 v26; // rax
  __int64 v27; // rax
  unsigned __int64 v28; // r14
  unsigned int i; // edx
  _BYTE *v30; // rax
  unsigned int v31; // esi
  _BYTE *v32; // rdi
  unsigned int v33; // r10d
  unsigned int v34; // r9d
  size_t v35; // rsi
  size_t v36; // rax
  const void *v37; // rdi
  __int64 v38; // rdx
  _BYTE *v39; // r15
  unsigned int v40; // eax
  _BYTE *v41; // r14
  unsigned int v42; // r8d
  int v43; // r14d
  __int64 v44; // rbx
  __int64 v45; // rax
  unsigned int v46; // esi
  int v47; // edx
  char v48; // cl
  unsigned int v49; // [rsp+4h] [rbp-FCh]
  char *v50; // [rsp+8h] [rbp-F8h]
  unsigned int v51; // [rsp+8h] [rbp-F8h]
  size_t v52; // [rsp+10h] [rbp-F0h]
  unsigned int v53; // [rsp+10h] [rbp-F0h]
  unsigned __int64 v54; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v55; // [rsp+18h] [rbp-E8h]
  const void *v56; // [rsp+20h] [rbp-E0h] BYREF
  size_t v57; // [rsp+28h] [rbp-D8h]
  __int64 v58; // [rsp+30h] [rbp-D0h]
  __int64 v59; // [rsp+38h] [rbp-C8h]
  int v60; // [rsp+40h] [rbp-C0h]
  _BYTE v62[176]; // [rsp+50h] [rbp-B0h] BYREF

  v4 = *(_QWORD **)(a1 + 16);
  v58 = 0;
  v59 = -1;
  v60 = 1;
  v57 = 0;
  v5 = v4[1];
  v56 = 0;
  v6 = (*v4 + v5 + 7) & 0xFFFFFFFFFFFFFFF8LL;
  v4[1] = v6 - *v4 + 48;
  if ( *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL) <= *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL) )
  {
    v7 = 0;
    if ( !v6 )
      goto LABEL_3;
    v7 = v6;
LABEL_29:
    *(_BYTE *)(v7 + 40) = 0;
    *(_DWORD *)(v7 + 8) = 22;
    *(_QWORD *)(v7 + 16) = 0;
    *(_QWORD *)v7 = &unk_49E0EE8;
    *(_QWORD *)(v7 + 24) = 0;
    *(_QWORD *)(v7 + 32) = 0;
    *(_DWORD *)(v7 + 44) = 0;
    goto LABEL_3;
  }
  v24 = (__int64 *)sub_22077B0(32);
  v25 = v24;
  if ( v24 )
  {
    *v24 = 0;
    v24[1] = 0;
    v24[2] = 0;
    v24[3] = 0;
  }
  v26 = sub_2207820(4096);
  v25[2] = 4096;
  *v25 = v26;
  v7 = v26;
  v27 = *(_QWORD *)(a1 + 16);
  v25[1] = 48;
  v25[3] = v27;
  *(_QWORD *)(a1 + 16) = v25;
  if ( v7 )
    goto LABEL_29;
LABEL_3:
  v8 = 2;
  v9 = sub_E20730(a2, 2u, "@_");
  if ( !v9 || !*a2 )
    goto LABEL_7;
  v10 = (char *)a2[1];
  v11 = *v10;
  --*a2;
  a2[1] = (size_t)(v10 + 1);
  if ( v11 == 48 )
  {
    v8 = (__int64)a2;
    v14 = sub_E219C0(a1, a2);
    v15 = v14;
    v9 = *(_BYTE *)(a1 + 8) | v23;
    if ( v9 )
    {
LABEL_7:
      *(_BYTE *)(a1 + 8) = 1;
      v7 = 0;
      _libc_free(v56, v8);
      return v7;
    }
    v16 = 1;
  }
  else
  {
    if ( v11 != 49 )
      goto LABEL_7;
    v8 = (__int64)a2;
    v14 = sub_E219C0(a1, a2);
    v15 = v14;
    if ( *(_BYTE *)(a1 + 8) || v13 )
      goto LABEL_7;
    v16 = 2;
  }
  v54 = v14;
  if ( v14 < v16 )
    goto LABEL_7;
  if ( !*a2 )
    goto LABEL_7;
  v52 = *a2;
  v50 = (char *)a2[1];
  v17 = memchr(v50, 64, *a2);
  v8 = (__int64)v17;
  if ( !v17 )
    goto LABEL_7;
  v8 = v17 - v50;
  if ( v17 - v50 == -1 )
    goto LABEL_7;
  v18 = &v50[++v8];
  a2[1] = (size_t)&v50[v8];
  v19 = v52 - v8;
  *a2 = v52 - v8;
  if ( v52 == v8 )
    goto LABEL_7;
  v20 = v54;
  if ( v9 )
  {
    *(_DWORD *)(v7 + 44) = 3;
    if ( v54 > 0x40 )
      *(_BYTE *)(v7 + 40) = 1;
    while ( 1 )
    {
      v21 = *a2;
      if ( !*a2 )
        goto LABEL_7;
      v22 = (_BYTE *)a2[1];
      if ( *v22 == 64 )
      {
        a2[1] = (size_t)(v22 + 1);
        *a2 = v21 - 1;
        goto LABEL_52;
      }
      if ( v21 != 1 )
      {
        v8 = (unsigned int)sub_E21E90(a1, (__int64 *)a2);
        if ( v15 != 2 || *(_BYTE *)(v7 + 40) )
          sub_E20DD0((__int64 *)&v56, v8);
        v15 -= 2LL;
        if ( !*(_BYTE *)(a1 + 8) )
          continue;
      }
      goto LABEL_7;
    }
  }
  v28 = 0;
  for ( i = 0; *v18 != 64; i = v28 )
  {
    v55 = v20;
    if ( v28 == 128 )
      goto LABEL_7;
    v8 = (__int64)a2;
    v62[v28] = sub_E21D30(a1, (__int64 *)a2);
    v19 = *a2;
    ++v28;
    if ( !*a2 )
      goto LABEL_7;
    v18 = (char *)a2[1];
    v20 = v55;
  }
  a2[1] = (size_t)(v18 + 1);
  *a2 = v19 - 1;
  if ( v20 > v28 )
    *(_BYTE *)(v7 + 40) = 1;
  if ( (v20 & 1) != 0 )
    goto LABEL_58;
  if ( v20 <= 0x1F )
  {
    v30 = &v62[v28 - 1];
    if ( i )
    {
      v31 = 0;
      v32 = &v30[~(unsigned __int64)(i - 1)];
      do
      {
        if ( *v30 )
          break;
        --v30;
        ++v31;
      }
      while ( v30 != v32 );
      if ( v31 > 3 )
      {
LABEL_49:
        if ( (v20 & 3) != 0 )
        {
LABEL_50:
          v33 = 2;
          *(_DWORD *)(v7 + 44) = 1;
          v34 = i >> 1;
          goto LABEL_59;
        }
        goto LABEL_68;
      }
      if ( v31 > 1 )
        goto LABEL_50;
    }
LABEL_58:
    *(_DWORD *)(v7 + 44) = 0;
    v34 = i;
    v33 = 1;
    goto LABEL_59;
  }
  if ( !i )
    goto LABEL_49;
  v39 = v62;
  v40 = 0;
  v41 = &v62[v28];
  do
    v40 += *v39++ == 0;
  while ( v41 != v39 );
  if ( 2 * i / 3 > v40 || (v20 & 3) != 0 )
  {
    if ( i / 3 <= v40 )
      goto LABEL_50;
    goto LABEL_58;
  }
LABEL_68:
  v33 = 4;
  *(_DWORD *)(v7 + 44) = 2;
  v34 = i >> 2;
LABEL_59:
  if ( v33 <= i )
  {
    v42 = 0;
    v43 = 0;
    v44 = v33;
    while ( 1 )
    {
      v45 = 0;
      v46 = 0;
      do
      {
        v47 = (unsigned __int8)v62[v42 + v45];
        v48 = 8 * v45++;
        v46 |= v47 << v48;
      }
      while ( v44 != v45 );
      if ( ++v43 >= v34 )
        break;
      v49 = v42;
      v51 = v33;
      v53 = v34;
      sub_E20DD0((__int64 *)&v56, v46);
      v33 = v51;
      v34 = v53;
      v42 = v51 + v49;
    }
    if ( *(_BYTE *)(v7 + 40) )
      sub_E20DD0((__int64 *)&v56, v46);
  }
LABEL_52:
  v35 = v57;
  v36 = sub_E213F0(a1, v57, v56);
  v37 = v56;
  *(_QWORD *)(v7 + 24) = v36;
  *(_QWORD *)(v7 + 32) = v38;
  _libc_free(v37, v35);
  return v7;
}
