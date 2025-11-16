// Function: sub_1415970
// Address: 0x1415970
//
char *__fastcall sub_1415970(__int64 a1, __int64 a2)
{
  unsigned int v3; // r8d
  __int64 v4; // rdi
  __int64 v5; // rcx
  unsigned int v6; // eax
  __int64 *v7; // rdx
  __int64 v8; // r10
  char *result; // rax
  int v10; // r11d
  __int64 *v11; // r12
  int v12; // eax
  int v13; // edx
  __int64 v14; // r13
  __int64 v15; // r14
  __int64 v16; // r15
  __int64 v17; // r15
  char *v18; // r14
  __int64 v19; // rax
  __int64 v20; // r15
  unsigned int v21; // esi
  int v22; // r14d
  __int64 v23; // rcx
  __int64 v24; // r8
  unsigned int v25; // edx
  __int64 *v26; // rax
  __int64 v27; // rdi
  __int64 v28; // rcx
  __int64 v29; // r14
  __int64 v30; // rdx
  __int64 v31; // r15
  unsigned __int64 v32; // r13
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // r15
  _BYTE *v36; // rcx
  __int64 v37; // r14
  __int64 v38; // rax
  __int64 *v39; // rax
  int v40; // eax
  __int64 v41; // rdi
  int v42; // esi
  __int64 v43; // r8
  unsigned int v44; // eax
  int v45; // r10d
  __int64 *v46; // r9
  int v47; // eax
  int v48; // esi
  __int64 v49; // r8
  int v50; // r10d
  unsigned int v51; // eax
  int v52; // r11d
  __int64 *v53; // r10
  int v54; // edi
  int v55; // edi
  unsigned int v56; // [rsp+Ch] [rbp-164h]
  char *v57; // [rsp+10h] [rbp-160h]
  __int64 v58; // [rsp+18h] [rbp-158h] BYREF
  __int64 *v59; // [rsp+28h] [rbp-148h] BYREF
  void *src; // [rsp+30h] [rbp-140h] BYREF
  __int64 v61; // [rsp+38h] [rbp-138h]
  _BYTE v62[304]; // [rsp+40h] [rbp-130h] BYREF

  v3 = *(_DWORD *)(a1 + 24);
  v58 = a2;
  if ( !v3 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_55;
  }
  v4 = *(_QWORD *)(a1 + 8);
  v5 = a2;
  v6 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v7 = (__int64 *)(v4 + 16LL * v6);
  v8 = *v7;
  if ( a2 != *v7 )
  {
    v10 = 1;
    v11 = 0;
    while ( v8 != -8 )
    {
      if ( v8 == -16 && !v11 )
        v11 = v7;
      v6 = (v3 - 1) & (v10 + v6);
      v7 = (__int64 *)(v4 + 16LL * v6);
      v8 = *v7;
      if ( a2 == *v7 )
        goto LABEL_3;
      ++v10;
    }
    v12 = *(_DWORD *)(a1 + 16);
    if ( !v11 )
      v11 = v7;
    ++*(_QWORD *)a1;
    v13 = v12 + 1;
    if ( 4 * (v12 + 1) < 3 * v3 )
    {
      if ( v3 - *(_DWORD *)(a1 + 20) - v13 > v3 >> 3 )
      {
LABEL_11:
        *(_DWORD *)(a1 + 16) = v13;
        if ( *v11 != -8 )
          --*(_DWORD *)(a1 + 20);
        v11[1] = 0;
        *v11 = v5;
        a2 = v58;
        goto LABEL_14;
      }
      sub_14157B0(a1, v3);
      v47 = *(_DWORD *)(a1 + 24);
      if ( v47 )
      {
        v48 = v47 - 1;
        v49 = *(_QWORD *)(a1 + 8);
        v50 = 1;
        v46 = 0;
        v41 = v58;
        v51 = (v47 - 1) & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
        v13 = *(_DWORD *)(a1 + 16) + 1;
        v11 = (__int64 *)(v49 + 16LL * v51);
        v5 = *v11;
        if ( v58 == *v11 )
          goto LABEL_11;
        while ( v5 != -8 )
        {
          if ( v5 == -16 && !v46 )
            v46 = v11;
          v51 = v48 & (v50 + v51);
          v11 = (__int64 *)(v49 + 16LL * v51);
          v5 = *v11;
          if ( v58 == *v11 )
            goto LABEL_11;
          ++v50;
        }
        goto LABEL_59;
      }
      goto LABEL_98;
    }
LABEL_55:
    sub_14157B0(a1, 2 * v3);
    v40 = *(_DWORD *)(a1 + 24);
    if ( v40 )
    {
      v41 = v58;
      v42 = v40 - 1;
      v43 = *(_QWORD *)(a1 + 8);
      v44 = (v40 - 1) & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
      v13 = *(_DWORD *)(a1 + 16) + 1;
      v11 = (__int64 *)(v43 + 16LL * v44);
      v5 = *v11;
      if ( v58 == *v11 )
        goto LABEL_11;
      v45 = 1;
      v46 = 0;
      while ( v5 != -8 )
      {
        if ( !v46 && v5 == -16 )
          v46 = v11;
        v44 = v42 & (v45 + v44);
        v11 = (__int64 *)(v43 + 16LL * v44);
        v5 = *v11;
        if ( v58 == *v11 )
          goto LABEL_11;
        ++v45;
      }
LABEL_59:
      v5 = v41;
      if ( v46 )
        v11 = v46;
      goto LABEL_11;
    }
LABEL_98:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_3:
  result = (char *)v7[1];
  if ( result )
    return result;
  v11 = v7;
LABEL_14:
  v14 = *(_QWORD *)(a2 + 8);
  if ( v14 )
  {
    while ( (unsigned __int8)(*(_BYTE *)(sub_1648700(v14) + 16) - 25) > 9u )
    {
      v14 = *(_QWORD *)(v14 + 8);
      if ( !v14 )
        goto LABEL_46;
    }
    v15 = v14;
    v16 = 0;
    src = v62;
    v61 = 0x2000000000LL;
    while ( 1 )
    {
      v15 = *(_QWORD *)(v15 + 8);
      if ( !v15 )
        break;
      while ( (unsigned __int8)(*(_BYTE *)(sub_1648700(v15) + 16) - 25) <= 9u )
      {
        v15 = *(_QWORD *)(v15 + 8);
        ++v16;
        if ( !v15 )
          goto LABEL_20;
      }
    }
LABEL_20:
    v17 = v16 + 1;
    if ( v17 > 32 )
    {
      sub_16CD150(&src, v62, v17, 8);
      v18 = (char *)src + 8 * (unsigned int)v61;
    }
    else
    {
      v18 = v62;
    }
    v19 = sub_1648700(v14);
LABEL_25:
    if ( v18 )
      *(_QWORD *)v18 = *(_QWORD *)(v19 + 40);
    while ( 1 )
    {
      v14 = *(_QWORD *)(v14 + 8);
      if ( !v14 )
        break;
      v19 = sub_1648700(v14);
      if ( (unsigned __int8)(*(_BYTE *)(v19 + 16) - 25) <= 9u )
      {
        v18 += 8;
        goto LABEL_25;
      }
    }
    v20 = (unsigned int)(v61 + v17);
    LODWORD(v61) = v20;
    if ( (unsigned int)v20 >= HIDWORD(v61) )
    {
      sub_16CD150(&src, v62, 0, 8);
      v20 = (unsigned int)v61;
    }
  }
  else
  {
LABEL_46:
    v20 = 0;
    src = v62;
    v61 = 0x2000000000LL;
  }
  *((_QWORD *)src + v20) = 0;
  v21 = *(_DWORD *)(a1 + 56);
  v22 = v61;
  LODWORD(v61) = v61 + 1;
  if ( !v21 )
  {
    ++*(_QWORD *)(a1 + 32);
    goto LABEL_80;
  }
  v23 = v58;
  v24 = *(_QWORD *)(a1 + 40);
  v25 = (v21 - 1) & (((unsigned int)v58 >> 9) ^ ((unsigned int)v58 >> 4));
  v26 = (__int64 *)(v24 + 16LL * v25);
  v27 = *v26;
  if ( *v26 != v58 )
  {
    v52 = 1;
    v53 = 0;
    while ( v27 != -8 )
    {
      if ( !v53 && v27 == -16 )
        v53 = v26;
      v25 = (v21 - 1) & (v52 + v25);
      v26 = (__int64 *)(v24 + 16LL * v25);
      v27 = *v26;
      if ( v58 == *v26 )
        goto LABEL_32;
      ++v52;
    }
    v54 = *(_DWORD *)(a1 + 48);
    if ( v53 )
      v26 = v53;
    ++*(_QWORD *)(a1 + 32);
    v55 = v54 + 1;
    if ( 4 * v55 < 3 * v21 )
    {
      if ( v21 - *(_DWORD *)(a1 + 52) - v55 > v21 >> 3 )
      {
LABEL_76:
        *(_DWORD *)(a1 + 48) = v55;
        if ( *v26 != -8 )
          --*(_DWORD *)(a1 + 52);
        *v26 = v23;
        *((_DWORD *)v26 + 2) = 0;
        goto LABEL_32;
      }
LABEL_81:
      sub_13FEAC0(a1 + 32, v21);
      sub_13FDDE0(a1 + 32, &v58, &v59);
      v26 = v59;
      v23 = v58;
      v55 = *(_DWORD *)(a1 + 48) + 1;
      goto LABEL_76;
    }
LABEL_80:
    v21 *= 2;
    goto LABEL_81;
  }
LABEL_32:
  *((_DWORD *)v26 + 2) = v22;
  v28 = *(_QWORD *)(a1 + 72);
  v29 = 8LL * (unsigned int)v61;
  v30 = *(_QWORD *)(a1 + 64);
  *(_QWORD *)(a1 + 144) += v29;
  if ( v29 + ((v30 + 7) & 0xFFFFFFFFFFFFFFF8LL) - v30 <= v28 - v30 )
  {
    result = (char *)((v30 + 7) & 0xFFFFFFFFFFFFFFF8LL);
    *(_QWORD *)(a1 + 64) = &result[v29];
  }
  else
  {
    v31 = v29 + 7;
    if ( (unsigned __int64)(v29 + 7) > 0x1000 )
    {
      v37 = malloc(v29 + 7);
      if ( !v37 )
        sub_16BD1C0("Allocation failed");
      v38 = *(unsigned int *)(a1 + 136);
      if ( (unsigned int)v38 >= *(_DWORD *)(a1 + 140) )
      {
        sub_16CD150(a1 + 128, a1 + 144, 0, 16);
        v38 = *(unsigned int *)(a1 + 136);
      }
      v39 = (__int64 *)(*(_QWORD *)(a1 + 128) + 16 * v38);
      *v39 = v37;
      v39[1] = v31;
      ++*(_DWORD *)(a1 + 136);
      result = (char *)((v37 + 7) & 0xFFFFFFFFFFFFFFF8LL);
    }
    else
    {
      v32 = 0x40000000000LL;
      v56 = *(_DWORD *)(a1 + 88);
      if ( v56 >> 7 < 0x1E )
        v32 = 4096LL << (v56 >> 7);
      v33 = malloc(v32);
      v34 = v56;
      v35 = v33;
      if ( !v33 )
      {
        sub_16BD1C0("Allocation failed");
        v34 = *(unsigned int *)(a1 + 88);
      }
      if ( (unsigned int)v34 >= *(_DWORD *)(a1 + 92) )
      {
        sub_16CD150(a1 + 80, a1 + 96, 0, 8);
        v34 = *(unsigned int *)(a1 + 88);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 80) + 8 * v34) = v35;
      *(_QWORD *)(a1 + 72) = v35 + v32;
      result = (char *)((v35 + 7) & 0xFFFFFFFFFFFFFFF8LL);
      ++*(_DWORD *)(a1 + 88);
      *(_QWORD *)(a1 + 64) = &result[v29];
    }
  }
  v11[1] = (__int64)result;
  v36 = src;
  if ( 8LL * (unsigned int)v61 )
  {
    memmove(result, src, 8LL * (unsigned int)v61);
    result = (char *)v11[1];
    v36 = src;
  }
  if ( v36 != v62 )
  {
    v57 = result;
    _libc_free((unsigned __int64)v36);
    return v57;
  }
  return result;
}
