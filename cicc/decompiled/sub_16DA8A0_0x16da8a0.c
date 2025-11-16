// Function: sub_16DA8A0
// Address: 0x16da8a0
//
__int64 __fastcall sub_16DA8A0(__int64 *a1)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  __int64 v4; // r15
  void *v5; // r15
  size_t v6; // r14
  __int64 v7; // rax
  __int64 v8; // r13
  signed __int64 v9; // rax
  __int64 v10; // rbx
  __int64 v11; // r9
  __int64 v12; // r14
  _QWORD *v13; // r13
  _QWORD *v14; // rdi
  _QWORD *v15; // rdi
  __int64 result; // rax
  int v17; // eax
  int v18; // eax
  int v19; // eax
  unsigned int v20; // ebx
  _QWORD *v21; // rcx
  __int64 v22; // rax
  unsigned int v23; // eax
  _QWORD *v24; // r13
  __int64 v25; // rax
  _QWORD *v26; // rcx
  _QWORD *v27; // r8
  _BYTE *v28; // rdi
  __int64 *v29; // rdx
  __int64 v30; // rax
  _BYTE *v31; // rax
  __int64 v32; // [rsp+0h] [rbp-50h]
  unsigned int v33; // [rsp+8h] [rbp-48h]
  _QWORD *v34; // [rsp+8h] [rbp-48h]
  _QWORD *v35; // [rsp+8h] [rbp-48h]
  __int64 v36; // [rsp+10h] [rbp-40h]
  _QWORD *v37; // [rsp+10h] [rbp-40h]
  _QWORD *v38; // [rsp+10h] [rbp-40h]
  __int64 v39; // [rsp+18h] [rbp-38h]
  _QWORD *v40; // [rsp+18h] [rbp-38h]

  v2 = *a1 + 80LL * *((unsigned int *)a1 + 2) - 80;
  v3 = sub_220F880();
  v4 = v3 - *(_QWORD *)v2;
  *(_QWORD *)(v2 + 8) = v3;
  v32 = v4;
  if ( *((unsigned int *)a1 + 2916) <= v4 / 1000 )
  {
    v23 = *((_DWORD *)a1 + 326);
    if ( v23 >= *((_DWORD *)a1 + 327) )
    {
      sub_16D99F0((unsigned __int64 *)a1 + 162);
      v23 = *((_DWORD *)a1 + 326);
    }
    v24 = (_QWORD *)(a1[162] + 80LL * v23);
    if ( v24 )
    {
      *v24 = *(_QWORD *)v2;
      v24[1] = *(_QWORD *)(v2 + 8);
      v24[2] = v24 + 4;
      sub_16D9890(v24 + 2, *(_BYTE **)(v2 + 16), *(_QWORD *)(v2 + 16) + *(_QWORD *)(v2 + 24));
      v24[6] = v24 + 8;
      sub_16D9890(v24 + 6, *(_BYTE **)(v2 + 48), *(_QWORD *)(v2 + 48) + *(_QWORD *)(v2 + 56));
      v23 = *((_DWORD *)a1 + 326);
    }
    *((_DWORD *)a1 + 326) = v23 + 1;
  }
  v5 = *(void **)(v2 + 16);
  v6 = *(_QWORD *)(v2 + 24);
  v33 = *((_DWORD *)a1 + 2);
  v36 = *a1;
  v7 = 80LL * v33 - 80;
  v8 = *a1 + v7;
  v9 = 0xCCCCCCCCCCCCCCCDLL * (v7 >> 4);
  if ( v9 >> 2 > 0 )
  {
    v10 = v8 - 320 * (v9 >> 2);
    while ( 1 )
    {
      if ( *(_QWORD *)(v8 - 56) == v6 && (!v6 || !memcmp(*(const void **)(v8 - 64), v5, v6)) )
        goto LABEL_11;
      v11 = v8 - 80;
      if ( *(_QWORD *)(v8 - 136) == v6 )
      {
        if ( !v6 || (v17 = memcmp(*(const void **)(v8 - 144), v5, v6), v11 = v8 - 80, !v17) )
        {
LABEL_18:
          v8 = v11;
          goto LABEL_11;
        }
      }
      v11 = v8 - 160;
      if ( *(_QWORD *)(v8 - 216) == v6 )
      {
        if ( !v6 )
          goto LABEL_18;
        v18 = memcmp(*(const void **)(v8 - 224), v5, v6);
        v11 = v8 - 160;
        if ( !v18 )
          goto LABEL_18;
        v11 = v8 - 240;
        if ( *(_QWORD *)(v8 - 296) == v6 )
          goto LABEL_22;
LABEL_7:
        v8 -= 320;
        if ( v10 == v8 )
          goto LABEL_25;
      }
      else
      {
        v11 = v8 - 240;
        if ( *(_QWORD *)(v8 - 296) != v6 )
          goto LABEL_7;
LABEL_22:
        if ( !v6 )
          goto LABEL_18;
        v39 = v11;
        v19 = memcmp(*(const void **)(v8 - 304), v5, v6);
        v11 = v39;
        if ( !v19 )
          goto LABEL_18;
        v8 -= 320;
        if ( v10 == v8 )
        {
LABEL_25:
          v9 = 0xCCCCCCCCCCCCCCCDLL * ((v8 - v36) >> 4);
          break;
        }
      }
    }
  }
  if ( v9 == 2 )
  {
LABEL_40:
    if ( *(_QWORD *)(v8 - 56) != v6 || v6 && memcmp(*(const void **)(v8 - 64), v5, v6) )
    {
      v8 -= 80;
      goto LABEL_42;
    }
    goto LABEL_11;
  }
  if ( v9 != 3 )
  {
    if ( v9 != 1 )
      goto LABEL_29;
LABEL_42:
    if ( *(_QWORD *)(v8 - 56) != v6 || v6 && memcmp(*(const void **)(v8 - 64), v5, v6) )
      goto LABEL_29;
    goto LABEL_11;
  }
  if ( *(_QWORD *)(v8 - 56) != v6 || v6 && memcmp(*(const void **)(v8 - 64), v5, v6) )
  {
    v8 -= 80;
    goto LABEL_40;
  }
LABEL_11:
  if ( v36 != v8 )
    goto LABEL_12;
LABEL_29:
  v20 = sub_16D19C0((__int64)(a1 + 1444), (unsigned __int8 *)v5, v6);
  v21 = (_QWORD *)(a1[1444] + 8LL * v20);
  v22 = *v21;
  if ( !*v21 )
    goto LABEL_47;
  if ( v22 == -8 )
  {
    --*((_DWORD *)a1 + 2892);
LABEL_47:
    v34 = v21;
    v25 = malloc(v6 + 25);
    v26 = v34;
    v27 = (_QWORD *)v25;
    if ( !v25 )
    {
      if ( v6 == -25 )
      {
        v30 = malloc(1u);
        v26 = v34;
        v27 = 0;
        if ( v30 )
        {
          v28 = (_BYTE *)(v30 + 24);
          v27 = (_QWORD *)v30;
          goto LABEL_58;
        }
      }
      v35 = v27;
      v38 = v26;
      sub_16BD1C0("Allocation failed", 1u);
      v26 = v38;
      v27 = v35;
    }
    v28 = v27 + 3;
    if ( v6 + 1 <= 1 )
    {
LABEL_49:
      v28[v6] = 0;
      *v27 = v6;
      v27[1] = 0;
      v27[2] = 0;
      *v26 = v27;
      ++*((_DWORD *)a1 + 2891);
      v29 = (__int64 *)(a1[1444] + 8LL * (unsigned int)sub_16D1CD0((__int64)(a1 + 1444), v20));
      v22 = *v29;
      if ( *v29 )
        goto LABEL_51;
      do
      {
        do
        {
          v22 = v29[1];
          ++v29;
        }
        while ( !v22 );
LABEL_51:
        ;
      }
      while ( v22 == -8 );
      goto LABEL_31;
    }
LABEL_58:
    v37 = v27;
    v40 = v26;
    v31 = memcpy(v28, v5, v6);
    v27 = v37;
    v26 = v40;
    v28 = v31;
    goto LABEL_49;
  }
LABEL_31:
  ++*(_QWORD *)(v22 + 8);
  *(_QWORD *)(v22 + 16) += v32;
  v33 = *((_DWORD *)a1 + 2);
  v36 = *a1;
LABEL_12:
  v12 = v33 - 1;
  *((_DWORD *)a1 + 2) = v12;
  v13 = (_QWORD *)(80 * v12 + v36);
  v14 = (_QWORD *)v13[6];
  if ( v14 != v13 + 8 )
    j_j___libc_free_0(v14, v13[8] + 1LL);
  v15 = (_QWORD *)v13[2];
  result = (__int64)(v13 + 4);
  if ( v15 != v13 + 4 )
    return j_j___libc_free_0(v15, v13[4] + 1LL);
  return result;
}
