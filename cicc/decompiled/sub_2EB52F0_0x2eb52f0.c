// Function: sub_2EB52F0
// Address: 0x2eb52f0
//
_QWORD *__fastcall sub_2EB52F0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  void *v7; // rdi
  const void *v8; // r8
  size_t v9; // r15
  unsigned __int64 v10; // r13
  __int64 v11; // r14
  __int64 v13; // rcx
  __int64 v14; // r9
  __int64 v15; // r8
  __int64 v16; // rdi
  __int64 v17; // rdx
  __int64 *v18; // r13
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 *v21; // rbx
  __int64 *v22; // r14
  __int64 v23; // rsi
  __int64 v25; // r13
  int v26; // esi
  const void *v27; // [rsp+8h] [rbp-38h]
  const void *v28; // [rsp+8h] [rbp-38h]

  v7 = a1 + 2;
  v8 = *(const void **)(a2 + 64);
  v9 = 8LL * *(unsigned int *)(a2 + 72);
  v10 = *(unsigned int *)(a2 + 72);
  if ( !a3 )
  {
    *a1 = v7;
    a1[1] = 0x800000000LL;
    if ( v9 > 0x40 )
    {
      v28 = v8;
      sub_C8D5F0((__int64)a1, v7, v10, 8u, (__int64)v8, a6);
      v8 = v28;
      v7 = (void *)(*a1 + 8LL * *((unsigned int *)a1 + 2));
    }
    else if ( !v9 )
    {
LABEL_20:
      *((_DWORD *)a1 + 2) = v9 + v10;
      sub_2E6E9F0((__int64)a1);
      return a1;
    }
    memcpy(v7, v8, v9);
    LODWORD(v9) = *((_DWORD *)a1 + 2);
    goto LABEL_20;
  }
  v11 = *(_QWORD *)(a3 + 8);
  *a1 = v7;
  a1[1] = 0x800000000LL;
  if ( v9 > 0x40 )
  {
    v27 = v8;
    sub_C8D5F0((__int64)a1, v7, v10, 8u, (__int64)v8, a6);
    v8 = v27;
    v7 = (void *)(*a1 + 8LL * *((unsigned int *)a1 + 2));
  }
  else if ( !v9 )
  {
    goto LABEL_4;
  }
  memcpy(v7, v8, v9);
  LODWORD(v9) = *((_DWORD *)a1 + 2);
LABEL_4:
  *((_DWORD *)a1 + 2) = v10 + v9;
  sub_2E6E9F0((__int64)a1);
  v15 = *(_BYTE *)(v11 + 8) & 1;
  if ( (*(_BYTE *)(v11 + 8) & 1) != 0 )
  {
    v16 = v11 + 16;
    v13 = 3;
  }
  else
  {
    v17 = *(unsigned int *)(v11 + 24);
    v16 = *(_QWORD *)(v11 + 16);
    if ( !(_DWORD)v17 )
      goto LABEL_24;
    v13 = (unsigned int)(v17 - 1);
  }
  v17 = (unsigned int)v13 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v18 = (__int64 *)(v16 + 72 * v17);
  v19 = *v18;
  if ( a2 != *v18 )
  {
    v26 = 1;
    while ( v19 != -4096 )
    {
      v14 = (unsigned int)(v26 + 1);
      v17 = (unsigned int)v13 & (v26 + (_DWORD)v17);
      v18 = (__int64 *)(v16 + 72LL * (unsigned int)v17);
      v19 = *v18;
      if ( a2 == *v18 )
        goto LABEL_7;
      ++v26;
    }
    if ( (_BYTE)v15 )
    {
      v25 = 288;
      goto LABEL_25;
    }
    v17 = *(unsigned int *)(v11 + 24);
LABEL_24:
    v25 = 72 * v17;
LABEL_25:
    v18 = (__int64 *)(v16 + v25);
  }
LABEL_7:
  v20 = 288;
  if ( !(_BYTE)v15 )
    v20 = 72LL * *(unsigned int *)(v11 + 24);
  if ( v18 != (__int64 *)(v16 + v20) )
  {
    v21 = (__int64 *)v18[1];
    v22 = &v21[*((unsigned int *)v18 + 4)];
    while ( v22 != v21 )
    {
      v23 = *v21++;
      sub_2E6EB60((__int64)a1, v23);
    }
    sub_2E6EC00((__int64)a1, (__int64)(v18 + 5), v17, v13, v15, v14);
  }
  return a1;
}
