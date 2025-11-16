// Function: sub_2F7B1D0
// Address: 0x2f7b1d0
//
void __fastcall sub_2F7B1D0(__int64 a1, __int64 a2, const void *a3, __int64 a4)
{
  size_t v4; // r12
  char *v8; // r15
  unsigned int v9; // esi
  char *v10; // rcx
  __int64 v11; // r9
  int v12; // r11d
  _QWORD *v13; // rdx
  unsigned int v14; // r8d
  _QWORD *v15; // rax
  __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  int v18; // eax
  int v19; // edi
  int v20; // eax
  int v21; // esi
  __int64 v22; // r8
  unsigned int v23; // eax
  __int64 v24; // r9
  int v25; // r11d
  _QWORD *v26; // r10
  int v27; // eax
  int v28; // eax
  __int64 v29; // r8
  _QWORD *v30; // r9
  unsigned int v31; // r12d
  int v32; // r10d
  __int64 v33; // rsi
  char *v34; // [rsp+8h] [rbp-38h]
  char *v35; // [rsp+8h] [rbp-38h]

  v4 = 4 * a4;
  if ( (unsigned __int64)(4 * a4) > 0x7FFFFFFFFFFFFFFCLL )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  if ( v4 )
  {
    v8 = (char *)sub_22077B0(4 * a4);
    memcpy(v8, a3, v4);
    v9 = *(_DWORD *)(a1 + 24);
    v10 = &v8[v4];
    if ( v9 )
      goto LABEL_4;
LABEL_22:
    ++*(_QWORD *)a1;
    goto LABEL_23;
  }
  v9 = *(_DWORD *)(a1 + 24);
  v10 = 0;
  v8 = 0;
  if ( !v9 )
    goto LABEL_22;
LABEL_4:
  v11 = *(_QWORD *)(a1 + 8);
  v12 = 1;
  v13 = 0;
  v14 = (v9 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v15 = (_QWORD *)(v11 + 32LL * v14);
  v16 = *v15;
  if ( a2 == *v15 )
  {
LABEL_5:
    v17 = v15[1];
    v15[1] = v8;
    v15[2] = v10;
    v15[3] = v10;
    if ( v17 )
      j_j___libc_free_0(v17);
    return;
  }
  while ( v16 != -4096 )
  {
    if ( v16 == -8192 && !v13 )
      v13 = v15;
    v14 = (v9 - 1) & (v12 + v14);
    v15 = (_QWORD *)(v11 + 32LL * v14);
    v16 = *v15;
    if ( a2 == *v15 )
      goto LABEL_5;
    ++v12;
  }
  if ( !v13 )
    v13 = v15;
  v18 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v19 = v18 + 1;
  if ( 4 * (v18 + 1) >= 3 * v9 )
  {
LABEL_23:
    v34 = v10;
    sub_2F7AE10(a1, 2 * v9);
    v20 = *(_DWORD *)(a1 + 24);
    if ( v20 )
    {
      v21 = v20 - 1;
      v22 = *(_QWORD *)(a1 + 8);
      v23 = (v20 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v19 = *(_DWORD *)(a1 + 16) + 1;
      v10 = v34;
      v13 = (_QWORD *)(v22 + 32LL * v23);
      v24 = *v13;
      if ( a2 != *v13 )
      {
        v25 = 1;
        v26 = 0;
        while ( v24 != -4096 )
        {
          if ( !v26 && v24 == -8192 )
            v26 = v13;
          v23 = v21 & (v25 + v23);
          v13 = (_QWORD *)(v22 + 32LL * v23);
          v24 = *v13;
          if ( a2 == *v13 )
            goto LABEL_17;
          ++v25;
        }
        if ( v26 )
          v13 = v26;
      }
      goto LABEL_17;
    }
    goto LABEL_47;
  }
  if ( v9 - *(_DWORD *)(a1 + 20) - v19 <= v9 >> 3 )
  {
    v35 = v10;
    sub_2F7AE10(a1, v9);
    v27 = *(_DWORD *)(a1 + 24);
    if ( v27 )
    {
      v28 = v27 - 1;
      v29 = *(_QWORD *)(a1 + 8);
      v30 = 0;
      v31 = v28 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v32 = 1;
      v19 = *(_DWORD *)(a1 + 16) + 1;
      v10 = v35;
      v13 = (_QWORD *)(v29 + 32LL * v31);
      v33 = *v13;
      if ( a2 != *v13 )
      {
        while ( v33 != -4096 )
        {
          if ( !v30 && v33 == -8192 )
            v30 = v13;
          v31 = v28 & (v32 + v31);
          v13 = (_QWORD *)(v29 + 32LL * v31);
          v33 = *v13;
          if ( a2 == *v13 )
            goto LABEL_17;
          ++v32;
        }
        if ( v30 )
          v13 = v30;
      }
      goto LABEL_17;
    }
LABEL_47:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_17:
  *(_DWORD *)(a1 + 16) = v19;
  if ( *v13 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v13 = a2;
  v13[1] = v8;
  v13[2] = v10;
  v13[3] = v10;
}
