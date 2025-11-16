// Function: sub_2448450
// Address: 0x2448450
//
void __fastcall sub_2448450(__int64 a1, unsigned int a2)
{
  unsigned int v2; // r13d
  __int64 v4; // r14
  char v5; // dl
  unsigned __int64 v6; // rax
  __int64 v7; // r15
  __int64 v8; // rdi
  __int64 v9; // rax
  bool v10; // zf
  __int64 v11; // r13
  _QWORD *v12; // rax
  unsigned __int64 v13; // rdx
  _QWORD *i; // rdx
  __int64 j; // rbx
  __int64 v16; // rax
  __int64 v17; // rcx
  int v18; // esi
  int v19; // r10d
  unsigned __int64 v20; // r9
  unsigned int v21; // edx
  unsigned __int64 v22; // rdi
  __int64 v23; // r8
  _QWORD *v24; // r15
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // rdi
  int v27; // edx
  __int64 v28; // rbx
  __int64 *v29; // r14
  __int64 v30; // rax
  __m128i *v31; // rdi
  _QWORD *v32; // r15
  unsigned __int64 v33; // rdi
  unsigned __int64 v34; // rdi
  __int64 v35; // rax
  __int64 v36; // [rsp+8h] [rbp-138h]
  __int64 v37; // [rsp+8h] [rbp-138h]
  __int64 v38[38]; // [rsp+10h] [rbp-130h] BYREF

  v2 = a2;
  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(_BYTE *)(a1 + 8) & 1;
  if ( a2 <= 4 )
  {
    if ( !v5 )
    {
      v7 = *(unsigned int *)(a1 + 24);
      *(_BYTE *)(a1 + 8) |= 1u;
      goto LABEL_8;
    }
    v28 = a1 + 16;
    v37 = a1 + 272;
    goto LABEL_30;
  }
  v6 = ((((((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
          | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
          | (a2 - 1)
          | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
        | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
        | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
        | (a2 - 1)
        | ((unsigned __int64)(a2 - 1) >> 1)) >> 16)
      | (((((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
        | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
        | (a2 - 1)
        | ((unsigned __int64)(a2 - 1) >> 1)) >> 8)
      | (((((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2) | (a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 4)
      | (((a2 - 1) | ((unsigned __int64)(a2 - 1) >> 1)) >> 2)
      | (a2 - 1)
      | ((unsigned __int64)(a2 - 1) >> 1))
     + 1;
  v2 = v6;
  if ( (unsigned int)v6 > 0x40 )
  {
    v28 = a1 + 16;
    v37 = a1 + 272;
    if ( !v5 )
    {
      v7 = *(unsigned int *)(a1 + 24);
      v8 = (unsigned __int64)(unsigned int)v6 << 6;
      goto LABEL_5;
    }
LABEL_30:
    v29 = v38;
    do
    {
      v30 = *(_QWORD *)v28;
      if ( *(_QWORD *)v28 != -4096 && v30 != -8192 )
      {
        if ( v29 )
          *v29 = v30;
        v31 = (__m128i *)(v29 + 1);
        v29 += 8;
        sub_24481E0(v31, (__m128i *)(v28 + 8));
        v32 = *(_QWORD **)(v28 + 24);
        while ( v32 )
        {
          v33 = (unsigned __int64)v32;
          v32 = (_QWORD *)*v32;
          j_j___libc_free_0(v33);
        }
        memset(*(void **)(v28 + 8), 0, 8LL * *(_QWORD *)(v28 + 16));
        v34 = *(_QWORD *)(v28 + 8);
        *(_QWORD *)(v28 + 32) = 0;
        *(_QWORD *)(v28 + 24) = 0;
        if ( v34 != v28 + 56 )
          j_j___libc_free_0(v34);
      }
      v28 += 64;
    }
    while ( v28 != v37 );
    if ( v2 > 4 )
    {
      *(_BYTE *)(a1 + 8) &= ~1u;
      v35 = sub_C7D670((unsigned __int64)v2 << 6, 8);
      *(_DWORD *)(a1 + 24) = v2;
      *(_QWORD *)(a1 + 16) = v35;
    }
    sub_2448280(a1, v38, v29);
    return;
  }
  if ( v5 )
  {
    v28 = a1 + 16;
    v2 = 64;
    v37 = a1 + 272;
    goto LABEL_30;
  }
  v7 = *(unsigned int *)(a1 + 24);
  v2 = 64;
  v8 = 4096;
LABEL_5:
  v9 = sub_C7D670(v8, 8);
  *(_DWORD *)(a1 + 24) = v2;
  *(_QWORD *)(a1 + 16) = v9;
LABEL_8:
  v10 = (*(_QWORD *)(a1 + 8) & 1LL) == 0;
  *(_QWORD *)(a1 + 8) &= 1uLL;
  v36 = v7 << 6;
  v11 = v4 + (v7 << 6);
  if ( v10 )
  {
    v12 = *(_QWORD **)(a1 + 16);
    v13 = (unsigned __int64)*(unsigned int *)(a1 + 24) << 6;
  }
  else
  {
    v12 = (_QWORD *)(a1 + 16);
    v13 = 256;
  }
  for ( i = (_QWORD *)((char *)v12 + v13); i != v12; v12 += 8 )
  {
    if ( v12 )
      *v12 = -4096;
  }
  for ( j = v4; v11 != j; j += 64 )
  {
    v16 = *(_QWORD *)j;
    if ( *(_QWORD *)j != -4096 && v16 != -8192 )
    {
      if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
      {
        v17 = a1 + 16;
        v18 = 3;
      }
      else
      {
        v27 = *(_DWORD *)(a1 + 24);
        v17 = *(_QWORD *)(a1 + 16);
        if ( !v27 )
        {
          MEMORY[0] = *(_QWORD *)j;
          BUG();
        }
        v18 = v27 - 1;
      }
      v19 = 1;
      v20 = 0;
      v21 = v18 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v22 = v17 + ((unsigned __int64)v21 << 6);
      v23 = *(_QWORD *)v22;
      if ( v16 != *(_QWORD *)v22 )
      {
        while ( v23 != -4096 )
        {
          if ( !v20 && v23 == -8192 )
            v20 = v22;
          v21 = v18 & (v19 + v21);
          v22 = v17 + ((unsigned __int64)v21 << 6);
          v23 = *(_QWORD *)v22;
          if ( v16 == *(_QWORD *)v22 )
            goto LABEL_20;
          ++v19;
        }
        if ( v20 )
          v22 = v20;
      }
LABEL_20:
      *(_QWORD *)v22 = v16;
      sub_24481E0((__m128i *)(v22 + 8), (__m128i *)(j + 8));
      *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
      v24 = *(_QWORD **)(j + 24);
      while ( v24 )
      {
        v25 = (unsigned __int64)v24;
        v24 = (_QWORD *)*v24;
        j_j___libc_free_0(v25);
      }
      memset(*(void **)(j + 8), 0, 8LL * *(_QWORD *)(j + 16));
      v26 = *(_QWORD *)(j + 8);
      *(_QWORD *)(j + 32) = 0;
      *(_QWORD *)(j + 24) = 0;
      if ( v26 != j + 56 )
        j_j___libc_free_0(v26);
    }
  }
  sub_C7D6A0(v4, v36, 8);
}
