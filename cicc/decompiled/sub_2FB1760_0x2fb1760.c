// Function: sub_2FB1760
// Address: 0x2fb1760
//
char __fastcall sub_2FB1760(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  __int64 v8; // rbx
  __int64 v9; // r14
  __int64 v10; // r13
  __int64 v11; // rax
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // rbx
  char v15; // al
  unsigned __int64 v16; // rdi
  __int64 *v17; // r14
  signed __int64 v18; // rsi
  __int64 *v19; // r15
  __int64 *v20; // rbx
  __int64 *v21; // r13
  char v22; // al
  unsigned __int64 i; // rax
  __int64 j; // rsi
  int v25; // edx
  __int64 v26; // rsi
  __int64 *v27; // rdx
  __int64 v28; // r10
  __int64 v29; // r14
  unsigned __int64 v30; // r14
  unsigned __int64 *v31; // rax
  int v33; // edx
  int v34; // r11d

  v7 = *(_QWORD *)(a1 + 40);
  v8 = *(_QWORD *)(v7 + 64);
  v9 = v8 + 8LL * *(unsigned int *)(v7 + 72);
  if ( v8 != v9 )
  {
    do
    {
      while ( 1 )
      {
        v10 = *(_QWORD *)(*(_QWORD *)v8 + 8LL);
        if ( (v10 & 6) != 0 && (v10 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          break;
        v8 += 8;
        if ( v9 == v8 )
          goto LABEL_9;
      }
      v11 = *(unsigned int *)(a1 + 208);
      a4 = *(unsigned int *)(a1 + 212);
      if ( v11 + 1 > a4 )
      {
        sub_C8D5F0(a1 + 200, (const void *)(a1 + 216), v11 + 1, 8u, a5, a6);
        v11 = *(unsigned int *)(a1 + 208);
      }
      v8 += 8;
      *(_QWORD *)(*(_QWORD *)(a1 + 200) + 8 * v11) = v10;
      ++*(_DWORD *)(a1 + 208);
    }
    while ( v9 != v8 );
LABEL_9:
    v7 = *(_QWORD *)(a1 + 40);
  }
  v12 = *(unsigned int *)(v7 + 112);
  v13 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  if ( (int)v12 < 0 )
  {
    v14 = *(_QWORD *)(*(_QWORD *)(v13 + 56) + 16 * (v12 & 0x7FFFFFFF) + 8);
  }
  else
  {
    v13 = *(_QWORD *)(v13 + 304);
    v14 = *(_QWORD *)(v13 + 8 * v12);
  }
  while ( v14 )
  {
    if ( (*(_BYTE *)(v14 + 3) & 0x10) == 0 )
    {
      v15 = *(_BYTE *)(v14 + 4);
      if ( (v15 & 8) == 0 )
      {
        v16 = *(unsigned int *)(a1 + 208);
        if ( (v15 & 1) == 0 )
          goto LABEL_29;
        while ( 1 )
        {
LABEL_25:
          v14 = *(_QWORD *)(v14 + 32);
          if ( !v14 )
            goto LABEL_17;
          if ( (*(_BYTE *)(v14 + 3) & 0x10) == 0 )
          {
            v22 = *(_BYTE *)(v14 + 4);
            if ( (v22 & 8) == 0 && (v22 & 1) == 0 )
              break;
          }
        }
LABEL_29:
        a4 = *(_QWORD *)(v14 + 16);
        a6 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 32LL);
        for ( i = a4; (*(_BYTE *)(i + 44) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
          ;
        for ( ; (*(_BYTE *)(a4 + 44) & 8) != 0; a4 = *(_QWORD *)(a4 + 8) )
          ;
        for ( j = *(_QWORD *)(a4 + 8); j != i; i = *(_QWORD *)(i + 8) )
        {
          v25 = *(unsigned __int16 *)(i + 68);
          a4 = (unsigned int)(v25 - 14);
          if ( (unsigned __int16)(v25 - 14) > 4u && (_WORD)v25 != 24 )
            break;
        }
        v26 = *(unsigned int *)(a6 + 144);
        a5 = *(_QWORD *)(a6 + 128);
        if ( (_DWORD)v26 )
        {
          a6 = (unsigned int)(v26 - 1);
          a4 = (unsigned int)a6 & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
          v27 = (__int64 *)(a5 + 16 * a4);
          v28 = *v27;
          if ( i == *v27 )
          {
LABEL_39:
            v29 = v27[1];
            v16 = (unsigned int)v16;
            v13 = (unsigned int)v16 + 1LL;
            v30 = v29 & 0xFFFFFFFFFFFFFFF8LL | 4;
            if ( v13 > *(unsigned int *)(a1 + 212) )
            {
              sub_C8D5F0(a1 + 200, (const void *)(a1 + 216), v13, 8u, a5, a6);
              v16 = *(unsigned int *)(a1 + 208);
            }
            *(_QWORD *)(*(_QWORD *)(a1 + 200) + 8 * v16) = v30;
            v16 = (unsigned int)(*(_DWORD *)(a1 + 208) + 1);
            *(_DWORD *)(a1 + 208) = v16;
            goto LABEL_25;
          }
          v33 = 1;
          while ( v28 != -4096 )
          {
            v34 = v33 + 1;
            a4 = (unsigned int)a6 & (v33 + (_DWORD)a4);
            v27 = (__int64 *)(a5 + 16LL * (unsigned int)a4);
            v28 = *v27;
            if ( i == *v27 )
              goto LABEL_39;
            v33 = v34;
          }
        }
        v27 = (__int64 *)(a5 + 16 * v26);
        goto LABEL_39;
      }
    }
    v14 = *(_QWORD *)(v14 + 32);
  }
  v16 = *(unsigned int *)(a1 + 208);
LABEL_17:
  v17 = *(__int64 **)(a1 + 200);
  v18 = 8 * v16;
  if ( v16 > 1 )
  {
    qsort(*(void **)(a1 + 200), v18 >> 3, 8u, (__compar_fn_t)sub_2FB04A0);
    v17 = *(__int64 **)(a1 + 200);
    v18 = 8LL * *(unsigned int *)(a1 + 208);
  }
  v19 = (__int64 *)((char *)v17 + v18);
  if ( (__int64 *)((char *)v17 + v18) != v17 )
  {
    v20 = v17;
    while ( 1 )
    {
      v21 = v20++;
      if ( v19 == v20 )
        break;
      v18 = *v20;
      if ( sub_2FB02A0(*v21, *v20) )
      {
        if ( v19 == v21 )
          break;
        v31 = (unsigned __int64 *)(v21 + 2);
        if ( v21 + 2 != v19 )
        {
          do
          {
            a4 = *v31;
            v18 = *v31 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*v21 & 0xFFFFFFFFFFFFFFF8LL) != v18 )
            {
              v21[1] = a4;
              ++v21;
            }
            ++v31;
          }
          while ( v19 != (__int64 *)v31 );
          v17 = *(__int64 **)(a1 + 200);
          v13 = (char *)&v17[*(unsigned int *)(a1 + 208)] - (char *)v19;
          v20 = (__int64 *)((char *)v21 + v13 + 8);
          if ( v19 != &v17[*(unsigned int *)(a1 + 208)] )
          {
            v18 = (signed __int64)v19;
            memmove(v21 + 1, v19, v13);
            v17 = *(__int64 **)(a1 + 200);
          }
        }
        goto LABEL_49;
      }
    }
  }
  v20 = v19;
LABEL_49:
  *(_DWORD *)(a1 + 208) = v20 - v17;
  return sub_2FB0FD0(a1, v18, v13, a4, a5, a6);
}
