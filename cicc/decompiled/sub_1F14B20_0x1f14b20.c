// Function: sub_1F14B20
// Address: 0x1f14b20
//
__int64 __fastcall sub_1F14B20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, unsigned int a6)
{
  __int64 v7; // rax
  unsigned __int64 v8; // rcx
  __int64 *v9; // rbx
  __int64 *v10; // r14
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned __int64 i; // rdx
  __int64 v15; // rbx
  char v16; // al
  __int64 *v17; // r13
  signed __int64 v18; // rsi
  __int64 *v19; // r15
  __int64 *v20; // rbx
  __int64 *v21; // r14
  char v22; // al
  __int64 v23; // rsi
  __int64 v24; // rdi
  __int64 v25; // rsi
  __int64 *v26; // rax
  __int64 v27; // r10
  unsigned __int64 v28; // r13
  int v29; // eax
  unsigned __int64 *v30; // rax
  __int64 result; // rax
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // rsi
  __int64 v36; // rdx
  int v37; // ecx
  int v38; // r8d
  int v39; // r9d
  int v40; // r11d
  __int64 v41; // rax

  v7 = *(_QWORD *)(a1 + 40);
  v8 = *(unsigned int *)(a1 + 208);
  v9 = *(__int64 **)(v7 + 64);
  v10 = &v9[*(unsigned int *)(v7 + 72)];
  if ( v9 != v10 )
  {
    do
    {
      while ( 1 )
      {
        v11 = *v9;
        v12 = *(_QWORD *)(*v9 + 8);
        if ( (v12 & 6) != 0 && (v12 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
          break;
        if ( v10 == ++v9 )
          goto LABEL_9;
      }
      if ( *(_DWORD *)(a1 + 212) <= (unsigned int)v8 )
      {
        sub_16CD150(a1 + 200, (const void *)(a1 + 216), 0, 8, a5, a6);
        v8 = *(unsigned int *)(a1 + 208);
      }
      ++v9;
      *(_QWORD *)(*(_QWORD *)(a1 + 200) + 8 * v8) = *(_QWORD *)(v11 + 8);
      v8 = (unsigned int)(*(_DWORD *)(a1 + 208) + 1);
      *(_DWORD *)(a1 + 208) = v8;
    }
    while ( v10 != v9 );
LABEL_9:
    v7 = *(_QWORD *)(a1 + 40);
  }
  v13 = *(unsigned int *)(v7 + 112);
  i = *(_QWORD *)(*(_QWORD *)a1 + 40LL);
  if ( (int)v13 < 0 )
  {
    v15 = *(_QWORD *)(*(_QWORD *)(i + 24) + 16 * (v13 & 0x7FFFFFFF) + 8);
  }
  else
  {
    i = *(_QWORD *)(i + 272);
    v15 = *(_QWORD *)(i + 8 * v13);
  }
  while ( v15 )
  {
    if ( (*(_BYTE *)(v15 + 3) & 0x10) == 0 )
    {
      v16 = *(_BYTE *)(v15 + 4);
      if ( (v16 & 8) == 0 )
      {
        if ( (v16 & 1) == 0 )
          goto LABEL_28;
        while ( 1 )
        {
LABEL_24:
          v15 = *(_QWORD *)(v15 + 32);
          if ( !v15 )
            goto LABEL_16;
          if ( (*(_BYTE *)(v15 + 3) & 0x10) == 0 )
          {
            v22 = *(_BYTE *)(v15 + 4);
            if ( (v22 & 8) == 0 && (v22 & 1) == 0 )
              break;
          }
        }
LABEL_28:
        v23 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 272LL);
        for ( i = *(_QWORD *)(v15 + 16); (*(_BYTE *)(i + 46) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
          ;
        v24 = *(_QWORD *)(v23 + 368);
        v25 = *(unsigned int *)(v23 + 384);
        if ( (_DWORD)v25 )
        {
          a5 = v25 - 1;
          a6 = (v25 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
          v26 = (__int64 *)(v24 + 16LL * a6);
          v27 = *v26;
          if ( *v26 == i )
          {
LABEL_32:
            v28 = v26[1] & 0xFFFFFFFFFFFFFFF8LL | 4;
            if ( *(_DWORD *)(a1 + 212) <= (unsigned int)v8 )
            {
              sub_16CD150(a1 + 200, (const void *)(a1 + 216), 0, 8, a5, a6);
              LODWORD(v8) = *(_DWORD *)(a1 + 208);
            }
            *(_QWORD *)(*(_QWORD *)(a1 + 200) + 8LL * (unsigned int)v8) = v28;
            v8 = (unsigned int)(*(_DWORD *)(a1 + 208) + 1);
            *(_DWORD *)(a1 + 208) = v8;
            goto LABEL_24;
          }
          v29 = 1;
          while ( v27 != -8 )
          {
            v40 = v29 + 1;
            v41 = a5 & (a6 + v29);
            a6 = v41;
            v26 = (__int64 *)(v24 + 16 * v41);
            v27 = *v26;
            if ( *v26 == i )
              goto LABEL_32;
            v29 = v40;
          }
        }
        v26 = (__int64 *)(v24 + 16 * v25);
        goto LABEL_32;
      }
    }
    v15 = *(_QWORD *)(v15 + 32);
  }
LABEL_16:
  v17 = *(__int64 **)(a1 + 200);
  v18 = 8 * v8;
  if ( v8 > 1 )
  {
    qsort(*(void **)(a1 + 200), v18 >> 3, 8u, (__compar_fn_t)sub_1F138C0);
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
      if ( sub_1F134F0(*v21, *v20) )
      {
        if ( v19 == v21 )
          break;
        v30 = (unsigned __int64 *)(v21 + 2);
        if ( v19 != v21 + 2 )
        {
          do
          {
            v8 = *v30;
            v18 = *v30 & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*v21 & 0xFFFFFFFFFFFFFFF8LL) != v18 )
            {
              v21[1] = v8;
              ++v21;
            }
            ++v30;
          }
          while ( v19 != (__int64 *)v30 );
          v17 = *(__int64 **)(a1 + 200);
          i = (char *)&v17[*(unsigned int *)(a1 + 208)] - (char *)v19;
          v20 = (__int64 *)((char *)v21 + i + 8);
          if ( v19 != &v17[*(unsigned int *)(a1 + 208)] )
          {
            v18 = (signed __int64)v19;
            memmove(v21 + 1, v19, i);
            v17 = *(__int64 **)(a1 + 200);
          }
        }
        goto LABEL_45;
      }
    }
  }
  v20 = v19;
LABEL_45:
  *(_DWORD *)(a1 + 208) = v20 - v17;
  result = sub_1F14370((_QWORD *)a1, v18, i, v8, a5, a6);
  if ( !(_BYTE)result )
  {
    *(_BYTE *)(a1 + 652) = 1;
    v35 = *(_QWORD *)(a1 + 40);
    sub_1DC0580(*(_QWORD **)(a1 + 16), v35, 0, v32, v33, v34);
    *(_DWORD *)(a1 + 288) = 0;
    *(_DWORD *)(a1 + 640) = 0;
    return sub_1F14370((_QWORD *)a1, v35, v36, v37, v38, v39);
  }
  return result;
}
