// Function: sub_1632B40
// Address: 0x1632b40
//
unsigned __int64 __fastcall sub_1632B40(__int64 a1, __int64 a2)
{
  __int64 v4; // rdi
  size_t v5; // rdx
  size_t v6; // r13
  int v7; // r14d
  unsigned int v8; // eax
  const void *v9; // rsi
  unsigned __int64 v10; // rdx
  unsigned __int64 v11; // rax
  int v12; // r13d
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rdx
  unsigned __int64 v16; // rdx
  unsigned __int64 v17; // rax
  int v18; // r13d
  __int64 v19; // r14
  __int64 v20; // rax
  __int64 v21; // rdx
  unsigned __int64 result; // rax
  unsigned __int64 v23; // rdx
  int v24; // r13d
  __int64 v25; // r14
  __int64 v26; // rax
  __int64 v27; // rdx
  const void *v28; // rsi

  sub_15A9210(a1 + 280);
  sub_2240AE0(a1 + 472, a2 + 192);
  v4 = a1 + 304;
  *(_BYTE *)(a1 + 280) = *(_BYTE *)a2;
  *(_DWORD *)(a1 + 284) = *(_DWORD *)(a2 + 4);
  *(_DWORD *)(a1 + 288) = *(_DWORD *)(a2 + 8);
  *(_DWORD *)(a1 + 292) = *(_DWORD *)(a2 + 12);
  *(_DWORD *)(a1 + 296) = *(_DWORD *)(a2 + 16);
  if ( a1 + 304 != a2 + 24 )
  {
    v5 = *(unsigned int *)(a2 + 32);
    v6 = *(unsigned int *)(a1 + 312);
    v7 = *(_DWORD *)(a2 + 32);
    if ( v5 <= v6 )
    {
      if ( *(_DWORD *)(a2 + 32) )
        memmove(*(void **)(a1 + 304), *(const void **)(a2 + 24), v5);
    }
    else
    {
      if ( v5 > *(unsigned int *)(a1 + 316) )
      {
        v6 = 0;
        *(_DWORD *)(a1 + 312) = 0;
        sub_16CD150(v4, a1 + 320, v5, 1);
        v5 = *(unsigned int *)(a2 + 32);
        v8 = *(_DWORD *)(a2 + 32);
      }
      else
      {
        v8 = *(_DWORD *)(a2 + 32);
        if ( *(_DWORD *)(a1 + 312) )
        {
          memmove(*(void **)(a1 + 304), *(const void **)(a2 + 24), *(unsigned int *)(a1 + 312));
          v5 = *(unsigned int *)(a2 + 32);
          v8 = *(_DWORD *)(a2 + 32);
        }
      }
      v9 = (const void *)(v6 + *(_QWORD *)(a2 + 24));
      if ( v9 != (const void *)(*(_QWORD *)(a2 + 24) + v5) )
        memcpy((void *)(v6 + *(_QWORD *)(a1 + 304)), v9, v8 - v6);
    }
    *(_DWORD *)(a1 + 312) = v7;
  }
  if ( a1 + 328 != a2 + 48 )
  {
    v10 = *(unsigned int *)(a2 + 56);
    v11 = *(unsigned int *)(a1 + 336);
    v12 = *(_DWORD *)(a2 + 56);
    if ( v10 <= v11 )
    {
      if ( *(_DWORD *)(a2 + 56) )
        memmove(*(void **)(a1 + 328), *(const void **)(a2 + 48), 8 * v10);
    }
    else
    {
      if ( v10 > *(unsigned int *)(a1 + 340) )
      {
        v13 = 0;
        *(_DWORD *)(a1 + 336) = 0;
        sub_16CD150(a1 + 328, a1 + 344, v10, 8);
        v10 = *(unsigned int *)(a2 + 56);
      }
      else
      {
        v13 = 8 * v11;
        if ( *(_DWORD *)(a1 + 336) )
        {
          memmove(*(void **)(a1 + 328), *(const void **)(a2 + 48), 8 * v11);
          v10 = *(unsigned int *)(a2 + 56);
        }
      }
      v14 = *(_QWORD *)(a2 + 48);
      v15 = 8 * v10;
      if ( v14 + v13 != v15 + v14 )
        memcpy((void *)(v13 + *(_QWORD *)(a1 + 328)), (const void *)(v14 + v13), v15 - v13);
    }
    *(_DWORD *)(a1 + 336) = v12;
  }
  if ( a1 + 504 != a2 + 224 )
  {
    v16 = *(unsigned int *)(a2 + 232);
    v17 = *(unsigned int *)(a1 + 512);
    v18 = *(_DWORD *)(a2 + 232);
    if ( v16 <= v17 )
    {
      if ( *(_DWORD *)(a2 + 232) )
        memmove(*(void **)(a1 + 504), *(const void **)(a2 + 224), 20 * v16);
    }
    else
    {
      if ( v16 > *(unsigned int *)(a1 + 516) )
      {
        v19 = 0;
        *(_DWORD *)(a1 + 512) = 0;
        sub_16CD150(a1 + 504, a1 + 520, v16, 20);
        v16 = *(unsigned int *)(a2 + 232);
      }
      else
      {
        v19 = 20 * v17;
        if ( *(_DWORD *)(a1 + 512) )
        {
          memmove(*(void **)(a1 + 504), *(const void **)(a2 + 224), 20 * v17);
          v16 = *(unsigned int *)(a2 + 232);
        }
      }
      v20 = *(_QWORD *)(a2 + 224);
      v21 = 20 * v16;
      if ( v20 + v19 != v21 + v20 )
        memcpy((void *)(v19 + *(_QWORD *)(a1 + 504)), (const void *)(v20 + v19), v21 - v19);
    }
    *(_DWORD *)(a1 + 512) = v18;
  }
  result = a2 + 408;
  if ( a1 + 688 != a2 + 408 )
  {
    v23 = *(unsigned int *)(a2 + 416);
    result = *(unsigned int *)(a1 + 696);
    v24 = *(_DWORD *)(a2 + 416);
    if ( v23 <= result )
    {
      if ( *(_DWORD *)(a2 + 416) )
        result = (unsigned __int64)memmove(*(void **)(a1 + 688), *(const void **)(a2 + 408), 4 * v23);
    }
    else
    {
      if ( v23 > *(unsigned int *)(a1 + 700) )
      {
        v25 = 0;
        *(_DWORD *)(a1 + 696) = 0;
        sub_16CD150(a1 + 688, a1 + 704, v23, 4);
        v23 = *(unsigned int *)(a2 + 416);
      }
      else
      {
        v25 = 4 * result;
        if ( *(_DWORD *)(a1 + 696) )
        {
          memmove(*(void **)(a1 + 688), *(const void **)(a2 + 408), 4 * result);
          v23 = *(unsigned int *)(a2 + 416);
        }
      }
      v26 = *(_QWORD *)(a2 + 408);
      v27 = 4 * v23;
      v28 = (const void *)(v26 + v25);
      result = v27 + v26;
      if ( v28 != (const void *)result )
        result = (unsigned __int64)memcpy((void *)(v25 + *(_QWORD *)(a1 + 688)), v28, v27 - v25);
    }
    *(_DWORD *)(a1 + 696) = v24;
  }
  return result;
}
