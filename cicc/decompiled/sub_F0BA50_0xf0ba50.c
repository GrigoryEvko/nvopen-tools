// Function: sub_F0BA50
// Address: 0xf0ba50
//
unsigned __int64 __fastcall sub_F0BA50(__int64 a1, __int64 a2)
{
  __int64 v4; // rdx
  __int64 v5; // rdi
  unsigned int v6; // esi
  __int64 *v7; // rax
  __int64 v8; // r9
  _QWORD *v9; // rdi
  __int64 v10; // rsi
  unsigned __int64 result; // rax
  int v12; // r8d
  const void *v13; // r9
  __int64 v14; // rdi
  int v15; // edx
  __int64 *v16; // rsi
  __int64 v17; // r8
  __int64 v18; // rax
  _QWORD *v19; // rdi
  int v20; // eax
  int v21; // r10d
  int v22; // esi
  int v23; // r9d
  __int64 v24[2]; // [rsp+8h] [rbp-18h] BYREF

  v4 = *(unsigned int *)(a1 + 2088);
  v24[0] = a2;
  v5 = *(_QWORD *)(a1 + 2072);
  if ( (_DWORD)v4 )
  {
    v6 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v5 + 16LL * v6);
    v8 = *v7;
    if ( a2 == *v7 )
    {
LABEL_3:
      if ( v7 != (__int64 *)(v5 + 16 * v4) )
      {
        *(_QWORD *)(*(_QWORD *)a1 + 8LL * *((unsigned int *)v7 + 2)) = 0;
        *v7 = -8192;
        --*(_DWORD *)(a1 + 2080);
        ++*(_DWORD *)(a1 + 2084);
      }
    }
    else
    {
      v20 = 1;
      while ( v8 != -4096 )
      {
        v21 = v20 + 1;
        v6 = (v4 - 1) & (v20 + v6);
        v7 = (__int64 *)(v5 + 16LL * v6);
        v8 = *v7;
        if ( a2 == *v7 )
          goto LABEL_3;
        v20 = v21;
      }
    }
  }
  if ( !*(_DWORD *)(a1 + 2112) )
  {
    v9 = *(_QWORD **)(a1 + 2128);
    v10 = (__int64)&v9[*(unsigned int *)(a1 + 2136)];
    result = (unsigned __int64)sub_F06A90(v9, v10, v24);
    if ( v10 == result )
      return result;
    v13 = (const void *)(result + 8);
    if ( v10 == result + 8 )
      goto LABEL_9;
LABEL_8:
    result = (unsigned __int64)memmove((void *)result, v13, v10 - (_QWORD)v13);
    v12 = *(_DWORD *)(a1 + 2136);
LABEL_9:
    *(_DWORD *)(a1 + 2136) = v12 - 1;
    return result;
  }
  result = *(unsigned int *)(a1 + 2120);
  v14 = *(_QWORD *)(a1 + 2104);
  if ( (_DWORD)result )
  {
    v15 = result - 1;
    result = ((_DWORD)result - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v16 = (__int64 *)(v14 + 8 * result);
    v17 = *v16;
    if ( a2 == *v16 )
    {
LABEL_13:
      *v16 = -8192;
      v18 = *(unsigned int *)(a1 + 2136);
      --*(_DWORD *)(a1 + 2112);
      v19 = *(_QWORD **)(a1 + 2128);
      ++*(_DWORD *)(a1 + 2116);
      v10 = (__int64)&v19[v18];
      result = (unsigned __int64)sub_F06A90(v19, v10, v24);
      v13 = (const void *)(result + 8);
      if ( result + 8 == v10 )
        goto LABEL_9;
      goto LABEL_8;
    }
    v22 = 1;
    while ( v17 != -4096 )
    {
      v23 = v22 + 1;
      result = v15 & (unsigned int)(v22 + result);
      v16 = (__int64 *)(v14 + 8LL * (unsigned int)result);
      v17 = *v16;
      if ( a2 == *v16 )
        goto LABEL_13;
      v22 = v23;
    }
  }
  return result;
}
