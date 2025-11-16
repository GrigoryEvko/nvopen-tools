// Function: sub_1ECDA00
// Address: 0x1ecda00
//
__int64 __fastcall sub_1ECDA00(_QWORD *a1, unsigned int a2, unsigned int a3)
{
  __int64 v5; // rbx
  __int64 *v6; // r14
  __int64 v7; // r9
  __int64 v8; // rsi
  __int64 v9; // rcx
  __int64 v10; // rdx
  int v11; // eax
  __int64 v12; // r8
  __int64 v13; // rdx
  unsigned int v14; // eax
  __int64 v15; // rdx
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rbx
  __int64 v19; // rsi
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 result; // rax
  __int64 v23; // rsi
  __int64 v24; // rcx
  __int64 v25; // rax
  _DWORD *v26; // rdi
  int v27[13]; // [rsp+Ch] [rbp-34h] BYREF

  v5 = 48LL * a2;
  v6 = (__int64 *)a1[19];
  if ( v6 )
  {
    v7 = 88LL * a3;
    v8 = v7 + *(_QWORD *)(*v6 + 160);
    v9 = v5 + *(_QWORD *)(*v6 + 208);
    v10 = *(_QWORD *)v9;
    v11 = *(_DWORD *)(v8 + 24);
    if ( a3 == *(_DWORD *)(v9 + 24) )
    {
      *(_DWORD *)(v8 + 24) = v11 - *(_DWORD *)(v10 + 16);
      v12 = *(_QWORD *)(v10 + 32);
    }
    else
    {
      *(_DWORD *)(v8 + 24) = v11 - *(_DWORD *)(v10 + 20);
      v12 = *(_QWORD *)(v10 + 24);
    }
    v13 = *(unsigned int *)(v8 + 20);
    if ( (_DWORD)v13 )
    {
      v14 = 0;
      do
      {
        v15 = v14++;
        *(_DWORD *)(*(_QWORD *)(v8 + 32) + 4 * v15) -= *(unsigned __int8 *)(v12 + v15);
        v13 = *(unsigned int *)(v8 + 20);
      }
      while ( (unsigned int)v13 > v14 );
    }
    if ( *(_QWORD *)(*(_QWORD *)(*v6 + 160) + v7 + 72) - *(_QWORD *)(*(_QWORD *)(*v6 + 160) + v7 + 64) == 12 )
    {
      v27[0] = a3;
      sub_1ECBF60(v6, a3);
      sub_1DF87A0(v6 + 1, (unsigned int *)v27);
      *(_DWORD *)(*(_QWORD *)(*v6 + 160) + 88LL * (unsigned int)v27[0] + 16) = 3;
    }
    else if ( *(_DWORD *)(v8 + 16) == 1 )
    {
      if ( (unsigned int)v13 > *(_DWORD *)(v8 + 24)
        || (v27[0] = 0, v26 = *(_DWORD **)(v8 + 32), &v26[v13] != sub_1ECB090(v26, (__int64)&v26[v13], v27)) )
      {
        v27[0] = a3;
        sub_1ECBF60(v6, a3);
        sub_1DF87A0(v6 + 7, (unsigned int *)v27);
        *(_DWORD *)(*(_QWORD *)(*v6 + 160) + 88LL * (unsigned int)v27[0] + 16) = 2;
      }
    }
  }
  v16 = a1[26];
  v17 = a1[20];
  v18 = v16 + v5;
  if ( a3 == *(_DWORD *)(v18 + 20) )
  {
    v23 = *(_QWORD *)(v18 + 32);
    v24 = v17 + 88LL * a3;
    v25 = 48LL * *(unsigned int *)(*(_QWORD *)(v24 + 72) - 4LL) + v16;
    if ( a3 == *(_DWORD *)(v25 + 20) )
      *(_QWORD *)(v25 + 32) = v23;
    else
      *(_QWORD *)(v25 + 40) = v23;
    result = *(_QWORD *)(v24 + 64);
    *(_DWORD *)(result + 4 * v23) = *(_DWORD *)(*(_QWORD *)(v24 + 72) - 4LL);
    *(_QWORD *)(v24 + 72) -= 4LL;
    *(_QWORD *)(v18 + 32) = -1;
  }
  else
  {
    v19 = *(_QWORD *)(v18 + 40);
    v20 = v17 + 88LL * *(unsigned int *)(v18 + 24);
    v21 = 48LL * *(unsigned int *)(*(_QWORD *)(v20 + 72) - 4LL) + v16;
    if ( *(_DWORD *)(v18 + 24) == *(_DWORD *)(v21 + 20) )
      *(_QWORD *)(v21 + 32) = v19;
    else
      *(_QWORD *)(v21 + 40) = v19;
    result = *(_QWORD *)(v20 + 64);
    *(_DWORD *)(result + 4 * v19) = *(_DWORD *)(*(_QWORD *)(v20 + 72) - 4LL);
    *(_QWORD *)(v20 + 72) -= 4LL;
    *(_QWORD *)(v18 + 40) = -1;
  }
  return result;
}
