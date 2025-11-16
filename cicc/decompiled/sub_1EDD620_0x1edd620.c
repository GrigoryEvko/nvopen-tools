// Function: sub_1EDD620
// Address: 0x1edd620
//
__int64 __fastcall sub_1EDD620(__int64 *a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // rbx
  int v5; // r14d
  __int64 v6; // rsi
  __int64 v7; // rax
  unsigned __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r15
  __int64 v12; // rax
  __int64 *v13; // rax
  __int64 v14; // r14
  __int64 v15; // rbx
  __int64 result; // rax
  __int64 i; // r13
  __int128 v18; // rax
  __int64 *v19; // rdx

  v3 = *a1;
  v4 = *(_QWORD *)(*a1 + 16);
  if ( *(_DWORD *)(a2 + 8) )
  {
    v6 = *(_QWORD *)(*a1 + 16);
    v11 = 0;
    v19 = (__int64 *)sub_1DB3C70((__int64 *)a2, v6);
    v8 = 3LL * *(unsigned int *)(a2 + 8);
    if ( v19 != (__int64 *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8)) )
    {
      v8 = v4 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_DWORD *)((*v19 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v19 >> 1) & 3) <= (*(_DWORD *)((v4 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                               | (unsigned int)(v4 >> 1)
                                                                                               & 3) )
        v11 = v19[2];
    }
  }
  else
  {
    v5 = *(_DWORD *)(a2 + 72);
    v6 = 16;
    v7 = sub_145CBF0(*(__int64 **)v3, 16, 16);
    *(_DWORD *)v7 = v5;
    v11 = v7;
    *(_QWORD *)(v7 + 8) = v4;
    v12 = *(unsigned int *)(a2 + 72);
    if ( (unsigned int)v12 >= *(_DWORD *)(a2 + 76) )
    {
      v6 = a2 + 80;
      sub_16CD150(a2 + 64, (const void *)(a2 + 80), 0, 8, v9, v10);
      v12 = *(unsigned int *)(a2 + 72);
    }
    *(_QWORD *)(*(_QWORD *)(a2 + 64) + 8 * v12) = v11;
    ++*(_DWORD *)(a2 + 72);
  }
  v13 = *(__int64 **)(v3 + 8);
  v14 = *(_QWORD *)(v3 + 24);
  v15 = *v13;
  result = 3LL * *((unsigned int *)v13 + 2);
  for ( i = v15 + 8 * result; i != v15; result = sub_1DB8610(a2, v6, *((__int64 *)&v18 + 1), v8, v9, v10, v18, v11) )
  {
    while ( v14 != *(_QWORD *)(v15 + 16) )
    {
      v15 += 24;
      if ( i == v15 )
        return result;
    }
    v18 = *(_OWORD *)v15;
    v15 += 24;
  }
  return result;
}
