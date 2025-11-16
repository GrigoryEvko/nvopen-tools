// Function: sub_2D2BB00
// Address: 0x2d2bb00
//
unsigned __int64 __fastcall sub_2D2BB00(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdi
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // rsi
  int i; // r14d
  unsigned __int64 v13; // rbx
  unsigned int v14; // eax
  __int64 v15; // rdx
  __int64 *v16; // r15
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  unsigned __int64 v19; // rbx
  __int64 v20; // rax
  __int64 v21; // rdx
  unsigned __int64 result; // rax
  __int64 v23; // r12
  unsigned __int64 v24; // rdx
  unsigned __int64 *v25; // rcx
  __int64 v26; // [rsp+0h] [rbp-40h]

  v7 = a1 + 8;
  LODWORD(v9) = *(_DWORD *)(v7 + 8);
  v10 = *(_QWORD *)v7 + 16LL * (unsigned int)(v9 - 1);
  v11 = *(_QWORD *)(*(_QWORD *)v10 + 8LL * *(unsigned int *)(v10 + 12));
  for ( i = *(_DWORD *)(*(_QWORD *)(v7 - 8) + 192LL) - v9; i; --i )
  {
    v13 = v11 & 0xFFFFFFFFFFFFFFC0LL;
    if ( a2 < *(_DWORD *)((v11 & 0xFFFFFFFFFFFFFFC0LL) + 0x80) )
    {
      v16 = (__int64 *)(v11 & 0xFFFFFFFFFFFFFFC0LL);
      v15 = 0;
    }
    else
    {
      v14 = 0;
      do
        v15 = ++v14;
      while ( a2 >= *(_DWORD *)((v11 & 0xFFFFFFFFFFFFFFC0LL) + 4LL * v14 + 0x80) );
      v16 = (__int64 *)(v13 + 8LL * v14);
    }
    v9 = (unsigned int)v9;
    v17 = (v15 << 32) | ((v11 & 0x3F) + 1);
    v18 = (unsigned int)v9 + 1LL;
    if ( v18 > *(unsigned int *)(a1 + 20) )
    {
      v26 = v17;
      sub_C8D5F0(v7, (const void *)(a1 + 24), v18, 0x10u, a5, a6);
      v9 = *(unsigned int *)(a1 + 16);
      v17 = v26;
    }
    v9 = *(_QWORD *)(a1 + 8) + 16 * v9;
    *(_QWORD *)v9 = v13;
    *(_QWORD *)(v9 + 8) = v17;
    LODWORD(v9) = *(_DWORD *)(a1 + 16) + 1;
    *(_DWORD *)(a1 + 16) = v9;
    v11 = *v16;
  }
  v19 = v11 & 0xFFFFFFFFFFFFFFC0LL;
  if ( a2 < *(_DWORD *)((v11 & 0xFFFFFFFFFFFFFFC0LL) + 4) )
  {
    v21 = 0;
  }
  else
  {
    v20 = 1;
    do
      v21 = (unsigned int)v20++;
    while ( a2 >= *(_DWORD *)(v19 + 8 * v20 - 4) );
  }
  v9 = (unsigned int)v9;
  result = *(unsigned int *)(a1 + 20);
  v23 = (v21 << 32) | ((v11 & 0x3F) + 1);
  v24 = (unsigned int)v9 + 1LL;
  if ( v24 > result )
  {
    result = sub_C8D5F0(v7, (const void *)(a1 + 24), v24, 0x10u, a5, a6);
    v9 = *(unsigned int *)(a1 + 16);
  }
  v25 = (unsigned __int64 *)(*(_QWORD *)(a1 + 8) + 16 * v9);
  *v25 = v19;
  v25[1] = v23;
  ++*(_DWORD *)(a1 + 16);
  return result;
}
