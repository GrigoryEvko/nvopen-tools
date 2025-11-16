// Function: sub_2FB3A50
// Address: 0x2fb3a50
//
__int64 __fastcall sub_2FB3A50(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v6; // rax
  unsigned int v7; // edi
  unsigned int v8; // edx
  __int64 v9; // rcx
  __int64 v10; // rsi
  __int64 v11; // rdi
  int i; // r8d
  __int64 *v13; // rdi
  unsigned int v14; // r12d
  unsigned int v15; // edx
  unsigned __int64 v16; // rdi
  unsigned int v17; // r8d
  __int64 v18; // rcx
  __int64 *v19; // rdx

  v6 = *(_QWORD *)(a1 + 96);
  v7 = *(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (a2 >> 1) & 3;
  if ( (*(_DWORD *)((v6 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v6 >> 1) & 3) > v7 )
  {
    v9 = 0;
  }
  else
  {
    v8 = 0;
    do
    {
      v9 = ++v8;
      v10 = *(_QWORD *)(a1 + 8 + 8LL * v8 + 88);
    }
    while ( (*(_DWORD *)((v10 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v10 >> 1) & 3) <= v7 );
  }
  v11 = *(_QWORD *)(a1 + 8 * v9 + 8);
  for ( i = *(_DWORD *)(a1 + 184) - 1; i; --i )
  {
    while ( 1 )
    {
      v13 = (__int64 *)(v11 & 0xFFFFFFFFFFFFFFC0LL);
      v14 = (a2 >> 1) & 3 | *(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24);
      if ( (*(_DWORD *)((v13[12] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v13[12] >> 1) & 3) <= v14 )
        break;
      v11 = *v13;
      if ( !--i )
        goto LABEL_9;
    }
    v15 = 0;
    do
      ++v15;
    while ( (*(_DWORD *)((v13[v15 + 12] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v13[v15 + 12] >> 1) & 3) <= v14 );
    v11 = v13[v15];
  }
LABEL_9:
  v16 = v11 & 0xFFFFFFFFFFFFFFC0LL;
  v17 = *(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (a2 >> 1) & 3;
  if ( v17 < (*(_DWORD *)((*(_QWORD *)(v16 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24)
            | (unsigned int)(*(__int64 *)(v16 + 8) >> 1) & 3) )
  {
    v19 = (__int64 *)v16;
    v18 = 0;
  }
  else
  {
    LODWORD(v18) = 0;
    do
      v18 = (unsigned int)(v18 + 1);
    while ( v17 >= (*(_DWORD *)((*(_QWORD *)(v16 + 16 * v18 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24)
                  | (unsigned int)(*(__int64 *)(v16 + 16 * v18 + 8) >> 1) & 3) );
    v19 = (__int64 *)(v16 + 16 * v18);
  }
  if ( v17 >= (*(_DWORD *)((*v19 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v19 >> 1) & 3) )
    return *(unsigned int *)(v16 + 4 * v18 + 144);
  return a3;
}
