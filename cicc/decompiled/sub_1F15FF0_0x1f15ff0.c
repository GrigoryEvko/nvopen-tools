// Function: sub_1F15FF0
// Address: 0x1f15ff0
//
__int64 __fastcall sub_1F15FF0(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int v5; // r8d
  __int64 v6; // rdi
  int v7; // edx
  __int64 v8; // rcx
  __int64 v9; // rdi
  int i; // r10d
  __int64 *v11; // rdi
  unsigned int v12; // edx
  unsigned __int64 v13; // rdi
  __int64 v14; // rcx
  __int64 *v15; // rdx

  v5 = *(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (a2 >> 1) & 3;
  if ( (*(_DWORD *)((*(_QWORD *)(a1 + 96) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*(__int64 *)(a1 + 96) >> 1) & 3) > v5 )
  {
    v8 = 0;
  }
  else
  {
    v6 = a1 + 8;
    v7 = 0;
    do
      v8 = (unsigned int)++v7;
    while ( (*(_DWORD *)((*(_QWORD *)(v6 + 8 * v8 + 88) & 0xFFFFFFFFFFFFFFF8LL) + 24)
           | (unsigned int)(*(__int64 *)(v6 + 8 * v8 + 88) >> 1) & 3) <= v5 );
  }
  v9 = *(_QWORD *)(a1 + 8 * v8 + 8);
  for ( i = *(_DWORD *)(a1 + 184) - 1; i; --i )
  {
    while ( 1 )
    {
      v11 = (__int64 *)(v9 & 0xFFFFFFFFFFFFFFC0LL);
      if ( (*(_DWORD *)((v11[12] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v11[12] >> 1) & 3) <= v5 )
        break;
      v9 = *v11;
      if ( !--i )
        goto LABEL_9;
    }
    v12 = 0;
    do
      ++v12;
    while ( (*(_DWORD *)((v11[v12 + 12] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v11[v12 + 12] >> 1) & 3) <= v5 );
    v9 = v11[v12];
  }
LABEL_9:
  v13 = v9 & 0xFFFFFFFFFFFFFFC0LL;
  if ( (*(_DWORD *)((*(_QWORD *)(v13 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*(__int64 *)(v13 + 8) >> 1) & 3) > v5 )
  {
    v15 = (__int64 *)v13;
    v14 = 0;
  }
  else
  {
    LODWORD(v14) = 0;
    do
      v14 = (unsigned int)(v14 + 1);
    while ( (*(_DWORD *)((*(_QWORD *)(v13 + 16 * v14 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24)
           | (unsigned int)(*(__int64 *)(v13 + 16 * v14 + 8) >> 1) & 3) <= v5 );
    v15 = (__int64 *)(v13 + 16 * v14);
  }
  if ( (*(_DWORD *)((*v15 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v15 >> 1) & 3) <= v5 )
    return *(unsigned int *)(v13 + 4 * v14 + 144);
  return a3;
}
