// Function: sub_1DB4100
// Address: 0x1db4100
//
bool __fastcall sub_1DB4100(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 *v5; // r9
  __int64 *v6; // rax
  __int64 v7; // r10
  __int64 *v8; // rdi
  unsigned int v9; // ecx
  __int64 v10; // rdx
  __int64 v11; // rcx

  v2 = *(unsigned int *)(a1 + 8);
  v3 = *(unsigned int *)(a2 + 8);
  if ( !(_DWORD)v2 )
    return (_DWORD)v3 == 0;
  v5 = *(__int64 **)a2;
  v6 = *(__int64 **)a1;
  v7 = *(_QWORD *)a2 + 24 * v3;
  if ( *(_QWORD *)a2 == v7 )
    return 1;
  v8 = &v6[3 * v2];
  while ( 1 )
  {
    v9 = *(_DWORD *)((*v5 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*v5 >> 1) & 3;
    if ( v9 >= (*(_DWORD *)((*(v8 - 2) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*(v8 - 2) >> 1) & 3) )
      break;
    if ( (*(_DWORD *)((v6[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v6[1] >> 1) & 3) <= v9 )
    {
      do
      {
        v10 = v6[4];
        v6 += 3;
      }
      while ( v9 >= (*(_DWORD *)((v10 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v10 >> 1) & 3) );
    }
    if ( v8 == v6 || v9 < (*(_DWORD *)((*v6 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v6 >> 1) & 3) )
      break;
    while ( 1 )
    {
      v11 = v6[1];
      if ( (*(_DWORD *)((v11 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v11 >> 1) & 3) >= (*(_DWORD *)((v5[1] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                             | (unsigned int)(v5[1] >> 1)
                                                                                             & 3) )
        break;
      v6 += 3;
      if ( v6 == v8 || *v6 != v11 )
        return 0;
    }
    v5 += 3;
    if ( (__int64 *)v7 == v5 )
      return 1;
  }
  return 0;
}
