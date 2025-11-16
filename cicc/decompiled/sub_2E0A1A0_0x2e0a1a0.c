// Function: sub_2E0A1A0
// Address: 0x2e0a1a0
//
bool __fastcall sub_2E0A1A0(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // rcx
  __int64 *v5; // rax
  __int64 *v6; // rdi
  __int64 v7; // r9
  __int64 *v8; // rsi
  __int64 v9; // r10
  __int64 v10; // r8
  unsigned __int64 v11; // r10
  unsigned int v12; // r8d
  unsigned int v13; // ecx
  __int64 v14; // rdx
  __int64 v15; // rcx

  v2 = *(unsigned int *)(a1 + 8);
  v3 = *(unsigned int *)(a2 + 8);
  if ( !(_DWORD)v2 )
    return (_DWORD)v3 == 0;
  v5 = *(__int64 **)a1;
  v6 = *(__int64 **)a2;
  v7 = *(_QWORD *)a2 + 24 * v3;
  if ( v7 == *(_QWORD *)a2 )
    return 1;
  v8 = &v5[3 * v2];
  v9 = *(v8 - 2);
  v10 = v9 >> 1;
  v11 = v9 & 0xFFFFFFFFFFFFFFF8LL;
  v12 = v10 & 3;
  while ( 1 )
  {
    v13 = *(_DWORD *)((*v6 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*v6 >> 1) & 3;
    if ( v13 >= (v12 | *(_DWORD *)(v11 + 24)) )
      break;
    if ( (*(_DWORD *)((v5[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v5[1] >> 1) & 3) <= v13 )
    {
      do
      {
        v14 = v5[4];
        v5 += 3;
      }
      while ( v13 >= (*(_DWORD *)((v14 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v14 >> 1) & 3) );
    }
    if ( v8 == v5 || v13 < (*(_DWORD *)((*v5 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v5 >> 1) & 3) )
      break;
    while ( 1 )
    {
      v15 = v5[1];
      if ( (*(_DWORD *)((v15 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v15 >> 1) & 3) >= (*(_DWORD *)((v6[1] & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                                             | (unsigned int)(v6[1] >> 1)
                                                                                             & 3) )
        break;
      v5 += 3;
      if ( v5 == v8 || *v5 != v15 )
        return 0;
    }
    v6 += 3;
    if ( (__int64 *)v7 == v6 )
      return 1;
  }
  return 0;
}
