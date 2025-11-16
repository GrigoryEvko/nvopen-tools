// Function: sub_2FB3BF0
// Address: 0x2fb3bf0
//
__int64 __fastcall sub_2FB3BF0(__int64 a1, __int64 a2, unsigned int a3)
{
  int v3; // ecx
  __int64 v6; // rsi
  unsigned int v7; // edx
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 *v10; // rcx
  __int64 result; // rax
  __int64 v12; // rcx

  v3 = *(_DWORD *)(a1 + 188);
  if ( !v3 )
    return a3;
  v6 = *(unsigned int *)(a1 + 184);
  v7 = *(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (a2 >> 1) & 3;
  v8 = *(_QWORD *)a1;
  if ( (_DWORD)v6 )
  {
    if ( (*(_DWORD *)((v8 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v8 >> 1) & 3) <= v7 )
    {
      v12 = *(_QWORD *)(a1 + 8LL * (unsigned int)(v3 - 1) + 96);
      if ( ((unsigned int)(v12 >> 1) & 3 | *(_DWORD *)((v12 & 0xFFFFFFFFFFFFFFF8LL) + 24)) > v7 )
        return sub_2FB3A50(a1, a2, a3);
    }
    return a3;
  }
  if ( (*(_DWORD *)((v8 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v8 >> 1) & 3) > v7 )
    return a3;
  v9 = *(_QWORD *)(a1 + 16LL * (unsigned int)(v3 - 1) + 8);
  if ( (*(_DWORD *)((v9 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v9 >> 1) & 3) <= v7 )
    return a3;
  v10 = (__int64 *)a1;
  if ( (*(_DWORD *)((*(_QWORD *)(a1 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*(__int64 *)(a1 + 8) >> 1) & 3) <= v7 )
  {
    do
      v6 = (unsigned int)(v6 + 1);
    while ( (*(_DWORD *)((*(_QWORD *)(a1 + 16 * v6 + 8) & 0xFFFFFFFFFFFFFFF8LL) + 24)
           | (unsigned int)(*(__int64 *)(a1 + 16 * v6 + 8) >> 1) & 3) <= v7 );
    v10 = (__int64 *)(a1 + 16 * v6);
  }
  result = a3;
  if ( (*(_DWORD *)((*v10 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((*v10 >> 1) & 3)) <= v7 )
    return *(unsigned int *)(a1 + 4 * v6 + 144);
  return result;
}
