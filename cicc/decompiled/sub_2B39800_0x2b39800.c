// Function: sub_2B39800
// Address: 0x2b39800
//
bool __fastcall sub_2B39800(__int64 a1, __int64 a2)
{
  __int64 v2; // r8
  int v3; // ecx
  __int64 *v4; // rdi
  unsigned int v5; // eax
  __int64 *v6; // rdx
  __int64 v7; // r9
  __int64 v9; // rdx
  int v10; // edx
  int v11; // r10d

  if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
  {
    v2 = a1 + 16;
    v3 = 3;
    v4 = (__int64 *)(a1 + 48);
  }
  else
  {
    v2 = *(_QWORD *)(a1 + 16);
    v9 = *(unsigned int *)(a1 + 24);
    v4 = (__int64 *)(v2 + 8 * v9);
    if ( !(_DWORD)v9 )
      return 0;
    v3 = v9 - 1;
  }
  v5 = v3 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v6 = (__int64 *)(v2 + 8LL * v5);
  v7 = *v6;
  if ( a2 == *v6 )
    return v6 != v4;
  v10 = 1;
  while ( v7 != -4096 )
  {
    v11 = v10 + 1;
    v5 = v3 & (v10 + v5);
    v6 = (__int64 *)(v2 + 8LL * v5);
    v7 = *v6;
    if ( a2 == *v6 )
      return v6 != v4;
    v10 = v11;
  }
  return 0;
}
