// Function: sub_FE89D0
// Address: 0xfe89d0
//
bool __fastcall sub_FE89D0(__int64 a1, unsigned int *a2)
{
  unsigned int v2; // ecx
  __int64 *v3; // r9
  __int64 *v4; // rsi
  __int64 *v5; // rax
  unsigned int v6; // r8d
  unsigned int v7; // edx

  v2 = *a2;
  if ( *a2 == -1 )
    return 0;
  v3 = *(__int64 **)(a1 + 32);
  v4 = (__int64 *)(a1 + 32);
  if ( (__int64 *)(a1 + 32) == v3 )
    return 0;
  v5 = *(__int64 **)(a1 + 56);
  if ( v4 == v5 )
  {
    v5 = (__int64 *)v5[1];
    v7 = v2 >> 7;
    *(_QWORD *)(a1 + 56) = v5;
    v6 = *((_DWORD *)v5 + 4);
    if ( v2 >> 7 == v6 )
    {
      if ( v4 == v5 )
        return 0;
      return (v5[((v2 >> 6) & 1) + 3] & (1LL << v2)) != 0;
    }
  }
  else
  {
    v6 = *((_DWORD *)v5 + 4);
    v7 = v2 >> 7;
    if ( v2 >> 7 == v6 )
      return (v5[((v2 >> 6) & 1) + 3] & (1LL << v2)) != 0;
  }
  if ( v6 > v7 )
  {
    if ( v3 != v5 )
    {
      while ( 1 )
      {
        v5 = (__int64 *)v5[1];
        if ( v3 == v5 )
          break;
        if ( *((_DWORD *)v5 + 4) <= v7 )
          goto LABEL_11;
      }
    }
    *(_QWORD *)(a1 + 56) = v5;
  }
  else
  {
    if ( v4 == v5 )
    {
LABEL_14:
      *(_QWORD *)(a1 + 56) = v5;
      return 0;
    }
    while ( v6 < v7 )
    {
      v5 = (__int64 *)*v5;
      if ( v4 == v5 )
        goto LABEL_14;
      v6 = *((_DWORD *)v5 + 4);
    }
LABEL_11:
    *(_QWORD *)(a1 + 56) = v5;
    if ( v4 == v5 )
      return 0;
  }
  if ( *((_DWORD *)v5 + 4) == v7 )
    return (v5[((v2 >> 6) & 1) + 3] & (1LL << v2)) != 0;
  return 0;
}
