// Function: sub_1370EA0
// Address: 0x1370ea0
//
bool __fastcall sub_1370EA0(__int64 a1, unsigned int *a2)
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
  v3 = *(__int64 **)(a1 + 40);
  v4 = (__int64 *)(a1 + 40);
  if ( v3 == (__int64 *)(a1 + 40) )
    return 0;
  v5 = *(__int64 **)(a1 + 32);
  if ( v4 != v5 )
  {
    v6 = *((_DWORD *)v5 + 4);
    v7 = v2 >> 7;
    if ( v2 >> 7 != v6 )
      goto LABEL_5;
    return (v5[((v2 >> 6) & 1) + 3] & (1LL << v2)) != 0;
  }
  v5 = (__int64 *)v5[1];
  v7 = v2 >> 7;
  *(_QWORD *)(a1 + 32) = v5;
  v6 = *((_DWORD *)v5 + 4);
  if ( v2 >> 7 == v6 )
  {
    if ( v4 == v5 )
      return 0;
    return (v5[((v2 >> 6) & 1) + 3] & (1LL << v2)) != 0;
  }
LABEL_5:
  if ( v7 >= v6 )
  {
    if ( v5 == v4 )
    {
LABEL_14:
      *(_QWORD *)(a1 + 32) = v5;
      return 0;
    }
    while ( v7 > v6 )
    {
      v5 = (__int64 *)*v5;
      if ( v4 == v5 )
        goto LABEL_14;
      v6 = *((_DWORD *)v5 + 4);
    }
LABEL_11:
    *(_QWORD *)(a1 + 32) = v5;
    if ( v4 == v5 )
      return 0;
    goto LABEL_12;
  }
  if ( v5 != v3 )
  {
    do
      v5 = (__int64 *)v5[1];
    while ( v3 != v5 && v7 < *((_DWORD *)v5 + 4) );
    goto LABEL_11;
  }
  *(_QWORD *)(a1 + 32) = v5;
LABEL_12:
  if ( v7 == *((_DWORD *)v5 + 4) )
    return (v5[((v2 >> 6) & 1) + 3] & (1LL << v2)) != 0;
  return 0;
}
