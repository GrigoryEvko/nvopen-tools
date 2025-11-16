// Function: sub_13FC1D0
// Address: 0x13fc1d0
//
char __fastcall sub_13FC1D0(__int64 a1, __int64 a2)
{
  __int64 *v2; // r14
  __int64 v3; // rax
  __int64 *v4; // rbx
  signed __int64 v5; // rax
  __int64 *v6; // r13
  char result; // al

  v2 = (__int64 *)a2;
  v3 = 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF);
  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
  {
    v4 = *(__int64 **)(a2 - 8);
    v2 = &v4[(unsigned __int64)v3 / 8];
  }
  else
  {
    v4 = (__int64 *)(a2 - v3);
  }
  v5 = 0xAAAAAAAAAAAAAAABLL * (v3 >> 3);
  if ( v5 >> 2 )
  {
    v6 = &v4[12 * (v5 >> 2)];
    while ( sub_13FC1A0(a1, *v4) )
    {
      if ( !sub_13FC1A0(a1, v4[3]) )
      {
        v4 += 3;
        return v2 == v4;
      }
      if ( !sub_13FC1A0(a1, v4[6]) )
        return v2 == v4 + 6;
      if ( !sub_13FC1A0(a1, v4[9]) )
        return v2 == v4 + 9;
      v4 += 12;
      if ( v6 == v4 )
      {
        v5 = 0xAAAAAAAAAAAAAAABLL * (v2 - v4);
        goto LABEL_14;
      }
    }
    return v2 == v4;
  }
LABEL_14:
  if ( v5 != 2 )
  {
    if ( v5 != 3 )
    {
      if ( v5 != 1 )
        return 1;
      goto LABEL_22;
    }
    if ( !sub_13FC1A0(a1, *v4) )
      return v2 == v4;
    v4 += 3;
  }
  if ( !sub_13FC1A0(a1, *v4) )
    return v4 == v2;
  v4 += 3;
LABEL_22:
  result = sub_13FC1A0(a1, *v4);
  if ( !result )
    return v4 == v2;
  return result;
}
