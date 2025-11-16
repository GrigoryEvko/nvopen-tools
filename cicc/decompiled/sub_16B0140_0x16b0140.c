// Function: sub_16B0140
// Address: 0x16b0140
//
__int64 __fastcall sub_16B0140(const void ***a1, const void ***a2)
{
  const void **v2; // rax
  const void *v3; // rdi
  size_t v4; // rbx
  size_t v5; // r12
  const void *v6; // rsi
  int v7; // eax
  __int64 result; // rax

  v2 = *a1;
  v3 = **a1;
  v4 = (size_t)v2[1];
  v5 = (size_t)(*a2)[1];
  v6 = **a2;
  if ( v4 <= v5 )
  {
    if ( !v4 || (v7 = memcmp(v3, v6, v4)) == 0 )
    {
      result = 0;
      if ( v4 == v5 )
        return result;
      return v4 < v5 ? -1 : 1;
    }
    return (v7 >> 31) | 1u;
  }
  result = 1;
  if ( v5 )
  {
    v7 = memcmp(v3, v6, v5);
    if ( !v7 )
      return v4 < v5 ? -1 : 1;
    return (v7 >> 31) | 1u;
  }
  return result;
}
