// Function: sub_C4FAE0
// Address: 0xc4fae0
//
__int64 __fastcall sub_C4FAE0(const void ***a1, const void ***a2)
{
  const void *v2; // r12
  const void *v3; // rbx
  size_t v4; // rdx
  int v5; // eax
  __int64 result; // rax

  v2 = (*a1)[1];
  v3 = (*a2)[1];
  v4 = (size_t)v3;
  if ( v2 <= v3 )
    v4 = (size_t)(*a1)[1];
  if ( v4 )
  {
    v5 = memcmp(**a1, **a2, v4);
    if ( v5 )
      return (v5 >> 31) | 1u;
  }
  result = 0;
  if ( v2 != v3 )
    return v2 < v3 ? -1 : 1;
  return result;
}
