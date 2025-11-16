// Function: sub_370CDF0
// Address: 0x370cdf0
//
char __fastcall sub_370CDF0(__int64 a1, __int64 a2)
{
  size_t v2; // r12
  size_t v3; // rbx
  size_t v4; // rdx
  unsigned int v5; // eax
  int v6; // eax

  v2 = *(_QWORD *)(a1 + 8);
  v3 = *(_QWORD *)(a2 + 8);
  v4 = v3;
  if ( v2 <= v3 )
    v4 = *(_QWORD *)(a1 + 8);
  if ( v4 && (v5 = memcmp(*(const void **)a1, *(const void **)a2, v4)) != 0 )
  {
    return v5 >> 31;
  }
  else
  {
    LOBYTE(v6) = v2 < v3;
    if ( v2 == v3 )
      LOBYTE(v6) = 0;
  }
  return v6;
}
