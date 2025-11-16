// Function: sub_149A790
// Address: 0x149a790
//
char __fastcall sub_149A790(__int64 a1, const void *a2, size_t a3)
{
  size_t v3; // r12
  unsigned int v5; // eax
  int v6; // eax

  v3 = *(_QWORD *)(a1 + 8);
  if ( a3 < v3 )
  {
    LOBYTE(v6) = 0;
    if ( !a3 )
      return v6;
    v5 = memcmp(*(const void **)a1, a2, a3);
    if ( !v5 )
      goto LABEL_5;
    return v5 >> 31;
  }
  if ( v3 )
  {
    v5 = memcmp(*(const void **)a1, a2, *(_QWORD *)(a1 + 8));
    if ( v5 )
      return v5 >> 31;
  }
  LOBYTE(v6) = 0;
  if ( a3 != v3 )
LABEL_5:
    LOBYTE(v6) = a3 > v3;
  return v6;
}
