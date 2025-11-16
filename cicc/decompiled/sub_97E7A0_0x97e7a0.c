// Function: sub_97E7A0
// Address: 0x97e7a0
//
char __fastcall sub_97E7A0(__int64 a1, const void *a2, size_t a3)
{
  size_t v3; // r12
  size_t v4; // rbx
  unsigned int v5; // eax
  int v6; // eax

  v3 = *(_QWORD *)(a1 + 8);
  v4 = a3;
  if ( v3 <= a3 )
    a3 = *(_QWORD *)(a1 + 8);
  if ( a3 && (v5 = memcmp(*(const void **)a1, a2, a3)) != 0 )
  {
    return v5 >> 31;
  }
  else
  {
    LOBYTE(v6) = v3 < v4;
    if ( v3 == v4 )
      LOBYTE(v6) = 0;
  }
  return v6;
}
