// Function: sub_149B2B0
// Address: 0x149b2b0
//
char __fastcall sub_149B2B0(__int64 a1, __int64 a2)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // r12
  unsigned int v4; // eax
  int v5; // eax

  v2 = *(_QWORD *)(a1 + 8);
  v3 = *(_QWORD *)(a2 + 8);
  if ( v3 < v2 )
  {
    LOBYTE(v5) = 0;
    if ( !v3 )
      return v5;
    v4 = memcmp(*(const void **)a1, *(const void **)a2, *(_QWORD *)(a2 + 8));
    if ( !v4 )
      goto LABEL_5;
    return v4 >> 31;
  }
  if ( v2 )
  {
    v4 = memcmp(*(const void **)a1, *(const void **)a2, *(_QWORD *)(a1 + 8));
    if ( v4 )
      return v4 >> 31;
  }
  LOBYTE(v5) = 0;
  if ( v3 != v2 )
LABEL_5:
    LOBYTE(v5) = v3 > v2;
  return v5;
}
