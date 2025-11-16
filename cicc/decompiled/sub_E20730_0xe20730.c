// Function: sub_E20730
// Address: 0xe20730
//
__int64 __fastcall sub_E20730(size_t *a1, size_t a2, const void *a3)
{
  unsigned int v3; // r14d
  size_t v4; // r13
  size_t v5; // r15

  v3 = 0;
  v4 = *a1;
  if ( *a1 >= a2 )
  {
    v5 = a1[1];
    if ( !a2 || !memcmp((const void *)a1[1], a3, a2) )
    {
      v3 = 1;
      a1[1] = a2 + v5;
      *a1 = v4 - a2;
    }
  }
  return v3;
}
