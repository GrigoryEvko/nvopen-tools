// Function: sub_C36030
// Address: 0xc36030
//
__int64 __fastcall sub_C36030(unsigned int **a1)
{
  unsigned int *v1; // rax
  unsigned int v2; // edx
  unsigned int v3; // r8d

  v1 = *a1;
  if ( (*a1)[4] != 1 )
  {
    v2 = *v1;
    return v2 + 1;
  }
  if ( v1[5] != 2 )
  {
    v2 = *v1;
    v3 = *v1;
    if ( *((_BYTE *)v1 + 25) )
      return v3;
    return v2 + 1;
  }
  return v1[1] - 1;
}
