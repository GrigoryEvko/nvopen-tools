// Function: sub_1E69F00
// Address: 0x1e69f00
//
__int64 __fastcall sub_1E69F00(__int64 a1, int a2)
{
  unsigned int *v2; // rax
  unsigned int *v3; // rdx

  v2 = *(unsigned int **)(a1 + 360);
  v3 = *(unsigned int **)(a1 + 368);
  if ( v2 == v3 )
    return 0;
  while ( v2[1] != a2 )
  {
    v2 += 2;
    if ( v3 == v2 )
      return 0;
  }
  return *v2;
}
