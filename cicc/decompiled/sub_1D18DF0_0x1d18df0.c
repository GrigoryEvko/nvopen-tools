// Function: sub_1D18DF0
// Address: 0x1d18df0
//
__int64 __fastcall sub_1D18DF0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  _QWORD *v3; // rdx

  v2 = *(_QWORD **)(a2 + 32);
  v3 = &v2[5 * *(unsigned int *)(a2 + 56)];
  if ( v2 == v3 )
    return 0;
  while ( a1 != *v2 )
  {
    v2 += 5;
    if ( v3 == v2 )
      return 0;
  }
  return 1;
}
