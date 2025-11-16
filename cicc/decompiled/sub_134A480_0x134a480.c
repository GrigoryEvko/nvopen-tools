// Function: sub_134A480
// Address: 0x134a480
//
__int64 __fastcall sub_134A480(__int64 a1, _QWORD *a2)
{
  _QWORD *v2; // rax
  __int64 i; // rax
  __int64 result; // rax

  v2 = a2 + 2;
  do
  {
    *v2 = ~*v2;
    ++v2;
  }
  while ( v2 != a2 + 10 );
  for ( i = 0; i != 8; ++i )
    *(_QWORD *)(a1 + 8 * i + 184) &= a2[i + 2];
  result = a2[1];
  *(_QWORD *)(a1 + 176) -= result;
  return result;
}
