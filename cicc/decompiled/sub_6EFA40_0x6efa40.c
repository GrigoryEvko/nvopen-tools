// Function: sub_6EFA40
// Address: 0x6efa40
//
__int64 __fastcall sub_6EFA40(__int64 *a1, int *a2)
{
  __int64 result; // rax

  result = sub_6EED10(*a1, a2, 0, 0, 0, 0);
  if ( *a2 )
  {
    result = sub_73E1B0(result, a2);
    *a1 = result;
  }
  return result;
}
