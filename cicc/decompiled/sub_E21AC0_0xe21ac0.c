// Function: sub_E21AC0
// Address: 0xe21ac0
//
__int64 __fastcall sub_E21AC0(__int64 a1, unsigned __int64 *a2)
{
  __int64 result; // rax
  char v3; // dl

  result = sub_E219C0(a1, a2);
  if ( result < 0 )
    *(_BYTE *)(a1 + 8) = 1;
  if ( v3 )
    return -result;
  return result;
}
