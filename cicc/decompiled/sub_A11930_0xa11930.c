// Function: sub_A11930
// Address: 0xa11930
//
__int64 __fastcall sub_A11930(__int64 a1, __int64 *a2, int a3)
{
  _BYTE *v3; // rax

  if ( !a3 )
    return sub_A05F80(a1, 0);
  v3 = (_BYTE *)sub_A117A0(*a2, a3 - 1);
  return sub_A05F80(a1, v3);
}
