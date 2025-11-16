// Function: sub_AD1300
// Address: 0xad1300
//
__int64 __fastcall sub_AD1300(__int64 **a1, __int64 *a2, __int64 a3)
{
  __int64 result; // rax

  result = sub_ACB110(a1, (__int64)a2, a3);
  if ( !result )
    return sub_AD1150(**a1 + 1744, (__int64)a1, a2, a3);
  return result;
}
