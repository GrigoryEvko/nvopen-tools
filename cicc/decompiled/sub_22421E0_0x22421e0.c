// Function: sub_22421E0
// Address: 0x22421e0
//
__int64 __fastcall sub_22421E0(__int64 a1, int *a2, unsigned __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // rdx

  for ( result = 0; a3 > (unsigned __int64)a2; result = v5 + __ROL8__(result, 7) )
    v5 = *a2++;
  return result;
}
