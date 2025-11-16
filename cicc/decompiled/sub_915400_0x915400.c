// Function: sub_915400
// Address: 0x915400
//
__int64 __fastcall sub_915400(__int64 *a1)
{
  __int64 result; // rax

  sub_9151E0(a1);
  sub_914410(a1);
  sub_90A560(a1);
  if ( a1[46] )
  {
    sub_BA93D0(*a1, 1, "Debug Info Version", 18, 3);
    sub_ADCDB0(a1[46] + 16);
  }
  result = dword_4D04654;
  if ( !dword_4D04654 )
    return sub_90A380((__int64 **)a1);
  return result;
}
