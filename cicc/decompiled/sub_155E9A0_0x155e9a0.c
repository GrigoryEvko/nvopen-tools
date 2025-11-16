// Function: sub_155E9A0
// Address: 0x155e9a0
//
char __fastcall sub_155E9A0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rdi

  v2 = *a1;
  if ( !v2 )
    return a2 != 0;
  if ( a2 )
    return sub_155E7C0(v2, a2);
  return 0;
}
