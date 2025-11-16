// Function: sub_1548A60
// Address: 0x1548a60
//
__int64 __fastcall sub_1548A60(_QWORD *a1)
{
  if ( a1[1] == *a1 )
    return 0;
  *a1 = *(_QWORD *)(*a1 + 8LL);
  return 1;
}
