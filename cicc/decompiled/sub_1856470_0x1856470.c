// Function: sub_1856470
// Address: 0x1856470
//
__int64 __fastcall sub_1856470(_QWORD *a1)
{
  if ( a1[1] == *a1 )
    return 0;
  *a1 = *(_QWORD *)(*a1 + 8LL);
  return 1;
}
