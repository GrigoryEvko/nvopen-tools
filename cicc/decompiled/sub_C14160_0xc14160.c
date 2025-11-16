// Function: sub_C14160
// Address: 0xc14160
//
__int64 __fastcall sub_C14160(_QWORD *a1)
{
  if ( a1[4] == *a1 )
    return 0;
  *a1 = *(_QWORD *)(*a1 + 8LL);
  return 1;
}
