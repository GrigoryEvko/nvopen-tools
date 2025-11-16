// Function: sub_A4F520
// Address: 0xa4f520
//
__int64 __fastcall sub_A4F520(_QWORD *a1)
{
  if ( a1[2] == *a1 )
    return 0;
  *a1 = *(_QWORD *)(*a1 + 8LL);
  return 1;
}
