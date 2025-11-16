// Function: sub_C11C00
// Address: 0xc11c00
//
__int64 __fastcall sub_C11C00(_QWORD *a1)
{
  if ( a1[4] == *a1 )
    return 0;
  *a1 = *(_QWORD *)(*a1 + 8LL);
  return 1;
}
