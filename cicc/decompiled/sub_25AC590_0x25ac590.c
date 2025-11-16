// Function: sub_25AC590
// Address: 0x25ac590
//
__int64 __fastcall sub_25AC590(_QWORD *a1)
{
  if ( a1[2] == *a1 )
    return 0;
  *a1 = *(_QWORD *)(*a1 + 8LL);
  return 1;
}
