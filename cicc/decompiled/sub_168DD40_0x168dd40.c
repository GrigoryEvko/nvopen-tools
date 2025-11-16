// Function: sub_168DD40
// Address: 0x168dd40
//
__int64 __fastcall sub_168DD40(_QWORD *a1)
{
  if ( a1[1] == *a1 )
    return 0;
  *a1 = *(_QWORD *)(*a1 + 8LL);
  return 1;
}
