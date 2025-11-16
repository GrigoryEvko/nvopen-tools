// Function: sub_35456C0
// Address: 0x35456c0
//
__int64 __fastcall sub_35456C0(_QWORD *a1, __int64 a2)
{
  if ( a2 == *a1 )
    return (__int64)(a1 + 5);
  if ( a2 == a1[1] )
    return (__int64)(a1 + 41);
  return a1[2] + 288LL * *(unsigned int *)(a2 + 200);
}
