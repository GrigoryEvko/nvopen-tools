// Function: sub_3545690
// Address: 0x3545690
//
__int64 __fastcall sub_3545690(_QWORD *a1, __int64 a2)
{
  if ( *a1 == a2 )
    return (__int64)(a1 + 5);
  if ( a1[1] == a2 )
    return (__int64)(a1 + 41);
  return a1[2] + 288LL * *(unsigned int *)(a2 + 200);
}
