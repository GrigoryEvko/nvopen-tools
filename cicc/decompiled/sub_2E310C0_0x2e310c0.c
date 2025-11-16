// Function: sub_2E310C0
// Address: 0x2e310c0
//
__int64 __fastcall sub_2E310C0(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax

  if ( a1 != a2 )
  {
    while ( a4 != a3 )
    {
      result = *a1;
      *(_QWORD *)(a3 + 24) = *a1;
      a3 = *(_QWORD *)(a3 + 8);
    }
  }
  return result;
}
