// Function: sub_C93710
// Address: 0xc93710
//
unsigned __int64 __fastcall sub_C93710(_QWORD *a1, char a2, unsigned __int64 a3)
{
  unsigned __int64 result; // rax

  if ( a1[1] <= a3 )
    a3 = a1[1];
  result = a3 - 1;
  if ( a3 )
  {
    do
    {
      if ( *(_BYTE *)(*a1 + result) != a2 )
        break;
    }
    while ( result-- != 0 );
  }
  return result;
}
