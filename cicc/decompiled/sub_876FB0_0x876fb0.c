// Function: sub_876FB0
// Address: 0x876fb0
//
__int64 __fastcall sub_876FB0(_QWORD *a1, __int64 a2)
{
  while ( 1 )
  {
    if ( !a1 )
      return 0;
    if ( a1[1] == a2 )
      break;
    a1 = (_QWORD *)*a1;
  }
  return 1;
}
