// Function: sub_87D720
// Address: 0x87d720
//
__int64 __fastcall sub_87D720(_QWORD *a1, __int64 a2)
{
  _QWORD *v2; // rbx

  if ( !a1 )
    return 0;
  v2 = a1;
  do
  {
    if ( sub_87D6D0(a2, v2[1]) )
      return 1;
    v2 = (_QWORD *)*v2;
  }
  while ( v2 );
  return 0;
}
