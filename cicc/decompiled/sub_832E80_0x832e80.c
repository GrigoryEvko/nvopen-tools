// Function: sub_832E80
// Address: 0x832e80
//
void __fastcall sub_832E80(_QWORD *a1)
{
  while ( a1 )
  {
    sub_832D70((__int64)a1);
    if ( !*a1 )
      break;
    if ( *(_BYTE *)(*a1 + 8LL) == 3 )
      a1 = (_QWORD *)sub_6BBB10(a1);
    else
      a1 = (_QWORD *)*a1;
  }
}
