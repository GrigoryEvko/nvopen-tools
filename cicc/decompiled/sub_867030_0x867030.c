// Function: sub_867030
// Address: 0x867030
//
void __fastcall sub_867030(_BYTE *a1)
{
  if ( a1 && !a1[42] )
  {
    if ( *(_QWORD *)a1 )
    {
      if ( a1[52] )
        *(_BYTE *)(*(_QWORD *)a1 + 52LL) = 1;
    }
    sub_85FCF0();
  }
}
