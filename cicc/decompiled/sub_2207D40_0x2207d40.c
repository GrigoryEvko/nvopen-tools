// Function: sub_2207D40
// Address: 0x2207d40
//
__int64 __fastcall sub_2207D40(__int64 a1)
{
  int v2; // eax

  if ( !sub_2207CD0((_QWORD *)a1) )
    return 0;
  if ( !*(_BYTE *)(a1 + 8) )
  {
    *(_QWORD *)a1 = 0;
    return a1;
  }
  v2 = fclose(*(FILE **)a1);
  *(_QWORD *)a1 = 0;
  if ( v2 )
    return 0;
  else
    return a1;
}
