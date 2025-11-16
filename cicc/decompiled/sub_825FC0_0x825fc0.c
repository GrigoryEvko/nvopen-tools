// Function: sub_825FC0
// Address: 0x825fc0
//
__int64 __fastcall sub_825FC0(__int64 a1)
{
  __int64 v1; // rax

  while ( 1 )
  {
    if ( !a1 )
      return 0;
    if ( (unsigned __int8)(*(_BYTE *)(a1 + 28) - 3) <= 2u )
    {
      v1 = *(_QWORD *)(a1 + 32);
      if ( (*(_BYTE *)(v1 + 124) & 2) != 0 && !*(_QWORD *)(v1 + 8) )
        break;
    }
    a1 = *(_QWORD *)(a1 + 16);
  }
  return 1;
}
