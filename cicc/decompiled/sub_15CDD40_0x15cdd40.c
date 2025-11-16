// Function: sub_15CDD40
// Address: 0x15cdd40
//
void __fastcall sub_15CDD40(__int64 *a1)
{
  __int64 i; // rbx

  for ( i = *a1; i; *a1 = i )
  {
    if ( (unsigned __int8)(*(_BYTE *)(sub_1648700(i) + 16) - 25) <= 9u )
      break;
    i = *(_QWORD *)(i + 8);
  }
}
