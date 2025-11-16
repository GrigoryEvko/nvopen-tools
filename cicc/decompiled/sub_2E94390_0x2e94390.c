// Function: sub_2E94390
// Address: 0x2e94390
//
__int64 __fastcall sub_2E94390(__int64 a1, __int64 a2)
{
  __int64 i; // r12

  for ( i = *(_QWORD *)(a2 + 8); a1 + 48 != i; i = *(_QWORD *)(i + 8) )
  {
    if ( (*(_BYTE *)(i + 44) & 4) == 0 )
      break;
  }
  sub_2E92D10(a1, a2, i);
  return i;
}
