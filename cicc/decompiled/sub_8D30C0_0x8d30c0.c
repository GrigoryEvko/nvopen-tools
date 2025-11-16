// Function: sub_8D30C0
// Address: 0x8d30c0
//
__int64 __fastcall sub_8D30C0(__int64 a1)
{
  char v1; // al
  unsigned __int8 v3; // al

  while ( 1 )
  {
    v1 = *(_BYTE *)(a1 + 140);
    if ( v1 != 12 )
      break;
    a1 = *(_QWORD *)(a1 + 160);
  }
  if ( v1 == 6 && (v3 = *(_BYTE *)(a1 + 168), (v3 & 1) != 0) )
    return ((v3 >> 1) ^ 1) & 1;
  else
    return 0;
}
