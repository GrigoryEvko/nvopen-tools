// Function: sub_16431F0
// Address: 0x16431f0
//
__int64 __fastcall sub_16431F0(__int64 a1)
{
  unsigned __int8 i; // al
  int v2; // edx
  __int64 result; // rax

  for ( i = *(_BYTE *)(a1 + 8); i == 16; i = *(_BYTE *)(a1 + 8) )
    a1 = *(_QWORD *)(a1 + 24);
  v2 = i;
  switch ( i )
  {
    case 1u:
      return 11;
    case 2u:
      return 24;
    case 3u:
      return 53;
    case 4u:
      return 64;
  }
  result = 0xFFFFFFFFLL;
  if ( v2 == 5 )
    return 113;
  return result;
}
