// Function: sub_32435C0
// Address: 0x32435c0
//
char __fastcall sub_32435C0(__int64 a1, _BYTE *a2, __int64 a3)
{
  char result; // al

  if ( !*a2 )
    *(_BYTE *)(a1 + 100) = *(_BYTE *)(a1 + 100) & 0xF8 | 2;
  result = sub_AF46F0(a3);
  if ( result )
    return sub_3243580(a1, a2);
  return result;
}
