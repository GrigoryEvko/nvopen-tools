// Function: sub_AF2CE0
// Address: 0xaf2ce0
//
__int64 __fastcall sub_AF2CE0(__int64 a1)
{
  unsigned __int8 v1; // al
  __int64 v2; // rdi

  v1 = *(_BYTE *)(a1 - 16);
  if ( (v1 & 2) != 0 )
    v2 = *(_QWORD *)(a1 - 32);
  else
    v2 = a1 - 16 - 8LL * ((v1 >> 2) & 0xF);
  return *(_QWORD *)(v2 + 32);
}
