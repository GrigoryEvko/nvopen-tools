// Function: sub_321DF00
// Address: 0x321df00
//
__int64 __fastcall sub_321DF00(__int64 a1)
{
  __int64 v1; // rax
  unsigned __int8 v2; // dl

  v1 = *(_QWORD *)(a1 + 8);
  v2 = *(_BYTE *)(v1 - 16);
  if ( (v2 & 2) != 0 )
    return *(_QWORD *)(*(_QWORD *)(v1 - 32) + 24LL);
  else
    return *(_QWORD *)(v1 - 16 - 8LL * ((v2 >> 2) & 0xF) + 24);
}
