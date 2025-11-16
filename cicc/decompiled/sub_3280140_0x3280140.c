// Function: sub_3280140
// Address: 0x3280140
//
__int64 __fastcall sub_3280140(__int64 a1)
{
  int v1; // eax
  unsigned int v2; // edx

  v1 = *(unsigned __int16 *)a1;
  if ( !(_WORD)v1 )
    return sub_3007030(a1);
  v2 = v1 - 10;
  LOBYTE(v2) = (unsigned __int16)(v1 - 126) <= 0x31u || (unsigned __int16)(v1 - 10) <= 6u;
  if ( !(_BYTE)v2 )
    LOBYTE(v2) = (unsigned __int16)(v1 - 208) <= 0x14u;
  return v2;
}
