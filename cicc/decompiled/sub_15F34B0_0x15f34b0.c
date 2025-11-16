// Function: sub_15F34B0
// Address: 0x15f34b0
//
__int64 __fastcall sub_15F34B0(__int64 a1)
{
  int v1; // edx
  __int64 result; // rax
  char v3; // al
  _BOOL4 v4; // edx

  v1 = *(unsigned __int8 *)(a1 + 16);
  result = (unsigned int)(v1 - 50);
  LOBYTE(result) = (*(_BYTE *)(a1 + 16) & 0xFB) == 35 || (unsigned int)result <= 2;
  if ( !(_BYTE)result && (((_BYTE)v1 - 36) & 0xFB) == 0 )
  {
    v3 = *(_BYTE *)(a1 + 17);
    v4 = (v3 & 0x10) != 0;
    result = (v3 & 2) != 0;
    if ( (_DWORD)result )
      return v4;
  }
  return result;
}
