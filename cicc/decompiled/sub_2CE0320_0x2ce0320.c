// Function: sub_2CE0320
// Address: 0x2ce0320
//
__int64 __fastcall sub_2CE0320(__int64 a1, unsigned int a2, _DWORD *a3)
{
  unsigned int v4; // eax
  unsigned int v5; // r13d
  __int64 result; // rax

  v4 = sub_CEA1F0(a2);
  if ( !(_BYTE)v4 )
  {
    v5 = v4;
    if ( (!byte_5014508 || !sub_2CE02B0(a1, a2)) && (a2 - 238 > 5 || ((1LL << ((unsigned __int8)a2 + 18)) & 0x29) == 0) )
      return v5;
    goto LABEL_7;
  }
  LOBYTE(result) = sub_CEA230(a2);
  if ( !(_BYTE)result )
  {
LABEL_7:
    *a3 = 0;
    return 1;
  }
  *a3 = 1;
  return (unsigned int)result;
}
