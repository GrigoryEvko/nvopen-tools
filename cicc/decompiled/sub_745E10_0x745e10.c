// Function: sub_745E10
// Address: 0x745e10
//
__int64 __fastcall sub_745E10(__int64 a1, _DWORD *a2, __int64 (__fastcall **a3)(const char *, _QWORD))
{
  __int64 result; // rax

  if ( *(char *)(a1 + 142) < 0 )
    sub_745D60("__aligned__", *(unsigned int *)(a1 + 136), a2, a3);
  result = *(unsigned __int8 *)(a1 + 140);
  if ( (unsigned __int8)(result - 9) > 2u )
  {
    if ( (_BYTE)result == 2 )
    {
      result = *(_BYTE *)(a1 + 161) & 0x28;
      if ( (_BYTE)result == 40 )
        return sub_7450F0("__packed__", a2, a3);
    }
  }
  else if ( (*(_BYTE *)(a1 + 179) & 0x20) != 0 )
  {
    return sub_7450F0("__packed__", a2, a3);
  }
  return result;
}
