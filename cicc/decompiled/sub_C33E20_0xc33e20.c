// Function: sub_C33E20
// Address: 0xc33e20
//
__int64 __fastcall sub_C33E20(__int64 a1, __int64 a2)
{
  char v2; // al
  char v3; // al
  __int64 result; // rax

  v2 = *(_BYTE *)(a2 + 20) & 8 | *(_BYTE *)(a1 + 20) & 0xF7;
  *(_BYTE *)(a1 + 20) = v2;
  v3 = *(_BYTE *)(a2 + 20) & 7 | v2 & 0xF8;
  *(_BYTE *)(a1 + 20) = v3;
  result = v3 & 7;
  *(_DWORD *)(a1 + 16) = *(_DWORD *)(a2 + 16);
  if ( (_BYTE)result == 1 || (_BYTE)result && (_BYTE)result != 3 )
    return sub_C33DD0(a1, a2);
  return result;
}
