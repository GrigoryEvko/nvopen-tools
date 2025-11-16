// Function: sub_28EFCB0
// Address: 0x28efcb0
//
__int64 __fastcall sub_28EFCB0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  _BYTE *v4; // r13
  _BYTE *v5; // rsi
  unsigned int v6; // ebx

  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    result = *(_QWORD *)(a2 - 8);
  else
    result = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  v4 = *(_BYTE **)result;
  v5 = *(_BYTE **)(result + 32);
  if ( v5 != *(_BYTE **)result && *v5 > 0x15u )
  {
    if ( *v4 <= 0x15u )
      return sub_B506C0((unsigned __int8 *)a2);
    v6 = sub_28EF780(a1, v5);
    result = sub_28EF780(a1, v4);
    if ( v6 < (unsigned int)result )
      return sub_B506C0((unsigned __int8 *)a2);
  }
  return result;
}
