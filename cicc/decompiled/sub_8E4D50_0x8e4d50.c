// Function: sub_8E4D50
// Address: 0x8e4d50
//
_BOOL8 __fastcall sub_8E4D50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _BOOL8 result; // rax
  unsigned __int8 v6; // dl
  __int64 v7; // rcx
  __int64 v8; // rdx

  result = 0;
  if ( *(_QWORD *)(a1 + 128) == *(_QWORD *)(a2 + 128) )
  {
    v6 = *(_BYTE *)(a1 + 144);
    v7 = (v6 ^ *(_BYTE *)(a2 + 144)) & 4;
    if ( ((v6 ^ *(_BYTE *)(a2 + 144)) & 4) == 0 )
    {
      v8 = *(_BYTE *)(a1 + 144) & 4;
      if ( (*(_BYTE *)(a1 + 144) & 4) == 0 )
        return (unsigned int)sub_8E4940(*(_QWORD *)(a1 + 120), *(_QWORD *)(a2 + 120), v8, v7, a5) != 0;
      v8 = *(unsigned __int16 *)(a2 + 136);
      if ( *(_WORD *)(a1 + 136) == (_WORD)v8 )
        return (unsigned int)sub_8E4940(*(_QWORD *)(a1 + 120), *(_QWORD *)(a2 + 120), v8, v7, a5) != 0;
    }
  }
  return result;
}
