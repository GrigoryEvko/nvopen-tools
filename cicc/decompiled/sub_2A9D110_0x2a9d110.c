// Function: sub_2A9D110
// Address: 0x2a9d110
//
__int64 __fastcall sub_2A9D110(_QWORD **a1, __int64 a2)
{
  _QWORD *v2; // rsi
  __int64 result; // rax

  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v2 = *(_QWORD **)(a2 - 8);
  else
    v2 = (_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  result = 0;
  if ( *v2 )
  {
    **a1 = *v2;
    return 1;
  }
  return result;
}
