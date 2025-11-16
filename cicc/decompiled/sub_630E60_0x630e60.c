// Function: sub_630E60
// Address: 0x630e60
//
__int64 __fastcall sub_630E60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 result; // rax
  __int64 v6; // rsi

  result = 0;
  if ( a4 )
    result = (*(_BYTE *)(a4 + 144) & 4) != 0;
  if ( !a2 )
    a2 = a1;
  for ( ; *(_BYTE *)(a2 + 140) == 12; a2 = *(_QWORD *)(a2 + 160) )
    ;
  v6 = *(_QWORD *)(a2 + 128);
  if ( (_BYTE)result )
  {
    result = *(unsigned __int8 *)(a4 + 137);
    if ( dword_4F06BA0 * v6 == result )
      return result;
    return sub_6851C0(2523, a5);
  }
  while ( *(_BYTE *)(a3 + 140) == 12 )
    a3 = *(_QWORD *)(a3 + 160);
  if ( v6 != *(_QWORD *)(a3 + 128) )
    return sub_6851C0(2523, a5);
  return result;
}
