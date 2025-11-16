// Function: sub_1A04030
// Address: 0x1a04030
//
__int64 *__fastcall sub_1A04030(__int64 a1, __int64 a2)
{
  __int64 *result; // rax
  __int64 v4; // r13
  __int64 v5; // rsi
  unsigned int v6; // ebx

  if ( (*(_BYTE *)(a2 + 23) & 0x40) != 0 )
    result = *(__int64 **)(a2 - 8);
  else
    result = (__int64 *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v4 = *result;
  v5 = result[3];
  if ( v5 != *result && *(_BYTE *)(v5 + 16) > 0x10u )
  {
    if ( *(_BYTE *)(v4 + 16) <= 0x10u )
      return (__int64 *)sub_15FB800(a2);
    v6 = sub_1A03A70(a1, v5);
    result = (__int64 *)sub_1A03A70(a1, v4);
    if ( v6 < (unsigned int)result )
      return (__int64 *)sub_15FB800(a2);
  }
  return result;
}
