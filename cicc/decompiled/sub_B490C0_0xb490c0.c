// Function: sub_B490C0
// Address: 0xb490c0
//
__int64 __fastcall sub_B490C0(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  unsigned int v3; // esi
  __int64 v4; // rsi

  result = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  if ( *(_DWORD *)(a1 + 72) < (unsigned int)result + a2 )
  {
    v3 = a2 >> 1;
    if ( !(_DWORD)result )
      LODWORD(result) = 1;
    v4 = 2 * ((unsigned int)result + v3);
    *(_DWORD *)(a1 + 72) = v4;
    return sub_BD2A80(a1, v4, 0);
  }
  return result;
}
