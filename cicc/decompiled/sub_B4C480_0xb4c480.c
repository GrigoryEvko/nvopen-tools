// Function: sub_B4C480
// Address: 0xb4c480
//
__int64 __fastcall sub_B4C480(__int64 a1, unsigned int a2)
{
  __int64 result; // rax
  __int64 v3; // rsi

  result = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  if ( *(_DWORD *)(a1 + 72) < a2 + (unsigned int)result )
  {
    v3 = 2 * ((unsigned int)result + (a2 >> 1));
    *(_DWORD *)(a1 + 72) = v3;
    return sub_BD2A80(a1, v3, 0);
  }
  return result;
}
