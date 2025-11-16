// Function: sub_15F7D80
// Address: 0x15f7d80
//
__int64 __fastcall sub_15F7D80(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v7; // rsi

  result = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  if ( *(_DWORD *)(a1 + 56) < a2 + (unsigned int)result )
  {
    v7 = 2 * ((unsigned int)result + (a2 >> 1));
    *(_DWORD *)(a1 + 56) = v7;
    return sub_16488D0(a1, v7, 0, a4, a5, a6);
  }
  return result;
}
