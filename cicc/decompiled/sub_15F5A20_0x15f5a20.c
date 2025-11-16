// Function: sub_15F5A20
// Address: 0x15f5a20
//
__int64 __fastcall sub_15F5A20(__int64 a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  unsigned int v7; // esi
  __int64 v8; // rsi

  result = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  if ( *(_DWORD *)(a1 + 56) < (unsigned int)result + a2 )
  {
    v7 = a2 >> 1;
    if ( !(_DWORD)result )
      LODWORD(result) = 1;
    v8 = 2 * ((unsigned int)result + v7);
    *(_DWORD *)(a1 + 56) = v8;
    return sub_16488D0(a1, v8, 0, a4, a5, a6);
  }
  return result;
}
