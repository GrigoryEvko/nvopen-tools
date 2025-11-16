// Function: sub_1E73BB0
// Address: 0x1e73bb0
//
__int64 __fastcall sub_1E73BB0(__int64 a1)
{
  __int64 result; // rax
  unsigned int v2; // edx
  _DWORD *v3; // rsi
  unsigned int v4; // r8d
  int v5; // r9d
  unsigned int v6; // eax

  result = *(unsigned int *)(a1 + 36);
  if ( (_DWORD)result )
  {
    v2 = *(_DWORD *)(a1 + 32);
    if ( (unsigned int)result < v2 )
    {
      v3 = *(_DWORD **)(a1 + 16);
      v4 = *(_DWORD *)(a1 + 40);
      v5 = v3[69];
      v6 = v5 * result;
      if ( v6 < v4 )
        v6 = *(_DWORD *)(a1 + 40);
      result = (v6 + v5 * v4 * v2 - 1) / v6;
      *(_BYTE *)(a1 + 44) = (unsigned int)result > v3[68] * v3[1];
    }
  }
  return result;
}
