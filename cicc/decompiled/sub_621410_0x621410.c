// Function: sub_621410
// Address: 0x621410
//
__int64 __fastcall sub_621410(__int64 a1, int a2, int *a3)
{
  int v4; // r11d
  unsigned int v5; // eax
  int v6; // edi
  char v7; // r9
  int v8; // esi
  char v10; // r9
  char v11; // r10
  __int64 result; // rax
  unsigned __int64 v13; // rdx

  v4 = 0;
  v5 = (unsigned int)(a2 >> 31) >> 28;
  v6 = a2 / 16;
  v7 = (a2 + v5) & 0xF;
  v8 = a2 / 16 + 1;
  v10 = v7 - v5;
  v11 = 16 - v10;
  for ( result = 0; result != 8; ++result )
  {
    if ( v6 > (int)result && *(_WORD *)(a1 + 2 * result) )
    {
      v4 = 1;
    }
    else if ( v6 == (_DWORD)result && (unsigned __int64)*(unsigned __int16 *)(a1 + 2 * result) >> v11 )
    {
      v4 = 1;
    }
    v13 = 0;
    if ( (unsigned int)(v6 + result) <= 7 )
      v13 = (unsigned __int64)*(unsigned __int16 *)(a1 + 2LL * v6 + 2 * result) << v10;
    if ( (unsigned int)v8 <= 7 )
      v13 |= (unsigned __int64)*(unsigned __int16 *)(a1 + 2LL * v8) >> v11;
    *(_WORD *)(a1 + 2 * result) = v13;
    ++v8;
  }
  *a3 = v4;
  return result;
}
