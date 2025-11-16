// Function: sub_8EEF20
// Address: 0x8eef20
//
__int64 __fastcall sub_8EEF20(__int64 a1)
{
  int v1; // esi
  __int64 result; // rax
  int v3; // eax
  int v4; // ecx
  int v5; // edx
  int v6; // eax
  int v7; // eax

  v1 = *(_DWORD *)(a1 + 8);
  result = 0;
  if ( v1 > 0 )
  {
    v3 = *(_DWORD *)(a1 + 28);
    v4 = 8;
    if ( (v3 & 7) != 0 )
      v4 = v3 % 8;
    v5 = v3 + 14;
    v6 = v3 + 7;
    if ( v6 < 0 )
      v6 = v5;
    v7 = v6 >> 3;
    if ( v1 > v4 )
      return (*(unsigned __int8 *)(a1 + v7 - 1 + 12) << (v1 - v4))
           + ((int)*(unsigned __int8 *)(a1 + v7 - 2 + 12) >> (8 - (v1 - v4)));
    else
      return (int)*(unsigned __int8 *)(a1 + v7 - 1 + 12) >> (v4 - v1);
  }
  return result;
}
