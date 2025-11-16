// Function: sub_8EEAD0
// Address: 0x8eead0
//
__int64 __fastcall sub_8EEAD0(unsigned __int8 *a1, int a2, int a3)
{
  int v4; // edi
  int v6; // ecx
  int v7; // edi
  _BYTE *v8; // rdx
  int v9; // esi
  int v10; // eax
  int v11; // edx
  __int64 result; // rax

  v4 = a2 + 14;
  v6 = a3;
  if ( a2 + 7 >= 0 )
    v4 = a2 + 7;
  v7 = v4 >> 3;
  if ( a2 <= 0 )
  {
    if ( a3 )
      return 256 - (unsigned int)*a1;
  }
  else if ( a3 )
  {
    v8 = a1;
    v9 = 0;
    while ( 1 )
    {
      v10 = (unsigned __int8)*v8 - v6;
      *v8 = v10;
      if ( v10 >= 0 )
        break;
      ++v9;
      ++v8;
      v6 = 1;
      if ( v7 <= v9 )
        return 256 - (unsigned int)*a1;
    }
  }
  LOBYTE(v11) = 0x80;
  if ( (a2 & 7) != 0 )
    v11 = 1 << (a2 % 8 - 1);
  result = 0;
  if ( ((unsigned __int8)v11 & a1[v7 - 1]) == 0 )
    return 256 - (unsigned int)*a1;
  return result;
}
