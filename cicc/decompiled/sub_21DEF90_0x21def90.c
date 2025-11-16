// Function: sub_21DEF90
// Address: 0x21def90
//
__int64 __fastcall sub_21DEF90(__int64 a1)
{
  int v1; // eax
  unsigned __int64 v2; // rdx
  __int64 result; // rax
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 *v6; // rdx
  __int64 v7; // rdx
  unsigned int v8; // edx

  v1 = *(unsigned __int16 *)(a1 + 24);
  v2 = (unsigned int)(v1 - 185);
  if ( (unsigned __int16)(v1 - 185) <= 0x35u )
  {
    v4 = 0x3FFFFD00000003LL;
    if ( !_bittest64(&v4, v2) )
      return 0;
  }
  else if ( (unsigned __int16)(v1 - 44) <= 1u )
  {
    if ( (*(_BYTE *)(a1 + 26) & 2) == 0 )
      return 0;
  }
  else if ( (__int16)v1 <= 658 )
  {
    return 0;
  }
  v5 = **(_QWORD **)(a1 + 104);
  result = 0;
  if ( (v5 & 4) == 0 )
  {
    v6 = (__int64 *)(v5 & 0xFFFFFFFFFFFFFFF8LL);
    if ( v6 )
    {
      v7 = *v6;
      if ( *(_BYTE *)(v7 + 8) == 15 )
      {
        v8 = *(_DWORD *)(v7 + 8) >> 8;
        if ( v8 == 4 )
        {
          return 2;
        }
        else if ( v8 > 4 )
        {
          result = 5;
          if ( v8 != 5 )
            return 4 * (unsigned int)(v8 == 101);
        }
        else
        {
          result = 1;
          if ( v8 != 1 )
            return 3 * (unsigned int)(v8 == 3);
        }
      }
    }
  }
  return result;
}
