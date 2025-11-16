// Function: sub_3009540
// Address: 0x3009540
//
__int64 __fastcall sub_3009540(__int64 a1, char a2)
{
  char v2; // bl
  __int64 result; // rax
  int v4; // esi
  __int16 v5; // di
  unsigned __int64 v6; // rax
  __int64 v7; // rdx
  int v8; // esi
  int v9; // edi
  int v10; // edx

  v2 = *(_BYTE *)(a1 + 8);
  switch ( v2 )
  {
    case 0:
      return 11;
    case 1:
      return 10;
    case 2:
      return 12;
    case 3:
      return 13;
    case 4:
      return 14;
    case 5:
      return 15;
    case 6:
      return 16;
    case 7:
      return 263;
    case 10:
      return 268;
    case 12:
      result = 2;
      v10 = *(_DWORD *)(a1 + 8) >> 8;
      if ( v10 != 1 )
      {
        result = 3;
        if ( v10 != 2 )
        {
          result = 4;
          if ( v10 != 4 )
          {
            result = 5;
            if ( v10 != 8 )
            {
              result = 6;
              if ( v10 != 16 )
              {
                result = 7;
                if ( v10 != 32 )
                {
                  result = 8;
                  if ( v10 != 64 )
                    return 9 * (unsigned int)(v10 == 128);
                }
              }
            }
          }
        }
      }
      return result;
    case 17:
    case 18:
      v4 = *(_DWORD *)(a1 + 32);
      v5 = sub_3009540(*(_QWORD *)(a1 + 24), 0);
      if ( v2 == 18 )
        return sub_2D43AD0(v5, v4);
      else
        return sub_2D43050(v5, v4);
    case 20:
      v6 = *(_QWORD *)(a1 + 32);
      v7 = *(_QWORD *)(a1 + 24);
      if ( v6 == 15 )
      {
        if ( *(_QWORD *)v7 != 0x2E34366863726161LL
          || *(_DWORD *)(v7 + 8) != 1868789363
          || *(_WORD *)(v7 + 12) != 28277
          || (result = 270, *(_BYTE *)(v7 + 14) != 116) )
        {
          if ( *(_DWORD *)v7 != 1919512691 || *(_WORD *)(v7 + 4) != 11894 )
            goto LABEL_2;
          return 271;
        }
      }
      else
      {
        if ( v6 <= 5 )
          goto LABEL_2;
        if ( *(_DWORD *)v7 == 1919512691 && *(_WORD *)(v7 + 4) == 11894 )
          return 271;
        if ( v6 != 18
          || *(_QWORD *)v7 ^ 0x65762E7663736972LL | *(_QWORD *)(v7 + 8) ^ 0x7075742E726F7463LL
          || *(_WORD *)(v7 + 16) != 25964 )
        {
          goto LABEL_2;
        }
        v8 = **(_DWORD **)(a1 + 40);
        v9 = 8 * v8 * *(_DWORD *)(**(_QWORD **)(a1 + 16) + 32LL);
        if ( v9 == 16 && v8 == 2 )
        {
          return 229;
        }
        else if ( v9 == 24 && v8 == 3 )
        {
          return 230;
        }
        else if ( v9 == 32 && v8 == 4 )
        {
          return 231;
        }
        else if ( v9 == 40 && v8 == 5 )
        {
          return 232;
        }
        else if ( v9 == 48 && v8 == 6 )
        {
          return 233;
        }
        else
        {
          return sub_3006DC0(v9, v8, v8 == 3, v9 == 32, v8 == 4, v8 == 5);
        }
      }
      return result;
    default:
LABEL_2:
      if ( !a2 )
        BUG();
      return 1;
  }
}
