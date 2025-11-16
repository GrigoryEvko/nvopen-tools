// Function: sub_CA7AB0
// Address: 0xca7ab0
//
__int64 __fastcall sub_CA7AB0(_BYTE *a1, __int64 a2)
{
  _DWORD *v3; // rax
  char v4; // al
  char v5; // al
  char v6; // al
  char v7; // al
  _WORD *v8; // rdx
  _BYTE *v9; // rax
  char v10; // al
  _WORD *v11; // rax

  switch ( a2 )
  {
    case 1LL:
      v4 = *a1;
      if ( *a1 == 110 )
        return 256;
      if ( v4 > 110 )
      {
        if ( v4 == 121 )
          return 257;
        return 0;
      }
      if ( v4 == 78 )
        return 256;
      if ( v4 != 89 )
        return 0;
      return 257;
    case 2LL:
      v5 = *a1;
      if ( *a1 == 110 )
      {
        v10 = a1[1];
      }
      else
      {
        if ( v5 > 110 )
        {
          if ( v5 != 111 )
            return 0;
          v6 = a1[1];
LABEL_19:
          if ( v6 != 110 )
            return 0;
          return 257;
        }
        if ( v5 != 78 )
        {
          if ( v5 != 79 )
            return 0;
          v6 = a1[1];
          if ( v6 == 78 )
            return 257;
          goto LABEL_19;
        }
        v10 = a1[1];
        if ( v10 == 79 )
          return 256;
      }
      if ( v10 != 111 )
        return 0;
      return 256;
    case 3LL:
      v7 = *a1;
      if ( *a1 == 111 )
      {
        v11 = a1 + 1;
      }
      else
      {
        if ( v7 > 111 )
        {
          v8 = a1 + 1;
          if ( v7 != 121 )
            return 0;
LABEL_36:
          if ( *v8 == 29541 )
            return 257;
          return 0;
        }
        if ( v7 != 79 )
        {
          if ( v7 != 89 )
            return 0;
          v8 = a1 + 1;
          if ( *(_WORD *)(a1 + 1) == 21317 )
            return 257;
          goto LABEL_36;
        }
        v11 = a1 + 1;
        if ( *(_WORD *)(a1 + 1) == 17990 )
          return 256;
      }
      if ( *v11 == 26214 )
        return 256;
      return 0;
    case 4LL:
      if ( *a1 == 84 )
      {
        v9 = a1 + 1;
        if ( *(_WORD *)(a1 + 1) == 21842 && a1[3] == 69 )
          return 257;
      }
      else
      {
        if ( *a1 != 116 )
          return 0;
        v9 = a1 + 1;
      }
      if ( *(_WORD *)v9 == 30066 && v9[2] == 101 )
        return 257;
      return 0;
    case 5LL:
      if ( *a1 == 70 )
      {
        v3 = a1 + 1;
        if ( *(_DWORD *)(a1 + 1) == 1163086913 )
          return 256;
      }
      else
      {
        if ( *a1 != 102 )
          return 0;
        v3 = a1 + 1;
      }
      if ( *v3 != 1702063201 )
        return 0;
      return 256;
    default:
      return 0;
  }
}
