// Function: sub_8F1A40
// Address: 0x8f1a40
//
__int64 __fastcall sub_8F1A40(_BYTE *a1, __int64 a2, int *a3)
{
  int v3; // eax
  _BYTE *v4; // rsi
  char *v5; // rax
  char v6; // dl
  _BYTE *v8; // rsi
  const char *v9; // rax
  char v10; // dl
  _BYTE *v11; // rsi
  const char *v12; // rax
  char v13; // cl
  _BYTE *v14; // rsi
  char *v15; // rax
  char v16; // dl

  v3 = *a3;
  if ( !*a3 )
  {
    v8 = &a1[a2];
    v9 = "<invalid floating-point value>";
    v10 = 60;
    if ( a1 < v8 )
    {
      while ( 1 )
      {
        ++a1;
        ++v9;
        *(a1 - 1) = v10;
        if ( v8 == a1 )
          break;
        v10 = *v9;
        if ( !*v9 )
        {
          *a1 = 0;
          return 4294967293LL;
        }
      }
    }
    return 4294967294LL;
  }
  switch ( v3 )
  {
    case 3:
      v14 = &a1[a2];
      if ( a1 < v14 )
      {
        v15 = "NaN";
        v16 = 78;
        while ( 1 )
        {
          ++a1;
          ++v15;
          *(a1 - 1) = v16;
          if ( v14 == a1 )
            break;
          v16 = *v15;
          if ( !*v15 )
          {
            *a1 = 0;
            return 1;
          }
        }
      }
      return 4294967294LL;
    case 4:
      v11 = &a1[a2];
      if ( a1 < v11 )
      {
        v12 = "Infinity";
        v13 = 73;
        while ( 1 )
        {
          ++a1;
          ++v12;
          *(a1 - 1) = v13;
          if ( v11 == a1 )
            break;
          v13 = *v12;
          if ( !*v12 )
          {
            *a1 = 0;
            return 2 - ((unsigned int)(a3[1] == 0) - 1);
          }
        }
      }
      return 4294967294LL;
    case 6:
      v4 = &a1[a2];
      if ( a1 < v4 )
      {
        v5 = "0.0";
        v6 = 48;
        while ( 1 )
        {
          ++a1;
          ++v5;
          *(a1 - 1) = v6;
          if ( v4 == a1 )
            break;
          v6 = *v5;
          if ( !*v5 )
          {
            *a1 = 0;
            return 0;
          }
        }
      }
      return 4294967294LL;
  }
  return 0xFFFFFFFFLL;
}
