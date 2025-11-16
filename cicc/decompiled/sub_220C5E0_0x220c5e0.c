// Function: sub_220C5E0
// Address: 0x220c5e0
//
__int64 __fastcall sub_220C5E0(char a1, int a2, int a3)
{
  int v3; // ecx
  char v4; // r8
  int v5; // eax
  int v6; // edx
  char v7; // bl
  unsigned __int16 v8; // ax

  v3 = a2;
  v4 = a3;
  if ( (_BYTE)a3 == 3 )
  {
    if ( a1 )
    {
      v7 = 2;
      if ( (_BYTE)a2 )
        v3 = 4;
      LOBYTE(v6) = (_BYTE)a2 == 0 ? 4 : 1;
    }
    else
    {
      v4 = 4;
      LOBYTE(v6) = 3 - ((_BYTE)a2 == 0);
      if ( (_BYTE)a2 )
        v3 = 2;
      v7 = (_BYTE)a2 == 0 ? 3 : 1;
    }
  }
  else
  {
    if ( (char)a3 <= 3 )
    {
      if ( (_BYTE)a3 == 2 )
      {
        v5 = a1 == 0 ? 2 : 0;
        if ( (_BYTE)a2 )
        {
          LOBYTE(v6) = 4;
          v3 = 3;
          v7 = 1;
          if ( !a1 )
            LOBYTE(v6) = v4;
          v4 = v5 + 2;
        }
        else
        {
          v7 = 4;
          LOBYTE(v6) = 3;
          if ( !a1 )
            v7 = v4;
          v4 = v5 + 2;
        }
        goto LABEL_15;
      }
      if ( (a3 & 0x80u) == 0 )
      {
        if ( (_BYTE)a2 )
        {
          LOBYTE(v6) = 1;
          v4 = 3;
          v3 = a1 == 0 ? 2 : 4;
        }
        else
        {
          v4 = 3;
          v6 = a1 == 0 ? 2 : 4;
        }
        v7 = a1 == 0 ? 4 : 2;
        goto LABEL_15;
      }
LABEL_14:
      v3 = 0;
      LOBYTE(v6) = 0;
      v7 = 0;
      v4 = 0;
      goto LABEL_15;
    }
    if ( (_BYTE)a3 != 4 )
      goto LABEL_14;
    if ( a1 )
    {
      v7 = 3;
      if ( (_BYTE)a2 )
        v3 = a3;
      v4 = 2;
      LOBYTE(v6) = (_BYTE)a2 == 0 ? 4 : 1;
    }
    else
    {
      if ( (_BYTE)a2 )
        v3 = 3;
      LOBYTE(v6) = ((_BYTE)a2 == 0) + 2;
      v7 = ((_BYTE)a2 == 0) + 1;
    }
  }
LABEL_15:
  LOBYTE(v8) = v4;
  HIBYTE(v8) = v7;
  return (v3 << 24) | ((unsigned __int8)v6 << 16) | (unsigned int)v8;
}
