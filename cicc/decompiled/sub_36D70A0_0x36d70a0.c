// Function: sub_36D70A0
// Address: 0x36d70a0
//
__int64 __fastcall sub_36D70A0(__int64 a1, char a2, char a3, char a4, char a5)
{
  unsigned int v5; // eax
  unsigned int v7; // eax
  unsigned int v8; // eax
  int v9; // eax
  int v10; // eax
  int v11; // eax
  int v12; // eax
  int v13; // eax

  if ( !a4 )
  {
    switch ( a1 )
    {
      case 1LL:
        v12 = a2 == 0 ? 2 : 0;
        if ( a5 )
        {
          if ( a3 )
            return (unsigned int)(v12 + 904);
          else
            return (unsigned int)(v12 + 903);
        }
        else if ( a3 )
        {
          return (unsigned int)(v12 + 936);
        }
        else
        {
          return (unsigned int)(v12 + 935);
        }
      case 2LL:
        v13 = a2 == 0 ? 2 : 0;
        if ( a5 )
        {
          if ( a3 )
            return (unsigned int)(v13 + 908);
          else
            return (unsigned int)(v13 + 907);
        }
        else if ( a3 )
        {
          return (unsigned int)(v13 + 940);
        }
        else
        {
          return (unsigned int)(v13 + 939);
        }
      case 3LL:
        v10 = a2 == 0 ? 2 : 0;
        if ( a5 )
        {
          if ( a3 )
            return (unsigned int)(v10 + 916);
          else
            return (unsigned int)(v10 + 915);
        }
        else if ( a3 )
        {
          return (unsigned int)(v10 + 948);
        }
        else
        {
          return (unsigned int)(v10 + 947);
        }
      case 4LL:
        v11 = a2 == 0 ? 2 : 0;
        if ( a5 )
        {
          if ( a3 )
            return (unsigned int)(v11 + 924);
          else
            return (unsigned int)(v11 + 923);
        }
        else if ( a3 )
        {
          return (unsigned int)(v11 + 956);
        }
        else
        {
          return (unsigned int)(v11 + 955);
        }
      case 5LL:
        v9 = a2 == 0 ? 2 : 0;
        if ( a5 )
        {
          if ( a3 )
            return (unsigned int)(v9 + 932);
          else
            return (unsigned int)(v9 + 931);
        }
        else if ( a3 )
        {
          return (unsigned int)(v9 + 964);
        }
        else
        {
          return (unsigned int)(v9 + 963);
        }
      default:
        goto LABEL_62;
    }
  }
  switch ( a1 )
  {
    case 4LL:
      v8 = a2 == 0 ? 0xFFFFFFFE : 0;
      if ( a5 )
      {
        if ( a3 )
          return v8 + 922;
        else
          return v8 + 921;
      }
      else if ( a3 )
      {
        return v8 + 954;
      }
      else
      {
        return v8 + 953;
      }
    case 5LL:
      v5 = a2 == 0 ? 0xFFFFFFFE : 0;
      if ( a5 )
      {
        if ( a3 )
          return v5 + 930;
        else
          return v5 + 929;
      }
      else if ( a3 )
      {
        return v5 + 962;
      }
      else
      {
        return v5 + 961;
      }
    case 3LL:
      v7 = a2 == 0 ? 0xFFFFFFFE : 0;
      if ( a5 )
      {
        if ( a3 )
          return v7 + 914;
        else
          return v7 + 913;
      }
      else if ( a3 )
      {
        return v7 + 946;
      }
      else
      {
        return v7 + 945;
      }
    default:
LABEL_62:
      BUG();
  }
}
