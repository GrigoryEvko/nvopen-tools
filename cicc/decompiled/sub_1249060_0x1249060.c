// Function: sub_1249060
// Address: 0x1249060
//
__int64 __fastcall sub_1249060(__int64 a1)
{
  unsigned int v1; // eax
  bool v2; // cc
  unsigned __int64 v4; // rsi
  int i; // eax
  const char *v6; // [rsp+0h] [rbp-40h] BYREF
  char v7; // [rsp+20h] [rbp-20h]
  char v8; // [rsp+21h] [rbp-1Fh]

  if ( *(_QWORD *)(a1 + 344) )
  {
    while ( 1 )
    {
LABEL_2:
      while ( 1 )
      {
        v1 = *(_DWORD *)(a1 + 240);
        v2 = v1 <= 0x198;
        if ( v1 != 408 )
          break;
LABEL_10:
        if ( (unsigned __int8)sub_12407A0(a1) )
          return 1;
      }
      while ( 1 )
      {
        if ( !v2 )
        {
          switch ( v1 )
          {
            case 0x1F7u:
              if ( (unsigned __int8)sub_1245520(a1) )
                return 1;
              goto LABEL_2;
            case 0x1F8u:
              if ( (unsigned __int8)sub_121AB80(a1) )
                return 1;
              goto LABEL_2;
            case 0x1FAu:
              if ( (unsigned __int8)sub_1243920(a1) )
                return 1;
              goto LABEL_2;
            case 0x1FCu:
              if ( (unsigned __int8)sub_12456E0(a1) )
                return 1;
              goto LABEL_2;
            case 0x1FDu:
              if ( (unsigned __int8)sub_1216BD0(a1) )
                return 1;
              goto LABEL_2;
            case 0x1FEu:
              if ( (unsigned __int8)sub_121AD70(a1) )
                return 1;
              goto LABEL_2;
            case 0x1FFu:
              if ( (unsigned __int8)sub_121B840(a1) )
                return 1;
              goto LABEL_2;
            default:
              goto LABEL_36;
          }
        }
        if ( v1 == 23 )
        {
          if ( (unsigned __int8)sub_1248FB0(a1) )
            return 1;
          goto LABEL_2;
        }
        if ( v1 <= 0x17 )
          break;
        if ( v1 == 163 )
        {
          if ( (unsigned __int8)sub_1218310(a1) )
            return 1;
          goto LABEL_2;
        }
        if ( v1 != 407 )
        {
          if ( v1 == 100 )
          {
            if ( !(unsigned __int8)sub_120B9D0(a1) )
              goto LABEL_2;
            return 1;
          }
LABEL_36:
          v4 = *(_QWORD *)(a1 + 232);
          v8 = 1;
          v6 = "expected top-level entity";
          v7 = 3;
          sub_11FD800(a1 + 176, v4, (__int64)&v6, 1);
          return 1;
        }
        if ( (unsigned __int8)sub_1240EC0(a1, 0) )
          return 1;
        v1 = *(_DWORD *)(a1 + 240);
        v2 = v1 <= 0x198;
        if ( v1 == 408 )
          goto LABEL_10;
      }
      if ( v1 != 14 )
        break;
      if ( (unsigned __int8)sub_123F610(a1) )
        return 1;
    }
    if ( v1 != 22 )
    {
      if ( !v1 )
        return 0;
      goto LABEL_36;
    }
    if ( !(unsigned __int8)sub_1247060(a1) )
      goto LABEL_2;
    return 1;
  }
  else
  {
    for ( i = *(_DWORD *)(a1 + 240); i == 65; i = *(_DWORD *)(a1 + 240) )
    {
LABEL_49:
      if ( (unsigned __int8)sub_120B790(a1) )
        return 1;
LABEL_50:
      ;
    }
    while ( 1 )
    {
      if ( i == 506 )
      {
        if ( (unsigned __int8)sub_1243920(a1) )
          return 1;
        goto LABEL_50;
      }
      if ( !i )
        return 0;
      i = sub_1205200(a1 + 176);
      *(_DWORD *)(a1 + 240) = i;
      if ( i == 65 )
        goto LABEL_49;
    }
  }
}
