// Function: sub_733920
// Address: 0x733920
//
__int64 __fastcall sub_733920(__int64 a1)
{
  __int64 result; // rax
  __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx
  char v5; // dl
  __int64 i; // rax
  __int64 v7; // rax
  __int64 v8; // rcx
  char v9; // dl
  _BYTE *v10; // rax

  result = 0;
  if ( !*(_QWORD *)(a1 + 24) )
  {
    switch ( *(_BYTE *)a1 )
    {
      case 0:
        result = 0;
        if ( !*(_QWORD *)(a1 + 48) )
          return ((*(_BYTE *)(a1 + 1) >> 1) ^ 1) & 1;
        return result;
      case 1:
        v5 = *(_BYTE *)(a1 + 8);
        if ( v5 == 23 )
        {
          v8 = *(_QWORD *)(a1 + 16);
          v9 = *(_BYTE *)(v8 + 28);
          if ( v9 == 2 )
          {
            result = 0;
            if ( *(_QWORD *)(v8 + 32) )
              return result;
          }
          v10 = *(_BYTE **)(a1 + 32);
          if ( v10 )
          {
            if ( *v10 == 4 )
            {
              result = 0;
              if ( (*(_BYTE *)(a1 + 1) & 4) != 0 )
                return result;
            }
          }
          result = 1;
          if ( !*(_QWORD *)(a1 + 48) )
            return result;
          result = 0;
          if ( v9 == 17 )
            return result;
        }
        else
        {
          if ( dword_4D048B8 )
          {
            result = 0;
            if ( v5 == 31 )
              return result;
          }
          if ( v5 == 20 )
          {
            if ( *(_QWORD *)(a1 + 48) )
            {
              if ( dword_4F04C58 == dword_4F04C64 )
              {
                result = 0;
                if ( *(char *)(*(_QWORD *)(qword_4F04C50 + 32LL) + 207LL) < 0 )
                  return result;
              }
            }
          }
        }
        result = 0;
        if ( (*(_BYTE *)(a1 + 1) & 1) != 0 )
          return result;
        v4 = *(_QWORD *)(a1 + 48);
        if ( v4 )
        {
          while ( 1 )
          {
            for ( i = *(_QWORD *)(v4 + 24); i; i = *(_QWORD *)(i + 32) )
            {
              if ( (*(_BYTE *)(i + 49) & 0x10) != 0 )
                return 0;
            }
            v4 = *(_QWORD *)(v4 + 56);
            if ( !v4 )
              return 1;
          }
        }
        return 1;
      case 2:
        v2 = *(_QWORD *)(a1 + 48);
        if ( !v2 )
          goto LABEL_26;
        break;
      case 4:
        return 1;
      case 5:
        return 0;
      default:
        sub_721090();
    }
    while ( 1 )
    {
      v3 = *(_QWORD *)(v2 + 24);
      if ( v3 )
        break;
LABEL_25:
      v2 = *(_QWORD *)(v2 + 56);
      if ( !v2 )
      {
LABEL_26:
        if ( !unk_4D044B4 )
          return 1;
        v7 = *(_QWORD *)(*(_QWORD *)(a1 + 32) + 24LL);
        if ( !v7 )
          return 1;
        while ( (*(_BYTE *)(v7 + 49) & 0x10) == 0 )
        {
          v7 = *(_QWORD *)(v7 + 32);
          if ( !v7 )
            return 1;
        }
        return 0;
      }
    }
    while ( (*(_BYTE *)(v3 + 49) & 0x10) == 0 )
    {
      v3 = *(_QWORD *)(v3 + 32);
      if ( !v3 )
        goto LABEL_25;
    }
    return 0;
  }
  return result;
}
