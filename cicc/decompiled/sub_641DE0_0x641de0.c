// Function: sub_641DE0
// Address: 0x641de0
//
__int64 __fastcall sub_641DE0(__int64 a1, int a2, int a3, int a4, _DWORD *a5)
{
  __int64 result; // rax
  char v8; // dl
  char v9; // dl

  result = 3;
  if ( (*(_BYTE *)(a1 + 193) & 0x10) == 0
    && ((*(_QWORD *)(a1 + 200) & 0x8000001000000LL) != 0x8000000000000LL || (*(_BYTE *)(a1 + 192) & 2) != 0) )
  {
    result = 3;
    if ( (*(_BYTE *)(a1 + 199) & 1) == 0 && (*(_BYTE *)(a1 - 8) & 0x10) == 0 )
    {
      v8 = *(_BYTE *)(a1 + 198);
      if ( (v8 & 0x20) != 0 )
      {
        if ( a4 )
        {
          if ( a3 )
          {
            *a5 = 3498;
            return 8;
          }
          else if ( !a2 )
          {
            *a5 = 3491;
            return 8;
          }
        }
        else
        {
          *a5 = 3495;
          return 8;
        }
      }
      else if ( (v8 & 0x10) != 0 )
      {
        v9 = v8 & 0x18;
        if ( v9 != 16 )
        {
          if ( v9 != 24 )
            return result;
          if ( a2 )
          {
            *a5 = 3490;
            result = 8;
          }
          else if ( (a4 & (a3 ^ 1)) != 0 )
          {
            *a5 = 3493;
            result = 5;
          }
          else if ( !a4 )
          {
            *a5 = 3497;
            result = 5;
          }
LABEL_20:
          *(_BYTE *)(a1 + 198) |= 0x18u;
          return result;
        }
        if ( a2 )
        {
          *a5 = 3489;
          return 8;
        }
        if ( !a4 )
        {
          *a5 = 3496;
          result = 5;
          goto LABEL_20;
        }
      }
      else if ( (*(_BYTE *)(a1 + 198) & 8) != 0 )
      {
        if ( a2 )
        {
          *a5 = 3488;
          return 8;
        }
        if ( (a4 & (a3 ^ 1)) != 0 )
        {
          *a5 = 3492;
          result = 5;
          goto LABEL_20;
        }
      }
      else
      {
        if ( a2 )
        {
          *a5 = 3487;
          return 8;
        }
        if ( (a4 & (a3 ^ 1)) != 0 )
        {
          *a5 = 3494;
          result = 5;
          goto LABEL_20;
        }
      }
    }
  }
  return result;
}
