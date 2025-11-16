// Function: sub_14A0F00
// Address: 0x14a0f00
//
__int64 __fastcall sub_14A0F00(__int64 a1, _DWORD *a2)
{
  int v2; // r8d
  __int64 result; // rax
  __int64 v4; // rdx
  __int64 v5; // r12
  unsigned __int8 v6; // dl
  __int64 v7; // rdx
  int v8; // r12d
  unsigned int v9; // r14d
  __int64 v10; // rax
  __int64 v11; // rax
  int v12; // r8d

  *a2 = 0;
  if ( *(_BYTE *)(a1 + 16) != 13 )
  {
    v5 = sub_14C49D0();
    if ( (*(_BYTE *)(a1 + 16) & 0xFB) == 8 )
    {
      if ( !v5 )
      {
        if ( (unsigned int)*(unsigned __int8 *)(a1 + 16) - 11 <= 1 )
        {
          *a2 = 1;
          v8 = sub_15958F0(a1);
          if ( v8 )
          {
            v9 = 0;
            while ( 1 )
            {
              v10 = sub_15A0940(a1, v9);
              if ( *(_BYTE *)(v10 + 16) != 13 )
                break;
              if ( *(_DWORD *)(v10 + 32) > 0x40u )
              {
                if ( (unsigned int)sub_16A5940(v10 + 24) != 1 )
                  break;
              }
              else
              {
                v11 = *(_QWORD *)(v10 + 24);
                if ( !v11 || (v11 & (v11 - 1)) != 0 )
                  break;
              }
              if ( v8 == ++v9 )
                return 3;
            }
            *a2 = 0;
          }
        }
        return 3;
      }
      v6 = *(_BYTE *)(v5 + 16);
      if ( v6 == 13 )
      {
        if ( *(_DWORD *)(v5 + 32) > 0x40u )
        {
          v12 = sub_16A5940(v5 + 24);
          result = 2;
          if ( v12 != 1 )
            return result;
        }
        else
        {
          v7 = *(_QWORD *)(v5 + 24);
          result = 2;
          if ( !v7 || (v7 & (v7 - 1)) != 0 )
            return result;
        }
        *a2 = 1;
        result = 2;
        v6 = *(_BYTE *)(v5 + 16);
      }
      else
      {
        result = 2;
      }
    }
    else
    {
      result = 0;
      if ( !v5 )
        return result;
      v6 = *(_BYTE *)(v5 + 16);
    }
    if ( v6 <= 3u || v6 == 17 )
      return 1;
    return result;
  }
  if ( *(_DWORD *)(a1 + 32) > 0x40u )
  {
    v2 = sub_16A5940(a1 + 24);
    result = 2;
    if ( v2 != 1 )
      return result;
    goto LABEL_4;
  }
  v4 = *(_QWORD *)(a1 + 24);
  result = 2;
  if ( v4 && (v4 & (v4 - 1)) == 0 )
  {
LABEL_4:
    *a2 = 1;
    return 2;
  }
  return result;
}
