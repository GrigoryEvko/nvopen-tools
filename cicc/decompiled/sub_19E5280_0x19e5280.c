// Function: sub_19E5280
// Address: 0x19e5280
//
bool __fastcall sub_19E5280(__int64 a1, unsigned __int64 a2, unsigned __int64 a3)
{
  unsigned __int8 v4; // al
  unsigned __int8 v5; // r13
  unsigned int v6; // r14d
  unsigned int v7; // edx
  bool result; // al
  int v9; // eax
  int v10; // eax

  v4 = *(_BYTE *)(a3 + 16);
  v5 = *(_BYTE *)(a2 + 16);
  if ( v4 == 5 )
  {
    if ( v5 == 5 )
      return a3 < a2;
    v6 = 2;
LABEL_25:
    result = 0;
    if ( v5 <= 0x10u )
      return result;
LABEL_9:
    if ( v5 == 17 )
    {
      v7 = *(_DWORD *)(a2 + 32) + 3;
    }
    else
    {
      v9 = sub_19E5210(a1 + 2392, a2);
      if ( !v9 )
      {
        result = 1;
        if ( v6 == -1 )
          return a3 < a2;
        return result;
      }
      v7 = v9 + *(_DWORD *)(a1 + 1392) + 4;
    }
    goto LABEL_11;
  }
  if ( v4 == 9 )
  {
    v6 = 1;
LABEL_14:
    if ( v5 == 5 )
      return 1;
    goto LABEL_7;
  }
  if ( v4 <= 0x10u )
  {
    v6 = 0;
    goto LABEL_14;
  }
  if ( v4 == 17 )
  {
    v6 = *(_DWORD *)(a3 + 32) + 3;
    goto LABEL_6;
  }
  v10 = sub_19E5210(a1 + 2392, a3);
  if ( !v10 )
  {
    if ( v5 == 5 )
      return 0;
    v6 = -1;
    goto LABEL_25;
  }
  v6 = v10 + *(_DWORD *)(a1 + 1392) + 4;
LABEL_6:
  if ( v5 == 5 )
  {
    v7 = 2;
    goto LABEL_11;
  }
LABEL_7:
  if ( v5 != 9 )
  {
    if ( v5 <= 0x10u )
    {
      v7 = 0;
      goto LABEL_17;
    }
    goto LABEL_9;
  }
  v7 = 1;
LABEL_11:
  result = 1;
  if ( v7 > v6 )
    return result;
LABEL_17:
  result = 0;
  if ( v7 >= v6 )
    return a3 < a2;
  return result;
}
