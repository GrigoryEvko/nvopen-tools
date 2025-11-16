// Function: sub_28C8D50
// Address: 0x28c8d50
//
bool __fastcall sub_28C8D50(__int64 a1, unsigned __int8 *a2, __int64 a3)
{
  unsigned __int64 v3; // r10
  unsigned __int64 v4; // r11
  unsigned __int8 v5; // al
  unsigned __int8 v6; // bl
  unsigned int v7; // r13d
  unsigned int v8; // edx
  bool result; // al
  int v10; // eax
  int v11; // eax

  v3 = (unsigned __int64)a2;
  v4 = a3;
  v5 = *(_BYTE *)a3;
  v6 = *a2;
  if ( *(_BYTE *)a3 == 5 )
  {
    if ( v6 == 5 )
      return v4 < v3;
    v7 = 3;
LABEL_20:
    result = 0;
    if ( v6 == 13 || (unsigned int)v6 - 12 <= 1 || v6 <= 0x15u )
      return result;
    goto LABEL_23;
  }
  if ( v5 == 13 )
  {
    v7 = 1;
LABEL_4:
    if ( v6 == 5 )
      return 1;
    goto LABEL_5;
  }
  v7 = 2;
  if ( (unsigned int)v5 - 12 <= 1 )
    goto LABEL_4;
  if ( v5 <= 0x15u )
  {
    v7 = 0;
    goto LABEL_4;
  }
  if ( v5 == 22 )
  {
    v7 = *(_DWORD *)(a3 + 32) + 4;
    goto LABEL_12;
  }
  v11 = sub_28C8CE0(a1, a3);
  if ( !v11 )
  {
    if ( v6 == 5 )
      return 0;
    v7 = -1;
    goto LABEL_20;
  }
  v7 = v11 + *(_DWORD *)(a1 + 1352) + 5;
LABEL_12:
  if ( v6 != 5 )
  {
LABEL_5:
    if ( v6 == 13 )
    {
      v8 = 1;
      result = 1;
      if ( !v7 )
        return result;
      goto LABEL_16;
    }
    v8 = 2;
    if ( (unsigned int)v6 - 12 <= 1 )
      goto LABEL_7;
    if ( v6 <= 0x15u )
    {
      v8 = 0;
      goto LABEL_16;
    }
LABEL_23:
    if ( v6 == 22 )
    {
      v8 = *(_DWORD *)(v3 + 32) + 4;
      result = 1;
      if ( v8 > v7 )
        return result;
      goto LABEL_16;
    }
    v10 = sub_28C8CE0(a1, v3);
    if ( !v10 )
    {
      result = 1;
      if ( v7 == -1 )
        return v4 < v3;
      return result;
    }
    v8 = v10 + *(_DWORD *)(a1 + 1352) + 5;
    goto LABEL_7;
  }
  v8 = 3;
LABEL_7:
  result = 1;
  if ( v8 > v7 )
    return result;
LABEL_16:
  result = 0;
  if ( v8 >= v7 )
    return v4 < v3;
  return result;
}
