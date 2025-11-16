// Function: sub_15FBEB0
// Address: 0x15fbeb0
//
__int64 __fastcall sub_15FBEB0(_QWORD *a1, char a2, __int64 a3, char a4)
{
  __int64 v4; // r12
  __int64 v5; // rbx
  char v6; // r15
  __int64 result; // rax
  unsigned int v8; // r14d
  unsigned int v9; // eax
  unsigned int v10; // edx
  char v11; // al
  char v12; // al

  v4 = *a1;
  if ( a3 == *a1 )
    return 47;
  v5 = a3;
  if ( *(_BYTE *)(v4 + 8) == 16 )
  {
    v6 = *(_BYTE *)(a3 + 8);
    if ( v6 != 16 )
    {
      v8 = sub_1643030(*a1);
      v10 = sub_1643030(v5);
      result = 47;
      if ( v6 == 11 )
        return result;
      if ( (unsigned __int8)(v6 - 1) <= 5u )
      {
        v11 = 16;
        goto LABEL_19;
      }
      goto LABEL_12;
    }
    result = 47;
    if ( *(_QWORD *)(v4 + 32) != *(_QWORD *)(a3 + 32) )
      return result;
    v4 = *(_QWORD *)(v4 + 24);
    v5 = *(_QWORD *)(a3 + 24);
  }
  v8 = sub_1643030(v4);
  v9 = sub_1643030(v5);
  v6 = *(_BYTE *)(v5 + 8);
  v10 = v9;
  if ( v6 != 11 )
  {
    if ( (unsigned __int8)(v6 - 1) <= 5u )
    {
      v11 = *(_BYTE *)(v4 + 8);
      if ( v11 == 11 )
        return 41 - ((unsigned int)(a2 == 0) - 1);
LABEL_19:
      if ( (unsigned __int8)(v11 - 1) <= 5u )
      {
        result = 43;
        if ( v10 >= v8 )
          return 3 * (unsigned int)(v10 <= v8) + 44;
        return result;
      }
      return 47;
    }
    if ( v6 == 16 )
      return 47;
LABEL_12:
    result = 47;
    if ( v6 == 15 )
    {
      result = 46;
      if ( *(_BYTE *)(v4 + 8) == 15 )
        return (unsigned int)(*(_DWORD *)(v4 + 8) >> 8 != *(_DWORD *)(v5 + 8) >> 8) + 47;
    }
    return result;
  }
  v12 = *(_BYTE *)(v4 + 8);
  if ( v12 == 11 )
  {
    result = 36;
    if ( v8 <= v10 )
    {
      result = 47;
      if ( v8 < v10 )
        return 37 - ((unsigned int)(a2 == 0) - 1);
    }
  }
  else if ( (unsigned __int8)(v12 - 1) > 5u )
  {
    return 2 * (unsigned int)(v12 == 16) + 45;
  }
  else
  {
    return 39 - ((unsigned int)(a4 == 0) - 1);
  }
  return result;
}
