// Function: sub_B50D10
// Address: 0xb50d10
//
__int64 __fastcall sub_B50D10(__int64 a1, char a2, __int64 a3, char a4)
{
  __int64 v4; // r12
  __int64 v5; // rbx
  int v7; // edx
  int v8; // ecx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  unsigned int v13; // edx
  unsigned __int8 v14; // al
  unsigned __int8 v15; // al
  __int64 result; // rax
  char v17; // al
  unsigned __int8 v18; // al
  unsigned int v19; // [rsp+8h] [rbp-48h]
  __int64 v20; // [rsp+10h] [rbp-40h] BYREF
  __int64 v21; // [rsp+18h] [rbp-38h]

  v4 = *(_QWORD *)(a1 + 8);
  if ( a3 == v4 )
    return 49;
  v5 = a3;
  v7 = *(unsigned __int8 *)(v4 + 8);
  if ( (unsigned int)(v7 - 17) <= 1 )
  {
    v8 = *(unsigned __int8 *)(v5 + 8);
    if ( (unsigned int)(v8 - 17) <= 1
      && ((_BYTE)v8 == 18) == ((_BYTE)v7 == 18)
      && *(_DWORD *)(v5 + 32) == *(_DWORD *)(v4 + 32) )
    {
      v4 = *(_QWORD *)(v4 + 24);
      v5 = *(_QWORD *)(v5 + 24);
    }
  }
  v9 = sub_BCAE30(v4);
  v21 = v10;
  v20 = v9;
  v19 = sub_CA1930(&v20);
  v11 = sub_BCAE30(v5);
  v21 = v12;
  v20 = v11;
  v13 = sub_CA1930(&v20);
  v14 = *(_BYTE *)(v5 + 8);
  if ( v14 == 12 )
  {
    v18 = *(_BYTE *)(v4 + 8);
    if ( v18 != 12 )
    {
      if ( v18 <= 3u || v18 == 5 || (v18 & 0xFD) == 4 )
        return 41 - ((unsigned int)(a4 == 0) - 1);
      if ( (unsigned __int8)(v18 - 17) > 1u )
        return 47;
      return 49;
    }
    result = 38;
    if ( v19 <= v13 )
    {
      if ( v19 < v13 )
        return 39 - ((unsigned int)(a2 == 0) - 1);
      return 49;
    }
    return result;
  }
  if ( v14 <= 3u || v14 == 5 || (v14 & 0xFD) == 4 )
  {
    v15 = *(_BYTE *)(v4 + 8);
    if ( v15 == 12 )
      return 43 - ((unsigned int)(a2 == 0) - 1);
    if ( v15 <= 3u || v15 == 5 || (v15 & 0xFD) == 4 )
    {
      result = 45;
      if ( v19 <= v13 )
      {
        result = 46;
        if ( v19 >= v13 )
          return 49;
      }
      return result;
    }
    if ( (unsigned __int8)(v15 - 17) > 1u )
LABEL_39:
      BUG();
    return 49;
  }
  if ( (unsigned __int8)(v14 - 17) <= 1u )
    return 49;
  if ( v14 != 14 )
    goto LABEL_39;
  v17 = *(_BYTE *)(v4 + 8);
  if ( v17 != 14 )
  {
    if ( v17 == 12 )
      return 48;
    goto LABEL_39;
  }
  result = 50;
  if ( *(_DWORD *)(v4 + 8) >> 8 == *(_DWORD *)(v5 + 8) >> 8 )
    return 49;
  return result;
}
