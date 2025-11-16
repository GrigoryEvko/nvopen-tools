// Function: sub_21CAB70
// Address: 0x21cab70
//
bool __fastcall sub_21CAB70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  bool result; // al
  char v7; // r12
  unsigned __int8 v8; // cl
  unsigned __int8 v9; // dl
  unsigned int v10; // r8d
  __int64 v11; // rsi
  __int64 v12; // r9
  __int64 v13; // r8
  unsigned int v14; // edi
  _QWORD v15[2]; // [rsp+0h] [rbp-40h] BYREF
  _QWORD v16[6]; // [rsp+10h] [rbp-30h] BYREF

  result = 0;
  v16[0] = a2;
  v16[1] = a3;
  v15[0] = a4;
  v15[1] = a5;
  if ( (_BYTE)a4 == 7 )
    return result;
  v7 = a4;
  if ( (_BYTE)a4 )
  {
    v8 = v16[0];
    v9 = v16[0];
    if ( (unsigned __int8)(v7 - 14) > 0x5Fu || word_435D740[(unsigned __int8)(v7 - 14)] <= 4u )
      goto LABEL_8;
  }
  else
  {
    if ( !sub_1F58D20((__int64)v15) || (unsigned int)sub_1F58D30((__int64)v15) <= 4 )
      return 1;
    v8 = v16[0];
  }
  v9 = v8;
  if ( !v8 )
  {
    result = sub_1F58D20((__int64)v16);
    if ( !result )
      return result;
    v10 = sub_1F58D30((__int64)v16);
    result = 0;
    if ( v10 <= 4 )
      return result;
    return 1;
  }
  if ( (unsigned __int8)(v8 - 14) > 0x5Fu )
    return 0;
  result = 0;
  if ( word_435D740[(unsigned __int8)(v8 - 14)] > 4u )
  {
LABEL_8:
    result = v8 == 0 || v7 == 0;
    if ( !result && *(_BYTE *)(a1 + 259LL * v8 + 2607) == 1 )
    {
      v11 = *(_QWORD *)(a1 + 74064);
      v12 = a1 + 74056;
      if ( !v11 )
        goto LABEL_28;
      v13 = a1 + 74056;
      do
      {
        v14 = *(_DWORD *)(v11 + 32);
        if ( v14 <= 0xB8 || v14 == 185 && v8 > *(_BYTE *)(v11 + 36) )
        {
          v11 = *(_QWORD *)(v11 + 24);
        }
        else
        {
          v13 = v11;
          v11 = *(_QWORD *)(v11 + 16);
        }
      }
      while ( v11 );
      if ( v12 == v13 || *(_DWORD *)(v13 + 32) > 0xB9u || *(_DWORD *)(v13 + 32) == 185 && v8 < *(_BYTE *)(v13 + 36) )
      {
LABEL_28:
        do
        {
          do
            ++v9;
          while ( !v9 );
        }
        while ( !*(_QWORD *)(a1 + 8LL * v9 + 120) || *(_BYTE *)(a1 + 259LL * v9 + 2607) == 1 );
      }
      else
      {
        v9 = *(_BYTE *)(v13 + 40);
      }
      if ( v7 == v9 )
        return result;
    }
    return 1;
  }
  return result;
}
