// Function: sub_DF7050
// Address: 0xdf7050
//
_QWORD *__fastcall sub_DF7050(_QWORD *a1, __int64 a2)
{
  _QWORD *result; // rax
  __int64 v4; // rcx
  __int64 v5; // rdx
  _QWORD *v6; // rcx
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rdx
  _BYTE *v10; // rdx
  __int64 v11; // rsi
  __int64 v12; // rsi
  __int64 v13; // rsi
  __int64 v14; // rsi
  _BYTE *v15; // rdx
  _BYTE *v16; // rdx
  _BYTE *v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // rcx
  __int64 v20; // rcx

  result = a1;
  v4 = (a2 - (__int64)a1) >> 5;
  v5 = (a2 - (__int64)a1) >> 3;
  if ( v4 <= 0 )
  {
LABEL_33:
    if ( v5 != 2 )
    {
      if ( v5 != 3 )
      {
        if ( v5 != 1 )
          return (_QWORD *)a2;
        goto LABEL_41;
      }
      v15 = (_BYTE *)*result;
      if ( *(_BYTE *)*result == 85 )
      {
        v19 = *((_QWORD *)v15 - 4);
        if ( v19 )
        {
          if ( !*(_BYTE *)v19
            && *(_QWORD *)(v19 + 24) == *((_QWORD *)v15 + 10)
            && (*(_BYTE *)(v19 + 33) & 0x20) != 0
            && *(_DWORD *)(v19 + 36) == 169 )
          {
            return result;
          }
        }
      }
      ++result;
    }
    v16 = (_BYTE *)*result;
    if ( *(_BYTE *)*result == 85 )
    {
      v20 = *((_QWORD *)v16 - 4);
      if ( v20 )
      {
        if ( !*(_BYTE *)v20
          && *(_QWORD *)(v20 + 24) == *((_QWORD *)v16 + 10)
          && (*(_BYTE *)(v20 + 33) & 0x20) != 0
          && *(_DWORD *)(v20 + 36) == 169 )
        {
          return result;
        }
      }
    }
    ++result;
LABEL_41:
    v17 = (_BYTE *)*result;
    if ( *(_BYTE *)*result == 85 )
    {
      v18 = *((_QWORD *)v17 - 4);
      if ( v18 )
      {
        if ( !*(_BYTE *)v18 && *(_QWORD *)(v18 + 24) == *((_QWORD *)v17 + 10) && (*(_BYTE *)(v18 + 33) & 0x20) != 0 )
        {
          if ( *(_DWORD *)(v18 + 36) != 169 )
            return (_QWORD *)a2;
          return result;
        }
      }
    }
    return (_QWORD *)a2;
  }
  v6 = &a1[4 * v4];
  while ( 1 )
  {
    v10 = (_BYTE *)*result;
    if ( *(_BYTE *)*result == 85 )
    {
      v11 = *((_QWORD *)v10 - 4);
      if ( v11 )
      {
        if ( !*(_BYTE *)v11
          && *(_QWORD *)(v11 + 24) == *((_QWORD *)v10 + 10)
          && (*(_BYTE *)(v11 + 33) & 0x20) != 0
          && *(_DWORD *)(v11 + 36) == 169 )
        {
          return result;
        }
      }
    }
    v7 = result[1];
    if ( *(_BYTE *)v7 == 85 )
    {
      v12 = *(_QWORD *)(v7 - 32);
      if ( v12 )
      {
        if ( !*(_BYTE *)v12
          && *(_QWORD *)(v12 + 24) == *(_QWORD *)(v7 + 80)
          && (*(_BYTE *)(v12 + 33) & 0x20) != 0
          && *(_DWORD *)(v12 + 36) == 169 )
        {
          return ++result;
        }
      }
    }
    v8 = result[2];
    if ( *(_BYTE *)v8 == 85 )
    {
      v13 = *(_QWORD *)(v8 - 32);
      if ( v13 )
      {
        if ( !*(_BYTE *)v13
          && *(_QWORD *)(v13 + 24) == *(_QWORD *)(v8 + 80)
          && (*(_BYTE *)(v13 + 33) & 0x20) != 0
          && *(_DWORD *)(v13 + 36) == 169 )
        {
          result += 2;
          return result;
        }
      }
    }
    v9 = result[3];
    if ( *(_BYTE *)v9 == 85 )
    {
      v14 = *(_QWORD *)(v9 - 32);
      if ( v14 )
      {
        if ( !*(_BYTE *)v14
          && *(_QWORD *)(v14 + 24) == *(_QWORD *)(v9 + 80)
          && (*(_BYTE *)(v14 + 33) & 0x20) != 0
          && *(_DWORD *)(v14 + 36) == 169 )
        {
          result += 3;
          return result;
        }
      }
    }
    result += 4;
    if ( result == v6 )
    {
      v5 = (a2 - (__int64)result) >> 3;
      goto LABEL_33;
    }
  }
}
