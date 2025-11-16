// Function: sub_1378050
// Address: 0x1378050
//
_QWORD *__fastcall sub_1378050(__int64 a1, __int64 a2)
{
  __int64 v4; // r14
  _QWORD *result; // rax
  __int64 v6; // rdi
  __int64 v7; // r15
  unsigned int v8; // r14d
  __int64 v9; // rax
  _QWORD *v10; // rsi
  unsigned int v11; // edi
  _QWORD *v12; // rcx
  int v13; // [rsp+Ch] [rbp-34h]

  v4 = sub_157EBA0(a2);
  if ( !(unsigned int)sub_15F4D60(v4) )
  {
    if ( *(_BYTE *)(v4 + 16) != 31 )
    {
      result = (_QWORD *)sub_157ECB0(a2);
      if ( !result )
        return result;
    }
    result = *(_QWORD **)(a1 + 80);
    if ( *(_QWORD **)(a1 + 88) == result )
    {
      v10 = &result[*(unsigned int *)(a1 + 100)];
      v11 = *(_DWORD *)(a1 + 100);
      if ( result != v10 )
      {
        v12 = 0;
        while ( a2 != *result )
        {
          if ( *result == -2 )
            v12 = result;
          if ( v10 == ++result )
          {
            if ( !v12 )
              goto LABEL_42;
LABEL_24:
            *v12 = a2;
            --*(_DWORD *)(a1 + 104);
            ++*(_QWORD *)(a1 + 72);
            return result;
          }
        }
        return result;
      }
LABEL_42:
      if ( v11 < *(_DWORD *)(a1 + 96) )
      {
LABEL_43:
        *(_DWORD *)(a1 + 100) = v11 + 1;
        *v10 = a2;
        ++*(_QWORD *)(a1 + 72);
        return result;
      }
    }
    return (_QWORD *)sub_16CCBA0(a1 + 72, a2);
  }
  if ( *(_BYTE *)(v4 + 16) == 29 )
  {
    result = (_QWORD *)sub_1377F70(a1 + 72, *(_QWORD *)(v4 - 48));
    if ( !(_DWORD)result )
      return result;
    result = *(_QWORD **)(a1 + 80);
    if ( *(_QWORD **)(a1 + 88) != result )
      return (_QWORD *)sub_16CCBA0(a1 + 72, a2);
    v10 = &result[*(unsigned int *)(a1 + 100)];
    v11 = *(_DWORD *)(a1 + 100);
    if ( result != v10 )
    {
      v12 = 0;
      while ( a2 != *result )
      {
        if ( *result == -2 )
          v12 = result;
        if ( v10 == ++result )
          goto LABEL_31;
      }
      return result;
    }
LABEL_32:
    if ( v11 < *(_DWORD *)(a1 + 96) )
      goto LABEL_43;
    return (_QWORD *)sub_16CCBA0(a1 + 72, a2);
  }
  v6 = sub_157EBA0(a2);
  if ( v6 && (v13 = sub_15F4D60(v6), v7 = sub_157EBA0(a2), v13) )
  {
    v8 = 0;
    while ( 1 )
    {
      v9 = sub_15F4DF0(v7, v8);
      result = (_QWORD *)sub_1377F70(a1 + 72, v9);
      if ( !(_DWORD)result )
        break;
      if ( ++v8 == v13 )
        goto LABEL_10;
    }
  }
  else
  {
LABEL_10:
    result = *(_QWORD **)(a1 + 80);
    if ( *(_QWORD **)(a1 + 88) != result )
      return (_QWORD *)sub_16CCBA0(a1 + 72, a2);
    v10 = &result[*(unsigned int *)(a1 + 100)];
    v11 = *(_DWORD *)(a1 + 100);
    if ( result == v10 )
      goto LABEL_32;
    v12 = 0;
    while ( a2 != *result )
    {
      if ( *result == -2 )
        v12 = result;
      if ( v10 == ++result )
      {
LABEL_31:
        if ( v12 )
          goto LABEL_24;
        goto LABEL_32;
      }
    }
  }
  return result;
}
