// Function: sub_11D9B10
// Address: 0x11d9b10
//
__int64 __fastcall sub_11D9B10(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  __int64 result; // rax
  __int64 v8; // r15
  __int64 v10; // r13
  __int64 i; // r11
  __int64 v12; // rsi
  _BYTE *v13; // rdx
  char v14; // cl
  __int64 v15; // rcx
  char v16; // si

  result = a3 - 1;
  v8 = a3 & 1;
  v10 = (a3 - 1) / 2;
  if ( a2 >= v10 )
  {
    v13 = (_BYTE *)(a1 + a2);
    if ( v8 )
      goto LABEL_13;
    result = a2;
    goto LABEL_16;
  }
  for ( i = a2; ; i = result )
  {
    v12 = 2 * i + 2;
    result = 2 * i + 1;
    v13 = (_BYTE *)(a1 + result);
    v14 = *(_BYTE *)(a1 + result);
    if ( *(char *)(a1 + v12) >= v14 )
    {
      v14 = *(_BYTE *)(a1 + v12);
      v13 = (_BYTE *)(a1 + v12);
      result = 2 * i + 2;
    }
    *(_BYTE *)(a1 + i) = v14;
    if ( result >= v10 )
      break;
  }
  if ( !v8 )
  {
LABEL_16:
    if ( (a3 - 2) / 2 == result )
    {
      result = 2 * result + 1;
      *v13 = *(_BYTE *)(a1 + result);
      v13 = (_BYTE *)(a1 + result);
    }
  }
  v15 = (result - 1) / 2;
  if ( result > a2 )
  {
    while ( 1 )
    {
      v13 = (_BYTE *)(a1 + result);
      v16 = *(_BYTE *)(a1 + v15);
      if ( a4 <= v16 )
        break;
      *v13 = v16;
      result = v15;
      if ( a2 >= v15 )
      {
        *(_BYTE *)(a1 + v15) = a4;
        return result;
      }
      v15 = (v15 - 1) / 2;
    }
  }
LABEL_13:
  *v13 = a4;
  return result;
}
