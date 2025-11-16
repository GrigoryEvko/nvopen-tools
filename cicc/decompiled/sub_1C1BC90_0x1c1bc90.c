// Function: sub_1C1BC90
// Address: 0x1c1bc90
//
char *__fastcall sub_1C1BC90(__int64 a1, __int16 a2, unsigned int a3)
{
  char *result; // rax
  char *v5; // rsi
  unsigned int v6; // [rsp+18h] [rbp-18h] BYREF
  unsigned int v7[3]; // [rsp+1Ch] [rbp-14h] BYREF

  LOWORD(v6) = a2;
  result = *(char **)(a1 + 16);
  v5 = *(char **)(a1 + 8);
  if ( a3 > 0xFFFE )
  {
    HIWORD(v6) = -1;
    if ( v5 == result )
    {
      result = sub_1C1BB00(a1, v5, &v6);
      v7[0] = a3;
      v5 = *(char **)(a1 + 8);
      if ( v5 != *(char **)(a1 + 16) )
      {
        if ( !v5 )
          goto LABEL_10;
        goto LABEL_9;
      }
    }
    else
    {
      if ( v5 )
      {
        *(_DWORD *)v5 = v6;
        v5 = *(char **)(a1 + 8);
        result = *(char **)(a1 + 16);
      }
      v5 += 4;
      v7[0] = a3;
      *(_QWORD *)(a1 + 8) = v5;
      if ( v5 != result )
      {
LABEL_9:
        result = (char *)v7[0];
        *(_DWORD *)v5 = v7[0];
        v5 = *(char **)(a1 + 8);
LABEL_10:
        *(_QWORD *)(a1 + 8) = v5 + 4;
        return result;
      }
    }
    return sub_1C1BB00(a1, v5, v7);
  }
  HIWORD(v6) = a3;
  if ( v5 != result )
  {
    if ( v5 )
    {
      result = (char *)v6;
      *(_DWORD *)v5 = v6;
      v5 = *(char **)(a1 + 8);
    }
    goto LABEL_10;
  }
  return sub_1C1BB00(a1, v5, &v6);
}
