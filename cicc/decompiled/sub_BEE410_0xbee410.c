// Function: sub_BEE410
// Address: 0xbee410
//
unsigned __int64 __fastcall sub_BEE410(__int64 a1, __int64 a2, const char **a3, _BYTE **a4)
{
  __int64 v4; // r12
  _BYTE *v8; // rax
  __int64 v9; // rsi
  unsigned __int64 result; // rax
  __int64 v11; // rdi
  _BYTE *v12; // rdi
  _BYTE *v13; // rsi
  __int64 v14; // rdi

  v4 = *(_QWORD *)a1;
  if ( !*(_QWORD *)a1 )
  {
    *(_BYTE *)(a1 + 153) = 1;
    result = *(unsigned __int8 *)(a1 + 154);
    *(_BYTE *)(a1 + 152) |= result;
    return result;
  }
  sub_CA0E80(a2, v4);
  v8 = *(_BYTE **)(v4 + 32);
  if ( (unsigned __int64)v8 >= *(_QWORD *)(v4 + 24) )
  {
    sub_CB5D20(v4, 10);
  }
  else
  {
    *(_QWORD *)(v4 + 32) = v8 + 1;
    *v8 = 10;
  }
  v9 = *(_QWORD *)a1;
  result = *(unsigned __int8 *)(a1 + 154);
  *(_BYTE *)(a1 + 153) = 1;
  *(_BYTE *)(a1 + 152) |= result;
  if ( v9 )
  {
    if ( *a3 )
    {
      sub_A62C00(*a3, v9, a1 + 16, *(_QWORD *)(a1 + 8));
      v11 = *(_QWORD *)a1;
      result = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
      if ( result >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
      {
        result = sub_CB5D20(v11, 10);
      }
      else
      {
        *(_QWORD *)(v11 + 32) = result + 1;
        *(_BYTE *)result = 10;
      }
    }
    v12 = *a4;
    if ( *a4 )
    {
      v13 = *(_BYTE **)a1;
      if ( *v12 > 0x1Cu )
      {
        sub_A693B0((__int64)v12, v13, a1 + 16, 0);
        v14 = *(_QWORD *)a1;
        result = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
        if ( result < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
          goto LABEL_11;
      }
      else
      {
        sub_A5C020(v12, (__int64)v13, 1, a1 + 16);
        v14 = *(_QWORD *)a1;
        result = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
        if ( result < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
        {
LABEL_11:
          *(_QWORD *)(v14 + 32) = result + 1;
          *(_BYTE *)result = 10;
          return result;
        }
      }
      return sub_CB5D20(v14, 10);
    }
  }
  return result;
}
