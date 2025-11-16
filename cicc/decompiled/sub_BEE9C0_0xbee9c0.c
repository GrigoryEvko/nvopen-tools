// Function: sub_BEE9C0
// Address: 0xbee9c0
//
unsigned __int64 __fastcall sub_BEE9C0(_BYTE *a1, __int64 a2, _BYTE *a3)
{
  __int64 v4; // r12
  _BYTE *v6; // rax
  _BYTE *v7; // rsi
  unsigned __int64 result; // rax
  __int64 v9; // rdi

  v4 = *(_QWORD *)a1;
  if ( !*(_QWORD *)a1 )
  {
    a1[153] = 1;
    result = (unsigned __int8)a1[154];
    a1[152] |= result;
    return result;
  }
  sub_CA0E80(a2, v4);
  v6 = *(_BYTE **)(v4 + 32);
  if ( (unsigned __int64)v6 >= *(_QWORD *)(v4 + 24) )
  {
    sub_CB5D20(v4, 10);
  }
  else
  {
    *(_QWORD *)(v4 + 32) = v6 + 1;
    *v6 = 10;
  }
  v7 = *(_BYTE **)a1;
  result = (unsigned __int8)a1[154];
  a1[153] = 1;
  a1[152] |= result;
  if ( v7 )
  {
    if ( *a3 <= 0x1Cu )
    {
      sub_A5C020(a3, (__int64)v7, 1, (__int64)(a1 + 16));
      v9 = *(_QWORD *)a1;
      result = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
      if ( result < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
        goto LABEL_7;
    }
    else
    {
      sub_A693B0((__int64)a3, v7, (__int64)(a1 + 16), 0);
      v9 = *(_QWORD *)a1;
      result = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
      if ( result < *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
      {
LABEL_7:
        *(_QWORD *)(v9 + 32) = result + 1;
        *(_BYTE *)result = 10;
        return result;
      }
    }
    return sub_CB5D20(v9, 10);
  }
  return result;
}
