// Function: sub_BEE0A0
// Address: 0xbee0a0
//
unsigned __int64 __fastcall sub_BEE0A0(_BYTE *a1, __int64 a2, _BYTE **a3)
{
  __int64 v4; // r12
  _BYTE *v6; // rax
  _BYTE *v7; // rsi
  unsigned __int64 result; // rax
  _BYTE *v9; // rdi
  __int64 v10; // rdi

  v4 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
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
      v9 = *a3;
      if ( *a3 )
      {
        if ( *v9 <= 0x1Cu )
        {
          sub_A5C020(v9, (__int64)v7, 1, (__int64)(a1 + 16));
          v10 = *(_QWORD *)a1;
          result = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
          if ( result >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
            return sub_CB5D20(v10, 10);
        }
        else
        {
          sub_A693B0((__int64)v9, v7, (__int64)(a1 + 16), 0);
          v10 = *(_QWORD *)a1;
          result = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
          if ( result >= *(_QWORD *)(*(_QWORD *)a1 + 24LL) )
            return sub_CB5D20(v10, 10);
        }
        *(_QWORD *)(v10 + 32) = result + 1;
        *(_BYTE *)result = 10;
      }
    }
  }
  else
  {
    result = (unsigned __int8)a1[154];
    a1[152] |= result;
    a1[153] = 1;
  }
  return result;
}
