// Function: sub_BEE2B0
// Address: 0xbee2b0
//
unsigned __int64 __fastcall sub_BEE2B0(__int64 *a1, __int64 a2, _BYTE **a3, const char **a4)
{
  __int64 v4; // r12
  _BYTE *v8; // rax
  _BYTE *v9; // rsi
  unsigned __int64 result; // rax
  _BYTE *v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // rdi

  v4 = *a1;
  if ( !*a1 )
  {
    *((_BYTE *)a1 + 153) = 1;
    result = *((unsigned __int8 *)a1 + 154);
    *((_BYTE *)a1 + 152) |= result;
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
  v9 = (_BYTE *)*a1;
  result = *((unsigned __int8 *)a1 + 154);
  *((_BYTE *)a1 + 153) = 1;
  *((_BYTE *)a1 + 152) |= result;
  if ( v9 )
  {
    v11 = *a3;
    if ( !*a3 )
      goto LABEL_9;
    if ( *v11 > 0x1Cu )
    {
      sub_A693B0((__int64)v11, v9, (__int64)(a1 + 2), 0);
      v12 = *a1;
      result = *(_QWORD *)(*a1 + 32);
      if ( result < *(_QWORD *)(*a1 + 24) )
        goto LABEL_8;
    }
    else
    {
      sub_A5C020(v11, (__int64)v9, 1, (__int64)(a1 + 2));
      v12 = *a1;
      result = *(_QWORD *)(*a1 + 32);
      if ( result < *(_QWORD *)(*a1 + 24) )
      {
LABEL_8:
        *(_QWORD *)(v12 + 32) = result + 1;
        *(_BYTE *)result = 10;
        goto LABEL_9;
      }
    }
    result = sub_CB5D20(v12, 10);
LABEL_9:
    if ( *a4 )
    {
      sub_A62C00(*a4, *a1, (__int64)(a1 + 2), a1[1]);
      v13 = *a1;
      result = *(_QWORD *)(*a1 + 32);
      if ( result >= *(_QWORD *)(*a1 + 24) )
      {
        return sub_CB5D20(v13, 10);
      }
      else
      {
        *(_QWORD *)(v13 + 32) = result + 1;
        *(_BYTE *)result = 10;
      }
    }
  }
  return result;
}
