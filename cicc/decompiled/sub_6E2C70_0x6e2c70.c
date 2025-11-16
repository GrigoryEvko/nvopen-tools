// Function: sub_6E2C70
// Address: 0x6e2c70
//
__int64 *__fastcall sub_6E2C70(__int64 a1, int a2, __int64 *a3, __int64 a4)
{
  __int64 v5; // rsi
  _QWORD *v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 *result; // rax

  v5 = (__int64)a3;
  v6 = (_QWORD *)a4;
  if ( a4 )
  {
    if ( a3 )
    {
      v7 = *a3;
      v8 = qword_4D03C50;
      if ( !*a3 )
        goto LABEL_7;
      goto LABEL_5;
    }
  }
  else if ( a3 )
  {
    v6 = a3 + 17;
    goto LABEL_4;
  }
  if ( a4 )
  {
    v5 = *(_QWORD *)(a4 + 16);
    if ( v5 )
    {
LABEL_4:
      v7 = *(_QWORD *)v5;
      v8 = qword_4D03C50;
      if ( !*(_QWORD *)v5 )
        goto LABEL_8;
LABEL_5:
      if ( (unsigned __int8)(*(_BYTE *)(v7 + 80) - 8) <= 1u )
        *(_QWORD *)(v8 + 120) = 0;
LABEL_7:
      if ( !v6 )
      {
LABEL_9:
        if ( (*(_BYTE *)(v8 + 24) & 2) != 0 )
          *(_BYTE *)(v5 + 176) |= 0x10u;
        result = sub_6E2B30(a1, v5);
        if ( a2 )
        {
          result = sub_6E1DF0(a1);
          if ( v6 )
            *((_BYTE *)v6 + 42) |= 2u;
        }
        return result;
      }
LABEL_8:
      *((_BYTE *)v6 + 42) = (*(_BYTE *)(v8 + 19) >> 5 << 7) | *((_BYTE *)v6 + 42) & 0x7F;
      goto LABEL_9;
    }
    *(_BYTE *)(a4 + 42) = (*(_BYTE *)(qword_4D03C50 + 19LL) >> 5 << 7) | *(_BYTE *)(a4 + 42) & 0x7F;
    result = sub_6E2B30(a1, 0);
    if ( a2 )
    {
      result = sub_6E1DF0(a1);
      *((_BYTE *)v6 + 42) |= 2u;
    }
  }
  else
  {
    result = sub_6E2B30(a1, 0);
    if ( a2 )
      return sub_6E1DF0(a1);
  }
  return result;
}
