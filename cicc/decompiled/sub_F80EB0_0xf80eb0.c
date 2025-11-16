// Function: sub_F80EB0
// Address: 0xf80eb0
//
_QWORD *__fastcall sub_F80EB0(__int64 *a1, unsigned __int64 a2, __int64 a3)
{
  unsigned int v5; // eax
  unsigned int v6; // r13d
  unsigned __int8 v7; // al
  _QWORD *result; // rax
  __int64 v9; // rax
  __int64 v10; // rsi
  __int64 v11; // rbx
  bool v12; // zf
  __int64 v13; // rbx
  __int64 v14; // rax
  __int64 v15; // rbx
  const char *v16; // [rsp+0h] [rbp-60h] BYREF
  char v17; // [rsp+20h] [rbp-40h]
  char v18; // [rsp+21h] [rbp-3Fh]

  v5 = sub_B50D10(a2, 0, a3, 0);
  v6 = v5;
  if ( v5 == 48 )
  {
    v10 = *(_DWORD *)(a3 + 8) >> 8;
    if ( *((_BYTE *)sub_AE2980(a1[1], v10) + 16) )
    {
      v18 = 1;
      v16 = "scevgep";
      v17 = 3;
      v14 = sub_AD6530(a3, v10);
      return sub_F7CA10(a1 + 65, v14, a2, (__int64)&v16, 0);
    }
  }
  else
  {
    if ( v5 == 49 )
    {
      if ( a3 == *(_QWORD *)(a2 + 8) )
        return (_QWORD *)a2;
      v7 = *(_BYTE *)a2;
      if ( (unsigned __int8)(*(_BYTE *)a2 - 67) > 0xCu )
      {
LABEL_5:
        if ( v7 <= 0x15u )
          return (_QWORD *)sub_ADAB70(v6, a2, (__int64 **)a3, 0);
        goto LABEL_10;
      }
      result = *(_QWORD **)(a2 - 32);
      if ( a3 == result[1] )
        return result;
      goto LABEL_10;
    }
    if ( v5 - 47 > 1 )
    {
LABEL_4:
      v7 = *(_BYTE *)a2;
      goto LABEL_5;
    }
  }
  v11 = sub_D97050(*a1, a3);
  v12 = v11 == sub_D97050(*a1, *(_QWORD *)(a2 + 8));
  v7 = *(_BYTE *)a2;
  if ( !v12 )
    goto LABEL_5;
  if ( v7 > 0x1Cu )
  {
    if ( (unsigned __int8)(v7 - 76) <= 1u )
    {
      v15 = sub_D97050(*a1, *(_QWORD *)(a2 + 8));
      if ( v15 == sub_D97050(*a1, *(_QWORD *)(*(_QWORD *)(a2 - 32) + 8LL)) )
        return *(_QWORD **)(a2 - 32);
      v7 = *(_BYTE *)a2;
      goto LABEL_14;
    }
LABEL_10:
    v9 = sub_F7D780((__int64)a1, a2);
    return (_QWORD *)sub_F80B30((__int64)a1, a2, a3, v6, v9);
  }
LABEL_14:
  if ( v7 != 5 )
    goto LABEL_5;
  if ( (unsigned int)*(unsigned __int16 *)(a2 + 2) - 47 > 1 )
    return (_QWORD *)sub_ADAB70(v6, a2, (__int64 **)a3, 0);
  v13 = sub_D97050(*a1, *(_QWORD *)(a2 + 8));
  if ( v13 != sub_D97050(*a1, *(_QWORD *)(*(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)) + 8LL)) )
    goto LABEL_4;
  return *(_QWORD **)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
}
