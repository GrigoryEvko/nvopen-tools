// Function: sub_2CAF220
// Address: 0x2caf220
//
_BYTE *__fastcall sub_2CAF220(__int64 a1, __int64 *a2, _BYTE *a3)
{
  __int64 v3; // rax
  __int64 v4; // r14
  _BYTE *v5; // rdx
  __int64 v7; // r15
  __int64 v8; // rbx
  __int64 i; // rsi
  _BYTE *v11; // rdx
  __int64 v13; // [rsp+8h] [rbp-38h]

  v3 = *((unsigned int *)a2 + 2);
  v4 = *a2 + 8 * v3;
  v13 = *a2;
  if ( v4 == *a2 )
    goto LABEL_15;
  v5 = *(_BYTE **)(v4 - 8);
  v7 = *a2;
  v8 = *a2 + 8 * v3;
  for ( i = (__int64)v5; ; i = *(_QWORD *)(v4 - 8) )
  {
    if ( (_BYTE *)i == v5 )
      goto LABEL_7;
    if ( (unsigned __int8)sub_B19DB0(a1, i, (__int64)v5) )
    {
      v5 = *(_BYTE **)(v8 - 8);
LABEL_7:
      v4 -= 8;
      if ( v7 == v4 )
        goto LABEL_8;
      continue;
    }
    if ( v7 == v4 )
      break;
    if ( v13 == v8 - 8 )
      goto LABEL_15;
    v7 = *a2;
    v4 = *a2 + 8LL * *((unsigned int *)a2 + 2);
    v5 = *(_BYTE **)(v8 - 16);
    if ( *a2 == v4 )
      goto LABEL_8;
    v8 -= 8;
  }
  v5 = *(_BYTE **)(v8 - 8);
LABEL_8:
  if ( v5 )
  {
LABEL_9:
    a3 = v5;
    if ( *v5 == 84 )
    {
      v11 = v5 + 24;
      while ( *(v11 - 24) == 84 )
      {
        v11 = (_BYTE *)*((_QWORD *)v11 + 1);
        if ( !v11 )
          BUG();
      }
      return v11 - 24;
    }
    return a3;
  }
LABEL_15:
  if ( a3 )
  {
    v5 = a3;
    goto LABEL_9;
  }
  return a3;
}
