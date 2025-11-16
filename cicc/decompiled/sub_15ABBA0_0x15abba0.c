// Function: sub_15ABBA0
// Address: 0x15abba0
//
__int64 __fastcall sub_15ABBA0(__int64 a1, unsigned __int8 *a2)
{
  __int64 result; // rax
  __int64 v4; // r13
  __int64 *v5; // rbx
  __int64 v6; // rsi
  __int64 v7; // r13
  __int64 *v8; // rbx
  __int64 v9; // rsi

  while ( 1 )
  {
    result = sub_15AB200(a1, (__int64)a2);
    if ( !(_BYTE)result )
      return result;
    sub_15AB790(a1, *(unsigned __int8 **)&a2[8 * (1LL - *((unsigned int *)a2 + 2))]);
    result = *a2;
    if ( (_BYTE)result == 14 )
      break;
    if ( (_BYTE)result == 13 )
    {
      sub_15ABBA0(a1, *(_QWORD *)&a2[8 * (3LL - *((unsigned int *)a2 + 2))]);
      result = 4LL - *((unsigned int *)a2 + 2);
      v7 = *(_QWORD *)&a2[8 * result];
      if ( !v7 )
        return result;
      result = 8LL * *(unsigned int *)(v7 + 8);
      v8 = (__int64 *)(v7 - result);
      if ( v7 - result == v7 )
        return result;
      v9 = *v8;
      result = *(unsigned __int8 *)*v8;
      if ( (unsigned __int8)result > 0xEu )
        goto LABEL_14;
LABEL_11:
      if ( (unsigned __int8)result > 0xAu )
        goto LABEL_17;
      while ( 1 )
      {
        while ( 1 )
        {
          if ( (__int64 *)v7 == ++v8 )
            return result;
          v9 = *v8;
          result = *(unsigned __int8 *)*v8;
          if ( (unsigned __int8)result <= 0xEu )
            goto LABEL_11;
LABEL_14:
          if ( (unsigned __int8)(result - 32) <= 1u )
            break;
          if ( (_BYTE)result == 17 )
            result = sub_15ABAC0(a1, v9);
        }
LABEL_17:
        result = sub_15ABBA0(a1, v9);
      }
    }
    if ( (_BYTE)result != 12 )
      return result;
    a2 = *(unsigned __int8 **)&a2[8 * (3LL - *((unsigned int *)a2 + 2))];
  }
  result = 3LL - *((unsigned int *)a2 + 2);
  v4 = *(_QWORD *)&a2[8 * result];
  if ( v4 )
  {
    result = 8LL * *(unsigned int *)(v4 + 8);
    v5 = (__int64 *)(v4 - result);
    if ( v4 - result != v4 )
    {
      do
      {
        v6 = *v5++;
        result = sub_15ABBA0(a1, v6);
      }
      while ( (__int64 *)v4 != v5 );
    }
  }
  return result;
}
