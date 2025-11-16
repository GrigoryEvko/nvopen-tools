// Function: sub_880640
// Address: 0x880640
//
_QWORD *__fastcall sub_880640(__int64 a1)
{
  __int64 v1; // rax
  __int64 ***i; // rbx
  __int64 v3; // r14
  _QWORD *result; // rax
  __int64 **v5; // rbx
  int v6; // edx
  __int64 v7; // r13
  __int64 *j; // rax
  _QWORD *k; // r12
  __int64 v10; // rdi

  v1 = a1;
  for ( i = *(__int64 ****)(a1 + 168); *(_BYTE *)(v1 + 140) == 12; v1 = *(_QWORD *)(v1 + 160) )
    ;
  v3 = *(_QWORD *)(*(_QWORD *)v1 + 96LL);
  result = (_QWORD *)sub_72AD30(a1);
  v5 = *i;
  v6 = 0;
  v7 = (__int64)result;
  if ( v5 )
  {
    while ( 1 )
    {
      while ( ((_BYTE)v5[12] & 1) == 0 )
      {
        v5 = (__int64 **)*v5;
        if ( !v5 )
          return result;
      }
      for ( j = v5[5]; *((_BYTE *)j + 140) == 12; j = (__int64 *)j[20] )
        ;
      result = *(_QWORD **)(*j + 96);
      if ( v6 )
      {
        for ( k = (_QWORD *)result[17]; k; k = (_QWORD *)*k )
        {
          while ( 1 )
          {
            result = *(_QWORD **)(v3 + 136);
            v10 = k[1];
            if ( result )
              break;
LABEL_24:
            result = sub_878810(v10, 0);
            *result = *(_QWORD *)(v3 + 136);
            *(_QWORD *)(v3 + 136) = result;
            k = (_QWORD *)*k;
            if ( !k )
              goto LABEL_14;
          }
          while ( v10 != result[1] )
          {
            result = (_QWORD *)*result;
            if ( !result )
              goto LABEL_24;
          }
        }
      }
      else
      {
        result = (_QWORD *)result[17];
        *(_QWORD *)(v3 + 136) = result;
        if ( !result )
        {
LABEL_26:
          result = sub_878810(v7, 0);
          *result = *(_QWORD *)(v3 + 136);
          v6 = 1;
          *(_QWORD *)(v3 + 136) = result;
          goto LABEL_15;
        }
        while ( v7 != result[1] )
        {
          result = (_QWORD *)*result;
          if ( !result )
            goto LABEL_26;
        }
      }
LABEL_14:
      v6 = 1;
LABEL_15:
      v5 = (__int64 **)*v5;
      if ( !v5 )
        return result;
    }
  }
  result = sub_878810((__int64)result, 1);
  *(_QWORD *)(v3 + 136) = result;
  return result;
}
