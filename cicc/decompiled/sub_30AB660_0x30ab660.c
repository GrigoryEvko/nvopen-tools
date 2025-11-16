// Function: sub_30AB660
// Address: 0x30ab660
//
__int64 __fastcall sub_30AB660(__int64 a1, __int64 a2, __int64 **a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  __int64 result; // rax
  __int64 v8; // r14
  __int64 v9; // rbx
  __int64 *v10; // r12
  __int64 i; // r15
  __int64 v12; // rsi
  __int64 **v13; // rax
  __int64 v14; // rsi

  result = *(unsigned int *)(a2 + 8);
  if ( (int)result > 0 )
  {
    v8 = (__int64)a3;
    v9 = 0;
    while ( 1 )
    {
      v10 = *(__int64 **)(*(_QWORD *)a2 + 8 * v9);
      for ( i = v10[2]; i; i = *(_QWORD *)(i + 8) )
      {
        while ( 1 )
        {
          v12 = *(_QWORD *)(i + 24);
          if ( !*(_BYTE *)(v8 + 28) )
            break;
          result = *(_QWORD *)(v8 + 8);
          a3 = (__int64 **)(result + 8LL * *(unsigned int *)(v8 + 20));
          if ( (__int64 **)result == a3 )
            goto LABEL_16;
          while ( v12 != *(_QWORD *)result )
          {
            result += 8;
            if ( a3 == (__int64 **)result )
              goto LABEL_16;
          }
          i = *(_QWORD *)(i + 8);
          if ( !i )
            goto LABEL_10;
        }
        result = (__int64)sub_C8CA60(v8, v12);
        if ( !result )
          goto LABEL_16;
      }
LABEL_10:
      if ( !*(_BYTE *)(v8 + 28) )
        break;
      v13 = *(__int64 ***)(v8 + 8);
      v14 = *(unsigned int *)(v8 + 20);
      a3 = &v13[v14];
      if ( v13 == a3 )
      {
LABEL_22:
        if ( (unsigned int)v14 >= *(_DWORD *)(v8 + 16) )
          break;
        *(_DWORD *)(v8 + 20) = v14 + 1;
        *a3 = v10;
        ++*(_QWORD *)v8;
      }
      else
      {
        while ( v10 != *v13 )
        {
          if ( a3 == ++v13 )
            goto LABEL_22;
        }
      }
LABEL_15:
      result = sub_30AB500(v10, a1, a2, a4, a5, a6);
LABEL_16:
      if ( *(_DWORD *)(a2 + 8) <= (int)++v9 )
        return result;
    }
    sub_C8CC70(v8, (__int64)v10, (__int64)a3, a4, a5, a6);
    goto LABEL_15;
  }
  return result;
}
