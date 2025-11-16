// Function: sub_1444AB0
// Address: 0x1444ab0
//
_QWORD *__fastcall sub_1444AB0(_QWORD *a1, __int64 a2, _QWORD *a3)
{
  _QWORD *v5; // rax
  _QWORD *result; // rax
  _QWORD *v7; // r14
  unsigned int i; // r12d
  _QWORD *v9; // rsi
  _QWORD *v10; // r8
  __int64 v11; // rdi
  __int64 v12; // rdx
  int v13; // [rsp+Ch] [rbp-44h]
  _QWORD *v14; // [rsp+10h] [rbp-40h]
  unsigned __int64 v15[7]; // [rsp+18h] [rbp-38h] BYREF

  v5 = (_QWORD *)a1[4];
  v15[0] = a2;
  v14 = v5;
  sub_1444990(a3, v15);
  sub_1443660(a1, v15[0]);
  result = (_QWORD *)sub_157EBA0(v15[0]);
  if ( result )
  {
    v13 = sub_15F4D60(result);
    result = (_QWORD *)sub_157EBA0(v15[0]);
    v7 = result;
    if ( v13 )
    {
      for ( i = 0; i != v13; ++i )
      {
        result = (_QWORD *)sub_15F4DF0(v7, i);
        v9 = result;
        if ( v14 != result )
        {
          result = (_QWORD *)a3[2];
          if ( !result )
            goto LABEL_12;
          v10 = a3 + 1;
          do
          {
            while ( 1 )
            {
              v11 = result[2];
              v12 = result[3];
              if ( result[4] >= (unsigned __int64)v9 )
                break;
              result = (_QWORD *)result[3];
              if ( !v12 )
                goto LABEL_10;
            }
            v10 = result;
            result = (_QWORD *)result[2];
          }
          while ( v11 );
LABEL_10:
          if ( a3 + 1 == v10 || v10[4] > (unsigned __int64)v9 )
LABEL_12:
            result = (_QWORD *)sub_1444AB0(a1, v9, a3);
        }
      }
    }
  }
  return result;
}
