// Function: sub_317F220
// Address: 0x317f220
//
_QWORD *__fastcall sub_317F220(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r14
  __int64 v5; // r15
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 v8; // rbx
  __int64 v9; // rax
  _BYTE *v10; // rsi
  __int64 v12[7]; // [rsp+18h] [rbp-38h] BYREF

  *a1 = 0;
  a1[1] = 0;
  a1[2] = 0;
  if ( a3 )
  {
    v4 = sub_317E6A0(a2, a3);
    v5 = sub_C1B090(a3, 0);
    v6 = sub_317E450(v4);
    v7 = *(_QWORD *)(v6 + 24);
    v8 = v6 + 8;
    if ( v7 != v6 + 8 )
    {
      while ( 1 )
      {
        if ( sub_317E640(v7 + 40) != v5 )
          goto LABEL_4;
        v9 = sub_317E470(v7 + 40);
        v12[0] = v9;
        if ( !v9 )
          goto LABEL_4;
        v10 = (_BYTE *)a1[1];
        if ( v10 == (_BYTE *)a1[2] )
        {
          sub_26C32C0((__int64)a1, v10, v12);
LABEL_4:
          v7 = sub_220EEE0(v7);
          if ( v8 == v7 )
            return a1;
        }
        else
        {
          if ( v10 )
          {
            *(_QWORD *)v10 = v9;
            v10 = (_BYTE *)a1[1];
          }
          a1[1] = v10 + 8;
          v7 = sub_220EEE0(v7);
          if ( v8 == v7 )
            return a1;
        }
      }
    }
  }
  return a1;
}
