// Function: sub_7AD800
// Address: 0x7ad800
//
_QWORD *__fastcall sub_7AD800(_QWORD *a1, __int64 a2)
{
  _QWORD *v3; // r14
  _QWORD *v4; // r13
  _QWORD *v5; // r12
  __int64 v6; // rbx
  __int64 v7; // rax
  _QWORD *v8; // rsi
  bool v9; // al
  _QWORD *v11; // [rsp+8h] [rbp-58h]
  _QWORD *v12; // [rsp+10h] [rbp-50h]
  _QWORD *v13; // [rsp+18h] [rbp-48h]
  _QWORD *v14; // [rsp+20h] [rbp-40h]
  _QWORD *v15; // [rsp+28h] [rbp-38h]

  v13 = a1;
  if ( a1[6] != a2 )
  {
    v13 = (_QWORD *)a1[5];
    if ( v13 )
    {
      v12 = (_QWORD *)a1[5];
      while ( 1 )
      {
        if ( a2 == v12[6] )
          return v12;
        v11 = (_QWORD *)v12[5];
        if ( v11 )
          break;
        v14 = 0;
        v9 = 0;
LABEL_52:
        v12 = (_QWORD *)v12[7];
        if ( !v12 || v9 )
          return v14;
      }
      while ( 1 )
      {
        if ( a2 == v11[6] )
          return v11;
        v13 = (_QWORD *)v11[5];
        if ( v13 )
          break;
        v15 = 0;
        v9 = 0;
LABEL_43:
        v11 = (_QWORD *)v11[7];
        if ( !v11 || v9 )
        {
          v14 = v15;
          goto LABEL_52;
        }
      }
      while ( a2 != v13[6] )
      {
        v14 = (_QWORD *)v13[5];
        if ( v14 )
        {
          while ( a2 != v14[6] )
          {
            v15 = (_QWORD *)v14[5];
            if ( v15 )
            {
              do
              {
                if ( a2 == v15[6] )
                {
                  v9 = 1;
                  goto LABEL_43;
                }
                v3 = (_QWORD *)v15[5];
                if ( v3 )
                {
                  while ( 1 )
                  {
                    if ( a2 == v3[6] )
                    {
                      v9 = 1;
                      goto LABEL_35;
                    }
                    v4 = (_QWORD *)v3[5];
                    if ( v4 )
                      break;
                    v9 = 0;
LABEL_39:
                    v3 = (_QWORD *)v3[7];
                    if ( !v3 || v9 )
                    {
                      v5 = v4;
                      goto LABEL_47;
                    }
                  }
                  while ( 1 )
                  {
                    if ( a2 == v4[6] )
                    {
                      v9 = 1;
                      goto LABEL_27;
                    }
                    v5 = (_QWORD *)v4[5];
                    if ( v5 )
                      break;
                    v9 = 0;
LABEL_31:
                    v4 = (_QWORD *)v4[7];
                    if ( !v4 || v9 )
                    {
                      v4 = v5;
                      goto LABEL_39;
                    }
                  }
                  while ( a2 != v5[6] )
                  {
                    v6 = v5[5];
                    if ( v6 )
                    {
                      do
                      {
                        v7 = sub_7AD800(v6, a2);
                        v6 = *(_QWORD *)(v6 + 56);
                        v8 = (_QWORD *)v7;
                        v9 = v7 != 0;
                      }
                      while ( v6 && !v9 );
                    }
                    else
                    {
                      v8 = 0;
                      v9 = 0;
                    }
                    v5 = (_QWORD *)v5[7];
                    if ( !v5 || v9 )
                    {
                      v5 = v8;
                      goto LABEL_31;
                    }
                  }
                  v9 = 1;
                }
                else
                {
                  v5 = 0;
                  v9 = 0;
                }
LABEL_47:
                v15 = (_QWORD *)v15[7];
              }
              while ( v15 && !v9 );
              v4 = v5;
            }
            else
            {
              v4 = 0;
              v9 = 0;
            }
LABEL_27:
            v14 = (_QWORD *)v14[7];
            if ( !v14 || v9 )
            {
              v3 = v4;
              goto LABEL_35;
            }
          }
          v9 = 1;
          goto LABEL_52;
        }
        v3 = 0;
        v9 = 0;
LABEL_35:
        v13 = (_QWORD *)v13[7];
        if ( !v13 || v9 )
        {
          v15 = v3;
          goto LABEL_43;
        }
      }
    }
  }
  return v13;
}
