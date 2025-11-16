// Function: sub_ED82D0
// Address: 0xed82d0
//
char *__fastcall sub_ED82D0(__int64 a1, _QWORD *a2, unsigned __int64 a3)
{
  _QWORD *v3; // r12
  _QWORD *v4; // rbx
  unsigned __int64 v5; // r13
  char *v6; // rcx
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v9; // r13
  char *v10; // rcx
  __int64 v11; // r13
  char *result; // rax
  _QWORD *v13; // r14
  _QWORD *v14; // r12
  unsigned __int64 v15; // r13
  _QWORD *v16; // r15
  _QWORD *v17; // r13
  _QWORD *v18; // rbx
  char *v19; // rcx
  __int64 v20; // r12
  unsigned __int64 v21; // r12
  _QWORD *v22; // [rsp+0h] [rbp-50h]
  __int64 v23; // [rsp+8h] [rbp-48h]
  _QWORD *v24; // [rsp+18h] [rbp-38h]

  v3 = a2;
  v4 = (_QWORD *)a1;
  v5 = a2[1] - *a2;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  if ( v5 )
  {
    if ( v5 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_36;
    a1 = v5;
    v6 = (char *)sub_22077B0(v5);
  }
  else
  {
    v5 = 0;
    v6 = 0;
  }
  *v4 = v6;
  v4[2] = &v6[v5];
  v4[1] = v6;
  v7 = a2[1];
  a2 = (_QWORD *)*a2;
  v8 = v3[1] - *v3;
  if ( v7 != *v3 )
  {
    a1 = (__int64)v6;
    v6 = (char *)memmove(v6, a2, v3[1] - *v3);
  }
  v4[1] = &v6[v8];
  v9 = v3[4] - v3[3];
  v4[3] = 0;
  v4[4] = 0;
  v4[5] = 0;
  if ( v9 )
  {
    if ( v9 >= 0 )
    {
      v10 = (char *)sub_22077B0(v9);
      goto LABEL_9;
    }
LABEL_36:
    sub_4261EA(a1, a2, a3);
  }
  v10 = 0;
LABEL_9:
  v4[3] = v10;
  v4[5] = &v10[v9];
  v11 = 0;
  v4[4] = v10;
  a2 = (_QWORD *)v3[3];
  result = (char *)(v3[4] - (_QWORD)a2);
  if ( result )
  {
    v11 = v3[4] - (_QWORD)a2;
    result = (char *)memmove(v10, a2, (size_t)result);
    v10 = result;
  }
  v4[4] = &v10[v11];
  v13 = (_QWORD *)v3[6];
  if ( v13 )
  {
    a1 = 72;
    v23 = sub_22077B0(72);
    if ( v23 )
    {
      v22 = v4;
      v14 = (_QWORD *)v23;
      do
      {
        v15 = v13[1] - *v13;
        *v14 = 0;
        v14[1] = 0;
        v14[2] = 0;
        if ( v15 )
        {
          if ( v15 > 0x7FFFFFFFFFFFFFF8LL )
            goto LABEL_36;
          a1 = v15;
          v16 = (_QWORD *)sub_22077B0(v15);
        }
        else
        {
          v16 = 0;
        }
        *v14 = v16;
        v14[1] = v16;
        v14[2] = (char *)v16 + v15;
        v17 = (_QWORD *)v13[1];
        if ( v17 != (_QWORD *)*v13 )
        {
          v24 = v14;
          v18 = (_QWORD *)*v13;
          do
          {
            if ( v16 )
            {
              a3 = v18[1] - *v18;
              *v16 = 0;
              v16[1] = 0;
              v21 = a3;
              v16[2] = 0;
              if ( a3 )
              {
                if ( a3 > 0x7FFFFFFFFFFFFFF0LL )
                  goto LABEL_36;
                a1 = a3;
                v19 = (char *)sub_22077B0(a3);
              }
              else
              {
                v19 = 0;
              }
              a3 = (unsigned __int64)&v19[v21];
              *v16 = v19;
              v16[1] = v19;
              v16[2] = &v19[v21];
              a2 = (_QWORD *)*v18;
              v20 = v18[1] - *v18;
              if ( v18[1] != *v18 )
              {
                a1 = (__int64)v19;
                v19 = (char *)memmove(v19, a2, v18[1] - *v18);
              }
              v16[1] = &v19[v20];
            }
            v18 += 3;
            v16 += 3;
          }
          while ( v17 != v18 );
          v14 = v24;
        }
        v14[1] = v16;
        v13 += 3;
        v14 += 3;
      }
      while ( (_QWORD *)(v23 + 72) != v14 );
      v4 = v22;
    }
    v4[6] = v23;
    return (char *)v23;
  }
  else
  {
    v4[6] = 0;
  }
  return result;
}
