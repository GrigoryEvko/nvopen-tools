// Function: sub_25A37B0
// Address: 0x25a37b0
//
__int64 __fastcall sub_25A37B0(char *a1, _QWORD *a2, unsigned __int64 a3)
{
  char *v3; // r12
  unsigned __int64 v4; // r13
  _QWORD *v5; // rbx
  _QWORD *v6; // r13
  _QWORD *v7; // r14
  char *v8; // rdi
  __int64 v9; // r15
  unsigned __int64 v10; // r15
  unsigned __int8 v12; // [rsp+Ch] [rbp-34h]

  v3 = a1;
  v4 = a2[1] - *a2;
  v12 = a3;
  *(_QWORD *)a1 = 0;
  *((_QWORD *)a1 + 1) = 0;
  *((_QWORD *)a1 + 2) = 0;
  if ( v4 )
  {
    if ( v4 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_17:
      sub_4261EA(a1, a2, a3);
    a1 = (char *)v4;
    v5 = (_QWORD *)sub_22077B0(v4);
  }
  else
  {
    v4 = 0;
    v5 = 0;
  }
  *(_QWORD *)v3 = v5;
  *((_QWORD *)v3 + 1) = v5;
  *((_QWORD *)v3 + 2) = (char *)v5 + v4;
  v6 = (_QWORD *)a2[1];
  if ( v6 != (_QWORD *)*a2 )
  {
    v7 = (_QWORD *)*a2;
    do
    {
      if ( v5 )
      {
        a3 = v7[1] - *v7;
        *v5 = 0;
        v5[1] = 0;
        v10 = a3;
        v5[2] = 0;
        if ( a3 )
        {
          if ( a3 > 0x7FFFFFFFFFFFFFF8LL )
            goto LABEL_17;
          v8 = (char *)sub_22077B0(a3);
        }
        else
        {
          v8 = 0;
        }
        *v5 = v8;
        v5[1] = v8;
        v5[2] = &v8[v10];
        a2 = (_QWORD *)*v7;
        v9 = v7[1] - *v7;
        if ( v7[1] != *v7 )
          v8 = (char *)memmove(v8, a2, v7[1] - *v7);
        a1 = &v8[v9];
        v5[1] = a1;
      }
      v7 += 3;
      v5 += 3;
    }
    while ( v6 != v7 );
  }
  *((_QWORD *)v3 + 1) = v5;
  v3[24] = v12;
  return v12;
}
