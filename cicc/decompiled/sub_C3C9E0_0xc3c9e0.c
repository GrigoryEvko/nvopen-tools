// Function: sub_C3C9E0
// Address: 0xc3c9e0
//
__int64 *__fastcall sub_C3C9E0(__int64 *a1, __int64 *a2)
{
  __int64 *v2; // r12
  void **v3; // rax
  void **v4; // rbx
  void *v5; // r14
  __int64 *v7; // r14
  __int64 *v8; // r15
  void *v9; // rbx
  __int64 v10; // rax
  void *v11; // rax
  __int64 *v12; // rsi
  __int64 *v13; // r13

  v2 = a1;
  if ( *a1 != *a2 || (v7 = (__int64 *)a2[1]) == 0 )
  {
    if ( a1 != a2 )
    {
      v3 = (void **)a1[1];
      if ( v3 )
      {
        v4 = &v3[3 * (_QWORD)*(v3 - 1)];
        if ( v3 != v4 )
        {
          v5 = sub_C33340();
          do
          {
            while ( 1 )
            {
              v4 -= 3;
              if ( *v4 == v5 )
                break;
              sub_C338F0((__int64)v4);
              if ( (void **)a1[1] == v4 )
                goto LABEL_9;
            }
            sub_969EE0((__int64)v4);
          }
          while ( (void **)a1[1] != v4 );
        }
LABEL_9:
        j_j_j___libc_free_0_0(v4 - 1);
      }
LABEL_10:
      sub_C3C790(a1, (_QWORD **)a2);
      return v2;
    }
    return v2;
  }
  v8 = (__int64 *)a1[1];
  v9 = sub_C33340();
  v10 = *v7;
  if ( (void *)*v8 != v9 )
  {
    if ( (void *)v10 != v9 )
    {
      sub_C33E70(v8, v7);
      v7 = (__int64 *)a2[1];
      v8 = (__int64 *)a1[1];
      goto LABEL_16;
    }
    if ( v7 == v8 )
      goto LABEL_16;
    sub_C338F0((__int64)v8);
LABEL_21:
    if ( v9 == (void *)*v7 )
      sub_C3C790(v8, (_QWORD **)v7);
    else
      sub_C33EB0(v8, v7);
    v7 = (__int64 *)a2[1];
    v8 = (__int64 *)a1[1];
    goto LABEL_16;
  }
  if ( (void *)v10 == v9 )
  {
    sub_C3C9E0(v8, v7);
    v7 = (__int64 *)a2[1];
    v8 = (__int64 *)a1[1];
    goto LABEL_16;
  }
  if ( v7 != v8 )
  {
    sub_969EE0((__int64)v8);
    goto LABEL_21;
  }
LABEL_16:
  v11 = (void *)v7[3];
  v12 = v7 + 3;
  v13 = v8 + 3;
  if ( (void *)v8[3] != v9 )
  {
    if ( v9 != v11 )
    {
      sub_C33E70(v8 + 3, v12);
      return v2;
    }
    if ( v12 == v13 )
      return v2;
    sub_C338F0((__int64)(v8 + 3));
    a2 = v7 + 3;
    goto LABEL_26;
  }
  if ( v9 == v11 )
  {
    sub_C3C9E0(v8 + 3, v12);
    return v2;
  }
  if ( v12 != v13 )
  {
    sub_969EE0((__int64)(v8 + 3));
    a2 = v7 + 3;
LABEL_26:
    a1 = v8 + 3;
    if ( v9 != (void *)v7[3] )
    {
      sub_C33EB0(a1, a2);
      return v2;
    }
    goto LABEL_10;
  }
  return v2;
}
