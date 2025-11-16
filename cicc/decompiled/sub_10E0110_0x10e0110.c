// Function: sub_10E0110
// Address: 0x10e0110
//
void *__fastcall sub_10E0110(void **a1, void **a2)
{
  void *result; // rax
  void *v3; // r13
  void *v4; // rbx
  void *v5; // rax
  void **v6; // rax
  void **v7; // r13
  void **v8; // rax
  void **v9; // rbx

  result = sub_C33340();
  v3 = *a2;
  v4 = result;
  if ( *a1 == result )
  {
    if ( result == v3 )
    {
      if ( a2 != a1 )
      {
        v8 = (void **)a1[1];
        if ( v8 )
        {
          v9 = &v8[3 * (_QWORD)*(v8 - 1)];
          if ( v8 != v9 )
          {
            do
            {
              v9 -= 3;
              if ( *v9 == v3 )
                sub_969EE0((__int64)v9);
              else
                sub_C338F0((__int64)v9);
            }
            while ( a1[1] != v9 );
          }
          j_j_j___libc_free_0_0(v9 - 1);
        }
        return sub_C3C840(a1, a2);
      }
    }
    else if ( a1 != a2 )
    {
      v6 = (void **)a1[1];
      if ( !v6 )
        return (void *)sub_C338E0((__int64)a1, (__int64)a2);
      v7 = &v6[3 * (_QWORD)*(v6 - 1)];
      if ( v6 != v7 )
      {
        do
        {
          v7 -= 3;
          if ( v4 == *v7 )
            sub_969EE0((__int64)v7);
          else
            sub_C338F0((__int64)v7);
        }
        while ( a1[1] != v7 );
      }
      j_j_j___libc_free_0_0(v7 - 1);
      v5 = *a2;
LABEL_6:
      if ( v4 != v5 )
        return (void *)sub_C338E0((__int64)a1, (__int64)a2);
      return sub_C3C840(a1, a2);
    }
  }
  else
  {
    if ( result != v3 )
      return (void *)sub_C33870((__int64)a1, (__int64)a2);
    if ( a1 != a2 )
    {
      sub_C338F0((__int64)a1);
      v5 = *a2;
      goto LABEL_6;
    }
  }
  return result;
}
