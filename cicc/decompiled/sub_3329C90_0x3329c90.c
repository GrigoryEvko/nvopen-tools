// Function: sub_3329C90
// Address: 0x3329c90
//
_QWORD *__fastcall sub_3329C90(_QWORD *a1, __int64 *a2, unsigned int a3, char a4)
{
  _DWORD *v5; // r15
  _DWORD *v6; // rax
  void *v8; // r13
  void **i; // rbx
  void **j; // rbx
  _QWORD v12[4]; // [rsp+10h] [rbp-90h] BYREF
  __int64 v13; // [rsp+30h] [rbp-70h] BYREF
  void **v14; // [rsp+38h] [rbp-68h]
  __int64 v15; // [rsp+50h] [rbp-50h] BYREF
  void **v16; // [rsp+58h] [rbp-48h]

  v5 = (_DWORD *)*a2;
  v6 = sub_C33340();
  if ( v5 == v6 )
  {
    v8 = v6;
    sub_C40CE0(&v13, (__int64)a2, a3, a4);
    sub_C3C840(&v15, &v13);
    sub_C3C840(a1, &v15);
    if ( v16 )
    {
      for ( i = &v16[3 * (_QWORD)*(v16 - 1)]; v16 != i; sub_969EE0((__int64)i) )
      {
        while ( 1 )
        {
          i -= 3;
          if ( v8 == *i )
            break;
          sub_C338F0((__int64)i);
          if ( v16 == i )
            goto LABEL_6;
        }
      }
LABEL_6:
      j_j_j___libc_free_0_0((unsigned __int64)(i - 1));
    }
    if ( v14 )
    {
      for ( j = &v14[3 * (_QWORD)*(v14 - 1)]; v14 != j; sub_969EE0((__int64)j) )
      {
        while ( 1 )
        {
          j -= 3;
          if ( v8 == *j )
            break;
          sub_C338F0((__int64)j);
          if ( v14 == j )
            goto LABEL_9;
        }
      }
LABEL_9:
      j_j_j___libc_free_0_0((unsigned __int64)(j - 1));
    }
  }
  else
  {
    sub_C33EB0(v12, a2);
    sub_C3BDC0((__int64)&v13, (__int64)v12, a3, a4);
    sub_C338E0((__int64)&v15, (__int64)&v13);
    sub_C407B0(a1, &v15, v5);
    sub_C338F0((__int64)&v15);
    sub_C338F0((__int64)&v13);
    sub_C338F0((__int64)v12);
  }
  return a1;
}
