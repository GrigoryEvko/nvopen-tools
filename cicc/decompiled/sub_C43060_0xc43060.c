// Function: sub_C43060
// Address: 0xc43060
//
__int64 __fastcall sub_C43060(__int64 a1, _QWORD *a2, unsigned __int64 a3, unsigned __int64 a4, unsigned __int8 a5)
{
  _DWORD *v7; // rax
  void *v8; // rbx
  void **v9; // rax
  void **v10; // r13
  void **i; // r12
  void **j; // r12
  __int64 v15; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v16; // [rsp+18h] [rbp-68h]
  __int64 v17; // [rsp+20h] [rbp-60h] BYREF
  void **v18; // [rsp+28h] [rbp-58h]
  void *v19; // [rsp+30h] [rbp-50h] BYREF
  void **v20; // [rsp+38h] [rbp-48h]

  v7 = sub_C33340();
  v8 = v7;
  if ( v7 == dword_3F65580 )
    sub_C3C460(&v19, (__int64)v7);
  else
    sub_C37380(&v19, (__int64)dword_3F65580);
  sub_C43000(a1, &v19, a3, a4, a5);
  if ( v19 == v8 )
    sub_C3E660((__int64)&v15, (__int64)&v19);
  else
    sub_C3A850((__int64)&v15, (__int64 *)&v19);
  sub_C3C640(&v17, (__int64)dword_3F655A0, &v15);
  v9 = (void **)a2[1];
  if ( v9 )
  {
    v10 = &v9[3 * (_QWORD)*(v9 - 1)];
    if ( v9 != v10 )
    {
      do
      {
        while ( 1 )
        {
          v10 -= 3;
          if ( v8 == *v10 )
            break;
          sub_C338F0((__int64)v10);
          if ( (void **)a2[1] == v10 )
            goto LABEL_11;
        }
        sub_969EE0((__int64)v10);
      }
      while ( (void **)a2[1] != v10 );
    }
LABEL_11:
    j_j_j___libc_free_0_0(v10 - 1);
  }
  sub_C3C840(a2, &v17);
  if ( v18 )
  {
    for ( i = &v18[3 * (_QWORD)*(v18 - 1)]; v18 != i; sub_969EE0((__int64)i) )
    {
      while ( 1 )
      {
        i -= 3;
        if ( v8 == *i )
          break;
        sub_C338F0((__int64)i);
        if ( v18 == i )
          goto LABEL_18;
      }
    }
LABEL_18:
    j_j_j___libc_free_0_0(i - 1);
  }
  if ( v16 > 0x40 && v15 )
    j_j___libc_free_0_0(v15);
  if ( v19 == v8 )
  {
    if ( v20 )
    {
      for ( j = &v20[3 * (_QWORD)*(v20 - 1)]; v20 != j; sub_969EE0((__int64)j) )
      {
        while ( 1 )
        {
          j -= 3;
          if ( v8 == *j )
            break;
          sub_C338F0((__int64)j);
          if ( v20 == j )
            goto LABEL_29;
        }
      }
LABEL_29:
      j_j_j___libc_free_0_0(j - 1);
    }
  }
  else
  {
    sub_C338F0((__int64)&v19);
  }
  return a1;
}
