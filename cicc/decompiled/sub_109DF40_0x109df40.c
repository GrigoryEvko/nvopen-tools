// Function: sub_109DF40
// Address: 0x109df40
//
_QWORD *__fastcall sub_109DF40(_QWORD *a1, __int64 a2, int a3)
{
  void *v4; // rax
  void *v5; // r13
  __int64 v7; // rdx
  _QWORD *i; // rbx
  void *v9; // [rsp+0h] [rbp-50h] BYREF
  _QWORD *v10; // [rsp+8h] [rbp-48h]

  v4 = sub_C33340();
  v5 = v4;
  if ( a3 < 0 )
  {
    v7 = -a3;
    if ( (void *)a2 == v4 )
      sub_C3C5A0(&v9, (__int64)v4, v7);
    else
      sub_C36740((__int64)&v9, a2, v7);
    if ( v9 == v5 )
      sub_C3CCB0((__int64)&v9);
    else
      sub_C34440((unsigned __int8 *)&v9);
    if ( v9 == v5 )
      sub_C3C840(a1, &v9);
    else
      sub_C338E0((__int64)a1, (__int64)&v9);
    if ( v9 == v5 )
    {
      if ( v10 )
      {
        for ( i = &v10[3 * *(v10 - 1)]; v10 != i; sub_91D830(i) )
          i -= 3;
        j_j_j___libc_free_0_0(i - 1);
      }
    }
    else
    {
      sub_C338F0((__int64)&v9);
    }
  }
  else if ( (void *)a2 == v4 )
  {
    sub_C3C5A0(a1, a2, a3);
  }
  else
  {
    sub_C36740((__int64)a1, a2, a3);
  }
  return a1;
}
