// Function: sub_96A790
// Address: 0x96a790
//
__int64 __fastcall sub_96A790(double (__fastcall *a1)(double), _QWORD *a2, _QWORD *a3)
{
  int *v4; // rax
  int *v5; // r14
  double v6; // xmm0_8
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // r14
  _QWORD *i; // r15
  _QWORD *j; // r12
  double v13; // [rsp+0h] [rbp-60h]
  __int64 v14; // [rsp+10h] [rbp-50h] BYREF
  _QWORD *v15; // [rsp+18h] [rbp-48h]

  feclearexcept(61);
  v4 = __errno_location();
  *v4 = 0;
  v5 = v4;
  sub_9690C0(&v14, (__int64)a3, a2);
  v6 = sub_C41B00(&v14);
  v13 = a1(v6);
  v7 = sub_C33340();
  if ( v14 == v7 )
  {
    if ( v15 )
    {
      for ( i = &v15[3 * *(v15 - 1)]; v15 != i; sub_969EE0((__int64)i) )
      {
        while ( 1 )
        {
          i -= 3;
          if ( v7 == *i )
            break;
          sub_C338F0(i);
          if ( v15 == i )
            goto LABEL_11;
        }
      }
LABEL_11:
      j_j_j___libc_free_0_0(i - 1);
    }
  }
  else
  {
    sub_C338F0(&v14);
  }
  if ( (unsigned int)(*v5 - 33) <= 1 || fetestexcept(29) )
  {
    feclearexcept(61);
    *v5 = 0;
    return 0;
  }
  else
  {
    v8 = sub_96A450(a3, v13);
    sub_9690C0(&v14, (__int64)a3, (_QWORD *)(v8 + 24));
    v9 = sub_AC8EA0(*a3, &v14);
    if ( v7 == v14 )
    {
      if ( v15 )
      {
        for ( j = &v15[3 * *(v15 - 1)]; v15 != j; sub_969EE0((__int64)j) )
        {
          while ( 1 )
          {
            j -= 3;
            if ( v7 == *j )
              break;
            sub_C338F0(j);
            if ( v15 == j )
              goto LABEL_22;
          }
        }
LABEL_22:
        j_j_j___libc_free_0_0(j - 1);
      }
    }
    else
    {
      sub_C338F0(&v14);
    }
  }
  return v9;
}
