// Function: sub_C85E10
// Address: 0xc85e10
//
__int64 *__fastcall sub_C85E10(__int64 *a1, unsigned int a2, _QWORD *a3, size_t a4)
{
  size_t v6; // r14
  size_t v7; // rcx
  size_t v8; // rdx
  _QWORD *v10; // [rsp+10h] [rbp-50h]
  __int64 v11; // [rsp+20h] [rbp-40h] BYREF
  char v12; // [rsp+28h] [rbp-38h]

  v6 = a3[1];
  v10 = a3 + 3;
  v7 = v6;
  while ( 1 )
  {
    v8 = v6 + a4;
    if ( v6 + a4 != v7 )
    {
      if ( v6 + a4 >= v7 && v8 > a3[2] )
      {
        sub_C8D290(a3, v10, v8, 1);
        v8 = v6 + a4;
      }
      a3[1] = v8;
    }
    sub_C835A0((__int64)&v11, a2, (void *)(v6 + *a3), a4);
    if ( (v12 & 1) != 0 )
    {
      *a1 = v11 | 1;
      goto LABEL_11;
    }
    if ( !v11 )
      break;
    v7 = a3[1];
    v6 += v11;
  }
  *a1 = 1;
LABEL_11:
  a3[1] = v6;
  return a1;
}
