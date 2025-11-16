// Function: sub_F070B0
// Address: 0xf070b0
//
char *__fastcall sub_F070B0(__int64 *src, __int64 *a2, __int64 *a3, __int64 *a4, _QWORD *a5)
{
  __int64 *v5; // r13
  __int64 *v6; // r12
  __int64 v8; // rax
  __int64 v9; // r14
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // rax
  signed __int64 v13; // r8
  char *v14; // rbx

  v5 = a3;
  v6 = src;
  if ( src != a2 && a3 != a4 )
  {
    do
    {
      v9 = *v5;
      v10 = sub_B140A0(*v6);
      v11 = sub_B140A0(v9);
      if ( sub_B445A0(v10, v11) )
      {
        v8 = *v5;
        ++a5;
        ++v5;
        *(a5 - 1) = v8;
        if ( v6 == a2 )
          break;
      }
      else
      {
        v12 = *v6;
        ++a5;
        ++v6;
        *(a5 - 1) = v12;
        if ( v6 == a2 )
          break;
      }
    }
    while ( v5 != a4 );
  }
  v13 = (char *)a2 - (char *)v6;
  if ( a2 != v6 )
  {
    memmove(a5, v6, (char *)a2 - (char *)v6);
    v13 = (char *)a2 - (char *)v6;
  }
  v14 = (char *)a5 + v13;
  if ( a4 != v5 )
    memmove(v14, v5, (char *)a4 - (char *)v5);
  return &v14[(char *)a4 - (char *)v5];
}
