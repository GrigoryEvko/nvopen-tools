// Function: sub_23FB390
// Address: 0x23fb390
//
char *__fastcall sub_23FB390(__int64 ***src, __int64 ***a2, __int64 ***a3, __int64 ***a4, _QWORD *a5)
{
  __int64 ***v6; // r13
  __int64 ***v7; // r12
  __int64 **v8; // rax
  __int64 **v9; // r14
  unsigned int v10; // ebx
  __int64 **v11; // rax
  char *v12; // r8

  v6 = a3;
  v7 = src;
  if ( a2 != src && a4 != a3 )
  {
    do
    {
      v9 = *v7;
      v10 = sub_22DADF0(***v6);
      if ( v10 < (unsigned int)sub_22DADF0(**v9) )
      {
        v8 = *v6;
        ++a5;
        ++v6;
        *(a5 - 1) = v8;
        if ( a2 == v7 )
          break;
      }
      else
      {
        v11 = *v7;
        ++a5;
        ++v7;
        *(a5 - 1) = v11;
        if ( a2 == v7 )
          break;
      }
    }
    while ( a4 != v6 );
  }
  if ( a2 != v7 )
    memmove(a5, v7, (char *)a2 - (char *)v7);
  v12 = (char *)a5 + (char *)a2 - (char *)v7;
  if ( a4 != v6 )
    v12 = (char *)memmove(v12, v6, (char *)a4 - (char *)v6);
  return &v12[(char *)a4 - (char *)v6];
}
