// Function: sub_311E1B0
// Address: 0x311e1b0
//
char *__fastcall sub_311E1B0(
        unsigned __int64 **src,
        unsigned __int64 **a2,
        unsigned __int64 **a3,
        unsigned __int64 **a4,
        _QWORD *a5,
        __int64 a6)
{
  unsigned __int64 **v7; // r13
  unsigned __int64 **v8; // r12
  unsigned __int64 *v9; // rax
  unsigned __int64 *v10; // rax
  signed __int64 v11; // r15
  char *v12; // r8
  __int64 v15[7]; // [rsp+8h] [rbp-38h] BYREF

  v7 = a3;
  v8 = src;
  v15[0] = a6;
  if ( src != a2 && a3 != a4 )
  {
    do
    {
      if ( (unsigned __int8)sub_311D9B0(v15, v7, v8) )
      {
        v9 = *v7;
        ++a5;
        ++v7;
        *(a5 - 1) = v9;
        if ( v8 == a2 )
          break;
      }
      else
      {
        v10 = *v8++;
        *a5++ = v10;
        if ( v8 == a2 )
          break;
      }
    }
    while ( v7 != a4 );
  }
  v11 = (char *)a2 - (char *)v8;
  if ( a2 != v8 )
    memmove(a5, v8, (char *)a2 - (char *)v8);
  v12 = (char *)a5 + v11;
  if ( a4 != v7 )
    v12 = (char *)memmove((char *)a5 + v11, v7, (char *)a4 - (char *)v7);
  return &v12[(char *)a4 - (char *)v7];
}
