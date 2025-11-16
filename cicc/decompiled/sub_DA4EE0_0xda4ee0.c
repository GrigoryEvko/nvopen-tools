// Function: sub_DA4EE0
// Address: 0xda4ee0
//
char *__fastcall sub_DA4EE0(
        unsigned __int64 *src,
        unsigned __int64 *a2,
        unsigned __int64 *a3,
        unsigned __int64 *a4,
        _QWORD *a5,
        __int64 a6)
{
  unsigned __int64 *v6; // r15
  unsigned __int64 *v7; // r12
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rax
  signed __int64 v12; // r8
  char *v13; // rbx
  __int64 v16; // [rsp+18h] [rbp-38h]

  v6 = a3;
  v7 = src;
  if ( src != a2 && a3 != a4 )
  {
    do
    {
      v16 = sub_DA4700(*(_QWORD **)a6, **(_QWORD **)(a6 + 8), *v6, *v7, *(_QWORD *)(a6 + 16), 0);
      if ( BYTE4(v16) && (int)v16 < 0 )
      {
        v11 = *v6;
        ++a5;
        ++v6;
        *(a5 - 1) = v11;
        if ( v7 == a2 )
          break;
      }
      else
      {
        v10 = *v7++;
        *a5++ = v10;
        if ( v7 == a2 )
          break;
      }
    }
    while ( v6 != a4 );
  }
  v12 = (char *)a2 - (char *)v7;
  if ( a2 != v7 )
  {
    memmove(a5, v7, (char *)a2 - (char *)v7);
    v12 = (char *)a2 - (char *)v7;
  }
  v13 = (char *)a5 + v12;
  if ( a4 != v6 )
    memmove(v13, v6, (char *)a4 - (char *)v6);
  return &v13[(char *)a4 - (char *)v6];
}
