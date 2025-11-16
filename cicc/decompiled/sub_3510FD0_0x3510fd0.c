// Function: sub_3510FD0
// Address: 0x3510fd0
//
char *__fastcall sub_3510FD0(__int64 *src, __int64 *a2, __int64 *a3, __int64 *a4, _QWORD *a5, __int64 a6)
{
  __int64 *v7; // r13
  __int64 *v8; // r12
  __int64 v10; // rax
  unsigned __int64 v11; // rbx
  __int64 v12; // rax
  char *v13; // r8
  __int64 v16; // [rsp+18h] [rbp-38h]

  v7 = a3;
  v8 = src;
  if ( src != a2 && a3 != a4 )
  {
    do
    {
      v16 = *v7;
      v11 = sub_2F06CB0(*(_QWORD *)(a6 + 536), *v8);
      if ( v11 < sub_2F06CB0(*(_QWORD *)(a6 + 536), v16) )
      {
        v10 = *v7;
        ++a5;
        ++v7;
        *(a5 - 1) = v10;
        if ( v8 == a2 )
          break;
      }
      else
      {
        v12 = *v8;
        ++a5;
        ++v8;
        *(a5 - 1) = v12;
        if ( v8 == a2 )
          break;
      }
    }
    while ( v7 != a4 );
  }
  if ( a2 != v8 )
    memmove(a5, v8, (char *)a2 - (char *)v8);
  v13 = (char *)a5 + (char *)a2 - (char *)v8;
  if ( a4 != v7 )
    v13 = (char *)memmove(v13, v7, (char *)a4 - (char *)v7);
  return &v13[(char *)a4 - (char *)v7];
}
