// Function: sub_198E7E0
// Address: 0x198e7e0
//
char *__fastcall sub_198E7E0(__int64 *src, __int64 *a2, __int64 *a3, __int64 *a4, _QWORD *a5, __int64 *a6)
{
  __int64 *v6; // r15
  __int64 *v8; // r12
  __int64 v10; // rax
  unsigned __int64 v11; // rbx
  __int64 v12; // rax
  char *v13; // r8
  __int64 v16; // [rsp+18h] [rbp-38h]

  v6 = a3;
  v8 = src;
  if ( src != a2 && a3 != a4 )
  {
    do
    {
      v16 = *v6;
      v11 = sub_1368AA0(a6, *v8);
      if ( v11 > sub_1368AA0(a6, v16) )
      {
        v10 = *v6;
        ++a5;
        ++v6;
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
    while ( v6 != a4 );
  }
  if ( a2 != v8 )
    memmove(a5, v8, (char *)a2 - (char *)v8);
  v13 = (char *)a5 + (char *)a2 - (char *)v8;
  if ( a4 != v6 )
    v13 = (char *)memmove(v13, v6, (char *)a4 - (char *)v6);
  return &v13[(char *)a4 - (char *)v6];
}
