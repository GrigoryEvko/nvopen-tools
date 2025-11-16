// Function: sub_385B560
// Address: 0x385b560
//
char *__fastcall sub_385B560(unsigned int *src, unsigned int *a2, char *a3, __int64 a4, _QWORD *a5)
{
  __int64 v5; // rax
  unsigned int *v6; // r15
  __int64 v8; // r14
  __int64 v9; // rbx
  char *v10; // rdi
  __int64 v13; // [rsp+8h] [rbp-38h]

  v5 = a2 - src;
  v6 = src;
  v13 = 2 * a4;
  if ( 2 * a4 <= v5 )
  {
    v8 = 8 * a4;
    v9 = -1 * a4;
    do
    {
      v10 = (char *)v6;
      v6 = (unsigned int *)((char *)v6 + v8);
      a3 = sub_385B4A0(v10, (char *)&v6[v9], &v6[v9], v6, a3, a5);
      v5 = a2 - v6;
    }
    while ( v5 >= v13 );
  }
  if ( v5 > a4 )
    v5 = a4;
  return sub_385B4A0((char *)v6, (char *)&v6[v5], &v6[v5], a2, a3, a5);
}
