// Function: sub_1441840
// Address: 0x1441840
//
unsigned int *__fastcall sub_1441840(unsigned int **a1, unsigned __int64 a2)
{
  unsigned int *v2; // r8
  __int64 v3; // rax
  __int64 v4; // rcx
  unsigned int *v5; // rdx

  v2 = *a1;
  v3 = 0xAAAAAAAAAAAAAAABLL * (((char *)a1[1] - (char *)*a1) >> 3);
  if ( (char *)a1[1] - (char *)*a1 > 0 )
  {
    do
    {
      while ( 1 )
      {
        v4 = v3 >> 1;
        v5 = &v2[2 * (v3 >> 1) + 2 * (v3 & 0xFFFFFFFFFFFFFFFELL)];
        if ( *v5 >= a2 )
          break;
        v2 = v5 + 6;
        v3 = v3 - v4 - 1;
        if ( v3 <= 0 )
          goto LABEL_6;
      }
      v3 >>= 1;
    }
    while ( v4 > 0 );
  }
LABEL_6:
  if ( a1[1] == v2 )
    sub_16BD130("Desired percentile exceeds the maximum cutoff", 1);
  return v2;
}
