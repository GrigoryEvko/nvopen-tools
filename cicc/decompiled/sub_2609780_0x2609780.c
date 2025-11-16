// Function: sub_2609780
// Address: 0x2609780
//
void __fastcall sub_2609780(unsigned int **a1)
{
  unsigned int *v1; // r13
  unsigned int *v2; // r14
  __int64 v3; // rbx
  char *v4; // rax
  unsigned __int64 v5; // r12

  v1 = a1[1];
  v2 = *a1;
  v3 = v1 - *a1;
  if ( (char *)v1 - (char *)*a1 <= 0 )
  {
LABEL_6:
    v5 = 0;
    sub_25FB820(v2, v1);
  }
  else
  {
    while ( 1 )
    {
      v4 = (char *)sub_2207800(4 * v3);
      v5 = (unsigned __int64)v4;
      if ( v4 )
        break;
      v3 >>= 1;
      if ( !v3 )
        goto LABEL_6;
    }
    sub_26096B0(v2, v1, v4, v3);
  }
  j_j___libc_free_0(v5);
}
