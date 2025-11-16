// Function: sub_2FD1370
// Address: 0x2fd1370
//
void __fastcall sub_2FD1370(char **a1)
{
  char *v1; // r13
  char *v2; // r14
  __int64 v3; // rbx
  char *v4; // rax
  unsigned __int64 v5; // r12

  v1 = a1[1];
  v2 = *a1;
  v3 = (v1 - *a1) >> 3;
  if ( v1 - *a1 <= 0 )
  {
LABEL_6:
    v5 = 0;
    sub_2FCFDB0(v2, v1);
  }
  else
  {
    while ( 1 )
    {
      v4 = (char *)sub_2207800(8 * v3);
      v5 = (unsigned __int64)v4;
      if ( v4 )
        break;
      v3 >>= 1;
      if ( !v3 )
        goto LABEL_6;
    }
    sub_2FD12A0(v2, v1, v4, (char *)v3);
  }
  j_j___libc_free_0(v5);
}
