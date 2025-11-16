// Function: sub_26093B0
// Address: 0x26093b0
//
void __fastcall sub_26093B0(__int64 a1)
{
  unsigned int *v1; // r13
  __int64 v2; // rax
  unsigned int *v3; // rbx
  __int64 v4; // r15
  char *v5; // rax
  unsigned __int64 v6; // r12

  v1 = *(unsigned int **)a1;
  v2 = 4LL * *(unsigned int *)(a1 + 8);
  v3 = (unsigned int *)(*(_QWORD *)a1 + v2);
  v4 = v2 >> 2;
  if ( v2 )
  {
    while ( 1 )
    {
      v5 = (char *)sub_2207800(4 * v4);
      v6 = (unsigned __int64)v5;
      if ( v5 )
        break;
      v4 >>= 1;
      if ( !v4 )
        goto LABEL_6;
    }
    sub_26092E0(v1, v3, v5, v4);
  }
  else
  {
LABEL_6:
    v6 = 0;
    sub_25FA340(v1, v3);
  }
  j_j___libc_free_0(v6);
}
