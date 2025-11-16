// Function: sub_94AAF0
// Address: 0x94aaf0
//
__int64 __fastcall sub_94AAF0(unsigned int **a1, __int64 a2)
{
  unsigned int *v2; // rbx
  __int64 v3; // r12
  __int64 v5; // rdx
  __int64 v6; // rsi
  __int64 result; // rax

  v2 = *a1;
  v3 = (__int64)&(*a1)[4 * *((unsigned int *)a1 + 2)];
  if ( (unsigned int *)v3 != *a1 )
  {
    do
    {
      v5 = *((_QWORD *)v2 + 1);
      v6 = *v2;
      v2 += 4;
      result = sub_B99FD0(a2, v6, v5);
    }
    while ( (unsigned int *)v3 != v2 );
  }
  return result;
}
