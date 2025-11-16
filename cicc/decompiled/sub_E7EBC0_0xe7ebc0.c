// Function: sub_E7EBC0
// Address: 0xe7ebc0
//
__int64 __fastcall sub_E7EBC0(__int64 a1, int **a2)
{
  __int64 v2; // r12
  int *v3; // rbx
  __int64 v4; // r14
  int v5; // eax
  __int64 v6; // rdi
  __int64 v7; // r13
  __int64 v9; // r13

  v2 = 0;
  v3 = *a2;
  v4 = (__int64)&(*a2)[12 * *((unsigned int *)a2 + 2)];
  if ( (int *)v4 != *a2 )
  {
    do
    {
      while ( 1 )
      {
        v5 = *v3;
        if ( *v3 != 2 )
          break;
        v2 += (unsigned int)sub_F03EF0((unsigned int)v3[1]) + *((_QWORD *)v3 + 3) + 1LL;
LABEL_5:
        v3 += 12;
        if ( (int *)v4 == v3 )
          return v2;
      }
      if ( v5 != 3 )
      {
        if ( v5 == 1 )
        {
          v9 = (unsigned int)sub_F03EF0((unsigned int)v3[1]);
          v2 += (unsigned int)sub_F03EF0((unsigned int)v3[2]) + v9;
        }
        goto LABEL_5;
      }
      v6 = (unsigned int)v3[1];
      v3 += 12;
      v7 = (unsigned int)sub_F03EF0(v6);
      v2 += (unsigned int)sub_F03EF0((unsigned int)*(v3 - 10)) + v7 + *((_QWORD *)v3 - 3) + 1LL;
    }
    while ( (int *)v4 != v3 );
  }
  return v2;
}
