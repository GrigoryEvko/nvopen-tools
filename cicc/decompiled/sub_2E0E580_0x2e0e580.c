// Function: sub_2E0E580
// Address: 0x2e0e580
//
void __fastcall sub_2E0E580(_QWORD *a1, __int64 a2, _QWORD *a3)
{
  __int64 v5; // rbx
  int *v6; // rdi
  int *v7; // rax

  v5 = a2;
  if ( a2 == a1[3] && a3 == a1 + 1 )
  {
    sub_2E094A0(a1[2]);
    a1[2] = 0;
    a1[3] = a3;
    a1[4] = a3;
    a1[5] = 0;
  }
  else if ( a3 != (_QWORD *)a2 )
  {
    do
    {
      v6 = (int *)v5;
      v5 = sub_220EF30(v5);
      v7 = sub_220F330(v6, a1 + 1);
      j_j___libc_free_0((unsigned __int64)v7);
      --a1[5];
    }
    while ( a3 != (_QWORD *)v5 );
  }
}
