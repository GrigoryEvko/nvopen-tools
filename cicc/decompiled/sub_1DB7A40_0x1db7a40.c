// Function: sub_1DB7A40
// Address: 0x1db7a40
//
void __fastcall sub_1DB7A40(__int64 *a1, __int64 a2, __int64 *a3)
{
  __int64 v5; // rbx
  __int64 v6; // rdi
  __int64 v7; // rax

  v5 = a2;
  if ( a2 == a1[3] && a3 == a1 + 1 )
  {
    sub_1DB3580(a1[2]);
    a1[2] = 0;
    a1[3] = (__int64)a3;
    a1[4] = (__int64)a3;
    a1[5] = 0;
  }
  else if ( a3 != (__int64 *)a2 )
  {
    do
    {
      v6 = v5;
      v5 = sub_220EF30(v5);
      v7 = sub_220F330(v6, a1 + 1);
      j_j___libc_free_0(v7, 56);
      --a1[5];
    }
    while ( a3 != (__int64 *)v5 );
  }
}
