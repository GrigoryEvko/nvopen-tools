// Function: sub_E10040
// Address: 0xe10040
//
__int64 __fastcall sub_E10040(__int64 a1, __int64 *a2)
{
  char *v3; // rsi
  unsigned __int64 v4; // rax
  char *v5; // rdi
  unsigned __int64 v6; // rsi
  unsigned __int64 v7; // rax
  __int64 v8; // rax
  char *v10; // rdi

  v3 = (char *)a2[1];
  v4 = a2[2];
  v5 = (char *)*a2;
  if ( (unsigned __int64)(v3 + 9) > v4 )
  {
    v6 = (unsigned __int64)(v3 + 1001);
    v7 = 2 * v4;
    if ( v6 > v7 )
      a2[2] = v6;
    else
      a2[2] = v7;
    v8 = realloc(v5);
    *a2 = v8;
    v5 = (char *)v8;
    if ( !v8 )
      abort();
    v3 = (char *)a2[1];
  }
  v10 = &v5[(_QWORD)v3];
  *(_QWORD *)v10 = 0x656D616E65707974LL;
  v10[8] = 32;
  a2[1] += 9;
  return 0x656D616E65707974LL;
}
