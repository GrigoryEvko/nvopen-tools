// Function: sub_1688BE0
// Address: 0x1688be0
//
char *__fastcall sub_1688BE0(__int64 a1, unsigned __int64 a2, int a3, int a4, int a5, int a6)
{
  unsigned __int64 *v6; // rdi
  unsigned __int64 v7; // r13
  unsigned __int64 v8; // r14
  char *v9; // rbx

  if ( a2 > 0xFFFFFFFFFFFFFFF7LL )
    return 0;
  v6 = (unsigned __int64 *)(a1 - 8);
  v7 = a2 + 8;
  v8 = *v6;
  v9 = realloc((unsigned __int64)v6, a2 + 8, a3, a4, a5, a6);
  if ( !v9 )
    return 0;
  if ( a2 < v8 )
    sub_1688BB0(v8 - v7);
  else
    sub_1688B00(v7 - v8);
  *(_QWORD *)v9 = v7;
  return v9 + 8;
}
