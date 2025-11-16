// Function: sub_B97910
// Address: 0xb97910
//
__int64 __fastcall sub_B97910(__int64 a1, unsigned __int64 a2, int a3)
{
  unsigned __int64 v4; // rax
  __int64 v5; // r14
  __int64 v6; // rax
  char *v7; // rdi
  __int64 v8; // rbx

  if ( a2 > 0xF )
  {
    v5 = 32;
    v8 = sub_22077B0(a1 + 32);
    v7 = (char *)(v8 + 16);
  }
  else
  {
    v4 = 2LL * (a3 != 0);
    if ( v4 < a2 )
      v4 = a2;
    v5 = 8 * v4 + 16;
    v6 = sub_22077B0(v5 + a1);
    v7 = (char *)(v6 + v5 - 16);
    v8 = v6;
    if ( v6 + v5 == 16 )
      return v6 + v5;
  }
  sub_B977E0(v7, a2, a3);
  return v8 + v5;
}
