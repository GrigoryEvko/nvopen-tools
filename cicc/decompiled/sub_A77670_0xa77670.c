// Function: sub_A77670
// Address: 0xa77670
//
__int64 __fastcall sub_A77670(__int64 a1, __int64 a2)
{
  const void *v2; // rax
  __int64 v3; // rdx
  int v5; // eax
  __int64 v6[5]; // [rsp+8h] [rbp-28h] BYREF

  v6[0] = a2;
  if ( sub_A71840((__int64)v6) )
  {
    v2 = (const void *)sub_A71FD0(v6);
    sub_A77520(a1 + 8, v2, v3, v6[0]);
  }
  else
  {
    v5 = sub_A71AE0(v6);
    sub_A77250(a1 + 8, v5, v6[0]);
  }
  return a1;
}
