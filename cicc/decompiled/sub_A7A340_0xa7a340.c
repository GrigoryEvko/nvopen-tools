// Function: sub_A7A340
// Address: 0xa7a340
//
unsigned __int64 __fastcall sub_A7A340(__int64 *a1, __int64 *a2, int a3, const void *a4, size_t a5)
{
  unsigned __int64 v8; // rax
  __int64 v10[7]; // [rsp+8h] [rbp-38h] BYREF

  v10[0] = sub_A74490(a1, a3);
  v8 = sub_A7A290(v10, a2, a4, a5);
  if ( v10[0] == v8 )
    return *a1;
  else
    return sub_A78500(a1, (unsigned __int64 *)a2, a3, v8);
}
