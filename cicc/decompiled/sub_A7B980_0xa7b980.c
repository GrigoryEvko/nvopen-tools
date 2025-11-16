// Function: sub_A7B980
// Address: 0xa7b980
//
unsigned __int64 __fastcall sub_A7B980(__int64 *a1, __int64 *a2, int a3, int a4)
{
  unsigned __int64 v6; // rax
  __int64 v8[5]; // [rsp+8h] [rbp-28h] BYREF

  v8[0] = sub_A74490(a1, a3);
  v6 = sub_A7B8F0(v8, a2, a4);
  if ( v8[0] == v6 )
    return *a1;
  else
    return sub_A78500(a1, (unsigned __int64 *)a2, a3, v6);
}
