// Function: sub_15E26F0
// Address: 0x15e26f0
//
__int64 __fastcall sub_15E26F0(__int64 *a1, int a2, __int64 *a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v7; // r12
  __int64 v9[2]; // [rsp+0h] [rbp-50h] BYREF
  __int64 v10; // [rsp+10h] [rbp-40h] BYREF

  v6 = sub_15E1360(*a1, a2, (__int64)a3);
  sub_15E1070(v9, a2, a3, a4);
  v7 = sub_1632190(a1, v9[0], v9[1], v6);
  if ( (__int64 *)v9[0] != &v10 )
    j_j___libc_free_0(v9[0], v10 + 1);
  return v7;
}
