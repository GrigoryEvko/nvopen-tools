// Function: sub_B3B200
// Address: 0xb3b200
//
__int64 __fastcall sub_B3B200(__int64 a1, char *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r14
  size_t v6; // rax
  __int64 v8[2]; // [rsp+0h] [rbp-50h] BYREF
  _QWORD v9[8]; // [rsp+10h] [rbp-40h] BYREF

  v8[0] = (__int64)v9;
  v5 = sub_2241E50(a1, a2, a3, a4, a5);
  v6 = strlen(a2);
  sub_B3B150(v8, a2, (__int64)&a2[v6]);
  sub_C63F00(a1, v8, 22, v5);
  if ( (_QWORD *)v8[0] != v9 )
    j_j___libc_free_0(v8[0], v9[0] + 1LL);
  return a1;
}
