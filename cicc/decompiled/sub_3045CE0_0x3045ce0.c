// Function: sub_3045CE0
// Address: 0x3045ce0
//
char *__fastcall sub_3045CE0(__int64 a1, __int64 *a2, int a3)
{
  __int64 v3; // r12
  char *v4; // r12
  _QWORD *v6[2]; // [rsp+0h] [rbp-30h] BYREF
  __int64 v7; // [rsp+10h] [rbp-20h] BYREF

  v3 = *(_QWORD *)(a1 + 537008) + 539408LL;
  sub_3045B30((__int64)v6, a1, *a2, a3);
  v4 = sub_C94910(v3, v6[0], (size_t)v6[1]);
  if ( v6[0] != &v7 )
    j_j___libc_free_0((unsigned __int64)v6[0]);
  return v4;
}
