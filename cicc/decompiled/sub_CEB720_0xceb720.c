// Function: sub_CEB720
// Address: 0xceb720
//
__int16 __fastcall sub_CEB720(__int64 a1, _BYTE *a2, __int64 a3)
{
  __int64 v3; // rdx
  char *v4; // rcx
  __int16 result; // ax
  __int64 v6[2]; // [rsp+0h] [rbp-30h] BYREF
  _QWORD v7[4]; // [rsp+10h] [rbp-20h] BYREF

  v6[0] = (__int64)v7;
  sub_CEB5A0(v6, a2, (__int64)&a2[a3]);
  result = sub_CEB590(v6, 0, v3, v4);
  if ( (_QWORD *)v6[0] != v7 )
    return j_j___libc_free_0(v6[0], v7[0] + 1LL);
  return result;
}
