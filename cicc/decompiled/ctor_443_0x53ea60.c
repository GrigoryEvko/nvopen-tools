// Function: ctor_443
// Address: 0x53ea60
//
__int64 ctor_443()
{
  __int64 v0; // rax
  __int64 v1; // r12
  __int64 result; // rax
  _QWORD v3[2]; // [rsp+0h] [rbp-70h] BYREF
  _QWORD v4[2]; // [rsp+10h] [rbp-60h] BYREF
  _QWORD v5[2]; // [rsp+20h] [rbp-50h] BYREF
  _QWORD v6[8]; // [rsp+30h] [rbp-40h] BYREF

  v0 = sub_C60B10();
  v5[0] = v6;
  v1 = v0;
  sub_2752470(v5, "Controls which instructions are eliminated");
  v3[0] = v4;
  sub_2752470(v3, "dce-transform");
  result = sub_CF9810(v1, v3, v5);
  if ( (_QWORD *)v3[0] != v4 )
    result = j_j___libc_free_0(v3[0], v4[0] + 1LL);
  if ( (_QWORD *)v5[0] != v6 )
    return j_j___libc_free_0(v5[0], v6[0] + 1LL);
  return result;
}
