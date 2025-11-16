// Function: sub_1DCBEC0
// Address: 0x1dcbec0
//
unsigned __int64 __fastcall sub_1DCBEC0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  unsigned __int64 result; // rax
  __int64 i; // rdi
  __int64 v8; // rcx
  __int64 v9; // [rsp+0h] [rbp-40h] BYREF
  __int64 v10; // [rsp+8h] [rbp-38h]
  __int64 v11; // [rsp+10h] [rbp-30h]

  v9 = 0;
  v10 = 0;
  v11 = 0;
  result = sub_1DCBB90(a1, a2, a3, a4, (__int64)&v9);
  for ( i = v10; v9 != v10; i = v10 )
  {
    v8 = *(_QWORD *)(i - 8);
    v10 = i - 8;
    result = sub_1DCBB90(a1, a2, a3, v8, (__int64)&v9);
  }
  if ( i )
    return j_j___libc_free_0(i, v11 - i);
  return result;
}
