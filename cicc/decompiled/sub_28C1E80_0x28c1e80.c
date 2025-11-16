// Function: sub_28C1E80
// Address: 0x28c1e80
//
__int64 *__fastcall sub_28C1E80(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 a4)
{
  int v4; // eax
  __int64 *v5; // rdi
  __int64 *v6; // rax
  unsigned __int64 v7; // rdi
  __int64 *v8; // r12
  __int64 *result; // rax
  __int64 *v10; // rdi
  __int64 *v11; // [rsp+0h] [rbp-30h] BYREF
  __int64 v12; // [rsp+8h] [rbp-28h]
  __int64 v13; // [rsp+10h] [rbp-20h] BYREF
  __int64 v14; // [rsp+18h] [rbp-18h]

  v4 = *a2;
  if ( v4 == 42 )
  {
    v5 = *(__int64 **)(a1 + 24);
    v13 = a3;
    v14 = a4;
    v11 = &v13;
    v12 = 0x200000002LL;
    v6 = sub_DC7EB0(v5, (__int64)&v11, 0, 0);
    v7 = (unsigned __int64)v11;
    v8 = v6;
    if ( v11 == &v13 )
      return v8;
LABEL_3:
    _libc_free(v7);
    return v8;
  }
  if ( v4 != 46 )
    BUG();
  v10 = *(__int64 **)(a1 + 24);
  v13 = a3;
  v14 = a4;
  v11 = &v13;
  v12 = 0x200000002LL;
  result = sub_DC8BD0(v10, (__int64)&v11, 0, 0);
  v7 = (unsigned __int64)v11;
  v8 = result;
  if ( v11 != &v13 )
    goto LABEL_3;
  return result;
}
