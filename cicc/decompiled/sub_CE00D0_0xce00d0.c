// Function: sub_CE00D0
// Address: 0xce00d0
//
_QWORD *__fastcall sub_CE00D0(__int64 a1)
{
  _QWORD *result; // rax
  __int64 v2; // r12
  size_t v3; // r15
  _QWORD *v4; // rbx
  void *src; // [rsp+0h] [rbp-90h] BYREF
  size_t n; // [rsp+8h] [rbp-88h]
  _QWORD v7[2]; // [rsp+10h] [rbp-80h] BYREF
  __int64 v8[2]; // [rsp+20h] [rbp-70h] BYREF
  __int64 v9; // [rsp+30h] [rbp-60h]
  __int64 v10; // [rsp+38h] [rbp-58h]
  __int64 v11; // [rsp+40h] [rbp-50h]
  __int64 v12; // [rsp+48h] [rbp-48h]
  void **p_src; // [rsp+50h] [rbp-40h]

  result = *(_QWORD **)(a1 + 32);
  v2 = result[26];
  if ( v2 && *(_QWORD *)(v2 + 80) )
  {
    p_src = &src;
    v12 = 0x100000000LL;
    src = v7;
    n = 0;
    v8[0] = (__int64)&unk_49DD210;
    LOBYTE(v7[0]) = 0;
    v8[1] = 0;
    v9 = 0;
    v10 = 0;
    v11 = 0;
    sub_CB5980((__int64)v8, 0, 0, 0);
    sub_CDD2D0(a1, (__int64)v8, 0, 1);
    if ( v11 != v9 )
      sub_CB5AE0(v8);
    v3 = n;
    v4 = *(_QWORD **)(v2 + 80);
    *v4 = (**(__int64 (__fastcall ***)(_QWORD, size_t))(a1 + 24))(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL), n + 1);
    memcpy(**(void ***)(v2 + 80), src, v3);
    *(_BYTE *)(**(_QWORD **)(v2 + 80) + v3) = 0;
    v8[0] = (__int64)&unk_49DD210;
    result = sub_CB5840((__int64)v8);
    if ( src != v7 )
      return (_QWORD *)j_j___libc_free_0(src, v7[0] + 1LL);
  }
  return result;
}
