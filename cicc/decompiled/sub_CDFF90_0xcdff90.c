// Function: sub_CDFF90
// Address: 0xcdff90
//
void *__fastcall sub_CDFF90(__int64 a1, _QWORD *a2, size_t *a3, char a4)
{
  void *v6; // rax
  size_t v7; // rdx
  void *v8; // rsi
  void *result; // rax
  void *src; // [rsp+10h] [rbp-90h] BYREF
  size_t n; // [rsp+18h] [rbp-88h]
  _QWORD v13[2]; // [rsp+20h] [rbp-80h] BYREF
  __int64 v14[2]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v15; // [rsp+40h] [rbp-60h]
  __int64 v16; // [rsp+48h] [rbp-58h]
  __int64 v17; // [rsp+50h] [rbp-50h]
  __int64 v18; // [rsp+58h] [rbp-48h]
  void **p_src; // [rsp+60h] [rbp-40h]

  v18 = 0x100000000LL;
  p_src = &src;
  v14[0] = (__int64)&unk_49DD210;
  src = v13;
  n = 0;
  LOBYTE(v13[0]) = 0;
  v14[1] = 0;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  sub_CB5980((__int64)v14, 0, 0, 0);
  sub_CDD2D0(a1, (__int64)v14, a4, 1);
  if ( v15 != v17 )
    sub_CB5AE0(v14);
  if ( a2 )
  {
    v6 = (void *)(**(__int64 (__fastcall ***)(_QWORD, size_t))(a1 + 24))(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 16LL), n);
    v7 = n;
    v8 = src;
    *a2 = v6;
    memcpy(v6, v8, v7);
  }
  if ( a3 )
    *a3 = n;
  v14[0] = (__int64)&unk_49DD210;
  result = sub_CB5840((__int64)v14);
  if ( src != v13 )
    return (void *)j_j___libc_free_0(src, v13[0] + 1LL);
  return result;
}
