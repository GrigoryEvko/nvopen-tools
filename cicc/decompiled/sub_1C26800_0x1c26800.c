// Function: sub_1C26800
// Address: 0x1c26800
//
void *__fastcall sub_1C26800(__int64 a1, _QWORD *a2, size_t *a3, char a4)
{
  void *v6; // rax
  size_t v7; // rdx
  void *v8; // rsi
  void *result; // rax
  void *src; // [rsp+0h] [rbp-80h] BYREF
  size_t n; // [rsp+8h] [rbp-78h]
  _QWORD v12[2]; // [rsp+10h] [rbp-70h] BYREF
  void *v13; // [rsp+20h] [rbp-60h] BYREF
  __int64 v14; // [rsp+28h] [rbp-58h]
  __int64 v15; // [rsp+30h] [rbp-50h]
  __int64 v16; // [rsp+38h] [rbp-48h]
  int v17; // [rsp+40h] [rbp-40h]
  void **p_src; // [rsp+48h] [rbp-38h]

  p_src = &src;
  src = v12;
  n = 0;
  v13 = &unk_49EFBE0;
  LOBYTE(v12[0]) = 0;
  v17 = 1;
  v16 = 0;
  v15 = 0;
  v14 = 0;
  sub_1C23B90(a1, (__int64)&v13, a4, 1);
  if ( v14 != v16 )
    sub_16E7BA0((__int64 *)&v13);
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
  result = sub_16E7BC0((__int64 *)&v13);
  if ( src != v12 )
    return (void *)j_j___libc_free_0(src, v12[0] + 1LL);
  return result;
}
