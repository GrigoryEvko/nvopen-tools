// Function: sub_C3E660
// Address: 0xc3e660
//
__int64 __fastcall sub_C3E660(__int64 a1, __int64 a2)
{
  __int64 *v2; // r15
  void *v3; // rbx
  __int64 *v4; // rsi
  _QWORD *v6; // [rsp+0h] [rbp-50h] BYREF
  _QWORD *v7; // [rsp+8h] [rbp-48h]
  _QWORD *v8; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v9; // [rsp+18h] [rbp-38h]

  v2 = *(__int64 **)(a2 + 8);
  v3 = sub_C33340();
  if ( (void *)*v2 == v3 )
    sub_C3E660(&v8, v2);
  else
    sub_C3A850((__int64)&v8, v2);
  if ( v9 <= 0x40 )
  {
    v6 = v8;
  }
  else
  {
    v6 = (_QWORD *)*v8;
    j_j___libc_free_0_0(v8);
  }
  v4 = (__int64 *)(*(_QWORD *)(a2 + 8) + 24LL);
  if ( v3 == (void *)*v4 )
    sub_C3E660(&v8, v4);
  else
    sub_C3A850((__int64)&v8, v4);
  if ( v9 <= 0x40 )
  {
    v7 = v8;
  }
  else
  {
    v7 = (_QWORD *)*v8;
    j_j___libc_free_0_0(v8);
  }
  sub_C438E0(a1, 128, 2, &v6);
  return a1;
}
