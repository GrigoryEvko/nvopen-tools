// Function: sub_202F000
// Address: 0x202f000
//
_QWORD *__fastcall sub_202F000(__int64 *a1, __int64 a2)
{
  _QWORD *v2; // rdi
  unsigned int v3; // ecx
  __int64 v4; // r9
  _QWORD *v5; // r12
  __int64 v7; // [rsp+0h] [rbp-40h] BYREF
  int v8; // [rsp+8h] [rbp-38h]
  __int64 v9; // [rsp+10h] [rbp-30h]

  sub_1F40D10(
    (__int64)&v7,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v2 = (_QWORD *)a1[1];
  v3 = (unsigned __int8)v8;
  v7 = 0;
  v8 = 0;
  v5 = sub_1D2B300(v2, 0x30u, (__int64)&v7, v3, v9, v4);
  if ( v7 )
    sub_161E7C0((__int64)&v7, v7);
  return v5;
}
