// Function: sub_2128640
// Address: 0x2128640
//
_QWORD *__fastcall sub_2128640(__int64 *a1, __int64 a2)
{
  _QWORD *v2; // r12
  unsigned int v3; // ecx
  __int64 v4; // r9
  _QWORD *v5; // r12
  __int64 v7; // [rsp+0h] [rbp-40h] BYREF
  int v8; // [rsp+8h] [rbp-38h]
  __int64 v9; // [rsp+10h] [rbp-30h]

  v2 = (_QWORD *)a1[1];
  sub_1F40D10((__int64)&v7, *a1, v2[6], **(unsigned __int8 **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v3 = (unsigned __int8)v8;
  v7 = 0;
  v8 = 0;
  v5 = sub_1D2B300(v2, 0x30u, (__int64)&v7, v3, v9, v4);
  if ( v7 )
    sub_161E7C0((__int64)&v7, v7);
  return v5;
}
