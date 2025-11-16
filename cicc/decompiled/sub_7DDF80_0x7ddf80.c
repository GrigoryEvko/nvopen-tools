// Function: sub_7DDF80
// Address: 0x7ddf80
//
__int64 __fastcall sub_7DDF80(_QWORD *a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi
  __int64 v4; // rax
  _BYTE v6[16]; // [rsp+0h] [rbp-B0h] BYREF
  int v7; // [rsp+10h] [rbp-A0h]
  _BYTE v8[32]; // [rsp+20h] [rbp-90h] BYREF
  _BYTE v9[112]; // [rsp+40h] [rbp-70h] BYREF

  sub_7E1740(a1[3]);
  sub_7FA330(v8, v8);
  v2 = a1[2];
  if ( v2 )
  {
    sub_7F9080(v2, v9);
    sub_7F9B80(v6);
    v3 = a1[4];
    v7 = 1;
    sub_7FEC50(v3, (unsigned int)v9, (unsigned int)v6, 0, 1, 0, (__int64)v8, 0, 0);
    *(_BYTE *)(a1[2] + 88LL) |= 4u;
  }
  sub_72CBE0();
  v4 = sub_7F8110("__exception_caught", 0, 0, 0, 0);
  return sub_7F88F0(v4, 0, 0, v8);
}
