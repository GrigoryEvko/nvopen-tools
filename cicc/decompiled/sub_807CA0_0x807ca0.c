// Function: sub_807CA0
// Address: 0x807ca0
//
__int64 __fastcall sub_807CA0(__int64 *a1)
{
  __int64 v1; // r13
  __int64 v2; // rax
  __int64 v3; // rbx
  int v5; // [rsp+Ch] [rbp-94h] BYREF
  __m128i v6[2]; // [rsp+10h] [rbp-90h] BYREF
  __m128i v7[7]; // [rsp+30h] [rbp-70h] BYREF

  v1 = a1[7];
  v2 = *(_QWORD *)(v1 + 112);
  if ( v2 )
    v3 = *(_QWORD *)(v2 + 8);
  else
    v3 = sub_7E9260(*a1, v1, &v5);
  sub_7264E0((__int64)a1, 3);
  a1[7] = v3;
  sub_7E1780((__int64)a1, (__int64)v6);
  sub_7F9080(v3, (__int64)v7);
  *(_QWORD *)(v1 + 8) = v3;
  return sub_7FEC50(v1, v7, 0, 0, 0, 0, v6, 0, 0);
}
