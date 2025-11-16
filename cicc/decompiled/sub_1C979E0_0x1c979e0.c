// Function: sub_1C979E0
// Address: 0x1c979e0
//
_QWORD *__fastcall sub_1C979E0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rax
  __int64 v4; // r13
  __int64 v5; // r14
  _QWORD *result; // rax
  __int64 v7; // r12
  __int64 v8; // [rsp-10h] [rbp-80h]
  _QWORD v9[2]; // [rsp+0h] [rbp-70h] BYREF
  _QWORD v10[2]; // [rsp+10h] [rbp-60h] BYREF
  const char *v11[2]; // [rsp+20h] [rbp-50h] BYREF
  __int64 v12; // [rsp+30h] [rbp-40h] BYREF

  v2 = *(_QWORD *)(*(_QWORD *)(a1 + 40) + 56LL);
  if ( (unsigned __int8)sub_1C2F070(v2) )
  {
    LOBYTE(v10[0]) = 0;
    v9[0] = v10;
    v9[1] = 0;
    sub_15E0530(v2);
    sub_1C315E0((__int64)v11, (__int64 *)(a1 + 48));
    sub_2241490(v9, v11[0], v11[1]);
    if ( (__int64 *)v11[0] != &v12 )
      j_j___libc_free_0(v11[0], v12 + 1);
    sub_2241490(v9, *(const char **)a2, *(_QWORD *)(a2 + 8));
    sub_1C3F040((__int64)v9);
    if ( (_QWORD *)v9[0] != v10 )
      j_j___libc_free_0(v9[0], v10[0] + 1LL);
  }
  v3 = sub_15E26F0(*(__int64 **)(v2 + 40), 205, 0, 0);
  LOWORD(v12) = 257;
  v4 = v3;
  v5 = *(_QWORD *)(*(_QWORD *)v3 + 24LL);
  result = sub_1648AB0(72, 1u, 0);
  v7 = (__int64)result;
  if ( result )
  {
    sub_15F1EA0((__int64)result, **(_QWORD **)(v5 + 16), 54, (__int64)(result - 3), 1, a1);
    *(_QWORD *)(v7 + 56) = 0;
    sub_15F5B40(v7, v5, v4, 0, 0, (__int64)v11, 0, 0);
    return (_QWORD *)v8;
  }
  return result;
}
