// Function: sub_17D0480
// Address: 0x17d0480
//
_QWORD *__fastcall sub_17D0480(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  __int64 v3; // r14
  __int64 *v4; // rax
  _QWORD *v5; // r13
  __int64 **v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r14
  __int64 v10; // rax
  __int64 *v11; // rax
  _QWORD *result; // rax
  __int64 v13[3]; // [rsp+0h] [rbp-70h] BYREF
  _QWORD *v14; // [rsp+18h] [rbp-58h]

  sub_17CE510((__int64)v13, a2, 0, 0, 0);
  v2 = *(_QWORD *)(a1 + 24);
  v3 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
  v4 = (__int64 *)sub_1643330(v14);
  v5 = (_QWORD *)sub_17CFB40(v2, v3, v13, v4, 8u);
  v6 = (__int64 **)sub_1643330(v14);
  v9 = sub_15A06D0(v6, v3, v7, v8);
  v10 = sub_1643360(v14);
  v11 = (__int64 *)sub_159C470(v10, 8, 0);
  result = sub_15E7280(v13, v5, v9, v11, 8u, 0, 0, 0, 0);
  if ( v13[0] )
    return (_QWORD *)sub_161E7C0((__int64)v13, v13[0]);
  return result;
}
