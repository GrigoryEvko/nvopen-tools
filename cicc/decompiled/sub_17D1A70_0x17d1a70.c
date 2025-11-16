// Function: sub_17D1A70
// Address: 0x17d1a70
//
unsigned __int16 __fastcall sub_17D1A70(__int64 a1, __int64 a2)
{
  unsigned __int16 result; // ax
  __int64 v3; // r12
  __int64 v4; // r14
  __int64 *v5; // rax
  _QWORD *v6; // r12
  __int64 **v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r14
  __int64 v11; // rax
  __int64 *v12; // rax
  __int64 v13[3]; // [rsp-78h] [rbp-78h] BYREF
  _QWORD *v14; // [rsp-60h] [rbp-60h]

  result = (*(_WORD *)(*(_QWORD *)(a1 + 8) + 18LL) >> 4) & 0x3FF;
  if ( result != 79 )
  {
    sub_17CE510((__int64)v13, a2, 0, 0, 0);
    v3 = *(_QWORD *)(a1 + 24);
    v4 = *(_QWORD *)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF));
    v5 = (__int64 *)sub_1643330(v14);
    v6 = (_QWORD *)sub_17CFB40(v3, v4, v13, v5, 8u);
    v7 = (__int64 **)sub_1643330(v14);
    v10 = sub_15A06D0(v7, v4, v8, v9);
    v11 = sub_1643360(v14);
    v12 = (__int64 *)sub_159C470(v11, 24, 0);
    result = (unsigned __int16)sub_15E7280(v13, v6, v10, v12, 8u, 0, 0, 0, 0);
    if ( v13[0] )
      return sub_161E7C0((__int64)v13, v13[0]);
  }
  return result;
}
