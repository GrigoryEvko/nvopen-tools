// Function: sub_E76670
// Address: 0xe76670
//
_BYTE *__fastcall sub_E76670(__int64 a1, _QWORD *a2)
{
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
  __int64 v7; // rsi
  _BYTE *result; // rax
  _QWORD v9[3]; // [rsp+0h] [rbp-30h] BYREF
  _BYTE v10[24]; // [rsp+18h] [rbp-18h] BYREF

  (*(void (__fastcall **)(_QWORD *, _QWORD, _QWORD))(*a2 + 176LL))(
    a2,
    *(_QWORD *)(*(_QWORD *)(a2[1] + 168LL) + 104LL),
    0);
  sub_E765B0(v9, a1, v3, v4, v5, v6);
  v7 = v9[0];
  (*(void (__fastcall **)(_QWORD *, _QWORD, _QWORD))(*a2 + 520LL))(a2, v9[0], v9[1]);
  result = v10;
  if ( (_BYTE *)v9[0] != v10 )
    return (_BYTE *)_libc_free(v9[0], v7);
  return result;
}
