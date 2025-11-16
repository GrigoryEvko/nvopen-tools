// Function: sub_879B10
// Address: 0x879b10
//
__int64 sub_879B10()
{
  __int64 v0; // rax
  __int64 result; // rax

  sub_666D40((__int64)"template<template<typename U, U... K> class S, typename T, T N>  struct __make_integer_seq;", 0);
  v0 = sub_879550("__make_integer_seq", 0, 0x80000u);
  *(_BYTE *)(*(_QWORD *)(v0 + 88) + 266LL) |= 0x20u;
  unk_4D049D8 = v0;
  sub_666D40(
    (__int64)"template<template<typename U, U... K> class S, typename T, T N>  __internal_alias_decl __make_integer_seq_alias = T;",
    0);
  result = sub_879550("__make_integer_seq_alias", 0, 0x80000u);
  *(_BYTE *)(*(_QWORD *)(result + 88) + 266LL) |= 0x20u;
  unk_4D049D0 = result;
  return result;
}
