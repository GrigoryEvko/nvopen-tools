// Function: sub_D5CB60
// Address: 0xd5cb60
//
bool __fastcall sub_D5CB60(unsigned __int8 *a1, __int64 (__fastcall *a2)(__int64, __int64), __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // r12
  __int64 *v6; // rcx
  bool result; // al
  __m128i v8; // [rsp+0h] [rbp-40h] BYREF
  bool v9; // [rsp+18h] [rbp-28h]

  v4 = sub_D5BAA0(a1);
  if ( !v4 )
    return (sub_D5BB80(a1) & 3) != 0;
  v5 = v4;
  v6 = (__int64 *)a2(a3, v4);
  if ( *(_BYTE *)(**(_QWORD **)(*(_QWORD *)(v5 + 24) + 16LL) + 8LL) != 14 )
    return (sub_D5BB80(a1) & 3) != 0;
  sub_D5BC90(&v8, v5, 7u, v6);
  result = v9;
  if ( !v9 )
    return (sub_D5BB80(a1) & 3) != 0;
  return result;
}
