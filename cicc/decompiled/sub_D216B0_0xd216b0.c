// Function: sub_D216B0
// Address: 0xd216b0
//
__int64 __fastcall sub_D216B0(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  void (__fastcall *v5)(__m128i *, __int64, __int64); // rax
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __m128i v12; // [rsp+0h] [rbp-50h] BYREF
  void (__fastcall *v13)(__m128i *, __m128i *, __int64); // [rsp+10h] [rbp-40h]
  __int64 v14; // [rsp+18h] [rbp-38h]

  v5 = *(void (__fastcall **)(__m128i *, __int64, __int64))(a3 + 16);
  v13 = 0;
  if ( v5 )
  {
    v5(&v12, a3, 2);
    v14 = *(_QWORD *)(a3 + 24);
    v13 = *(void (__fastcall **)(__m128i *, __m128i *, __int64))(a3 + 16);
  }
  sub_D1D220(a1, a2 + 312, &v12);
  if ( v13 )
    v13(&v12, &v12, 3);
  sub_D21290(a1, (__int64)a4);
  sub_D1EE30(a1, a2, v7, v8, v9, v10);
  sub_D1FEC0(a1, a4);
  return a1;
}
