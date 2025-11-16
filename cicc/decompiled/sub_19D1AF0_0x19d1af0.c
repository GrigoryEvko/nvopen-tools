// Function: sub_19D1AF0
// Address: 0x19d1af0
//
__int64 __fastcall sub_19D1AF0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v5; // rax
  __int64 *v6; // rax
  _QWORD *v7; // rax
  bool v8; // zf
  __int64 v9; // rcx
  unsigned __int64 v10; // rdx
  __int64 v11; // rdx
  __m128i v12[3]; // [rsp+0h] [rbp-90h] BYREF
  __m128i v13; // [rsp+30h] [rbp-60h] BYREF
  __int64 v14; // [rsp+40h] [rbp-50h]

  if ( !*(_QWORD *)(a1 + 32) )
    sub_4263D6(a1, a2, a3);
  v3 = (*(__int64 (__fastcall **)(__int64))(a1 + 40))(a1 + 16);
  if ( (*(_BYTE *)(**(_QWORD **)(a1 + 8) + 72LL) & 0xC0) == 0 )
    return 0;
  sub_141F730(&v13, a2);
  sub_141F800(v12, a2);
  if ( (unsigned __int8)sub_134CB50(v3, (__int64)v12, (__int64)&v13) )
    return 0;
  v5 = *(_DWORD *)(a2 + 20) & 0xFFFFFFF;
  v13.m128i_i64[0] = **(_QWORD **)(a2 - 24 * v5);
  v13.m128i_i64[1] = **(_QWORD **)(a2 + 24 * (1 - v5));
  v14 = **(_QWORD **)(a2 + 24 * (2 - v5));
  v6 = (__int64 *)sub_15F2050(a2);
  v7 = (_QWORD *)sub_15E26F0(v6, 133, v13.m128i_i64, 3);
  v8 = *(_QWORD *)(a2 - 24) == 0;
  *(_QWORD *)(a2 + 64) = *(_QWORD *)(*v7 + 24LL);
  if ( !v8 )
  {
    v9 = *(_QWORD *)(a2 - 16);
    v10 = *(_QWORD *)(a2 - 8) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v10 = v9;
    if ( v9 )
      *(_QWORD *)(v9 + 16) = *(_QWORD *)(v9 + 16) & 3LL | v10;
  }
  *(_QWORD *)(a2 - 24) = v7;
  v11 = v7[1];
  *(_QWORD *)(a2 - 16) = v11;
  if ( v11 )
    *(_QWORD *)(v11 + 16) = (a2 - 16) | *(_QWORD *)(v11 + 16) & 3LL;
  *(_QWORD *)(a2 - 8) = (unsigned __int64)(v7 + 1) | *(_QWORD *)(a2 - 8) & 3LL;
  v7[1] = a2 - 24;
  sub_14191F0(*(_QWORD *)a1, a2);
  return 1;
}
