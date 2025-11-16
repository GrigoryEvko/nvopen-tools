// Function: sub_B826F0
// Address: 0xb826f0
//
__int64 __fastcall sub_B826F0(__int64 a1, __int64 a2)
{
  _BYTE *v3; // rsi
  __int64 v4; // rdx
  _BYTE *v5; // rsi
  __int64 v6; // rdx
  __int64 v8[2]; // [rsp+0h] [rbp-80h] BYREF
  _QWORD v9[2]; // [rsp+10h] [rbp-70h] BYREF
  _QWORD *v10; // [rsp+20h] [rbp-60h] BYREF
  _QWORD v11[2]; // [rsp+30h] [rbp-50h] BYREF
  __m128i v12; // [rsp+40h] [rbp-40h]

  v3 = *(_BYTE **)a2;
  v4 = *(_QWORD *)(a2 + 8);
  v8[0] = (__int64)v9;
  sub_B7F1D0(v8, v3, (__int64)&v3[v4]);
  v5 = *(_BYTE **)(a2 + 32);
  v6 = *(_QWORD *)(a2 + 40);
  v10 = v11;
  sub_B7F1D0((__int64 *)&v10, v5, (__int64)&v5[v6]);
  v12 = _mm_loadu_si128((const __m128i *)(a2 + 64));
  sub_B180C0(a1, (unsigned __int64)v8);
  if ( v10 != v11 )
    j_j___libc_free_0(v10, v11[0] + 1LL);
  if ( (_QWORD *)v8[0] != v9 )
    j_j___libc_free_0(v8[0], v9[0] + 1LL);
  return a1;
}
