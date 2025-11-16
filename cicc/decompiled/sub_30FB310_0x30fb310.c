// Function: sub_30FB310
// Address: 0x30fb310
//
void __fastcall sub_30FB310(__int64 a1, __int64 a2)
{
  _BYTE *v3; // rsi
  __int64 v4; // rdx
  _BYTE *v5; // rsi
  __int64 v6; // rdx
  __int64 v7[2]; // [rsp+0h] [rbp-80h] BYREF
  _QWORD v8[2]; // [rsp+10h] [rbp-70h] BYREF
  _QWORD *v9; // [rsp+20h] [rbp-60h] BYREF
  _QWORD v10[2]; // [rsp+30h] [rbp-50h] BYREF
  __m128i v11; // [rsp+40h] [rbp-40h]

  v3 = *(_BYTE **)a2;
  v4 = *(_QWORD *)(a2 + 8);
  v7[0] = (__int64)v8;
  sub_30FA730(v7, v3, (__int64)&v3[v4]);
  v5 = *(_BYTE **)(a2 + 32);
  v6 = *(_QWORD *)(a2 + 40);
  v9 = v10;
  sub_30FA730((__int64 *)&v9, v5, (__int64)&v5[v6]);
  v11 = _mm_loadu_si128((const __m128i *)(a2 + 64));
  sub_B180C0(a1, (unsigned __int64)v7);
  if ( v9 != v10 )
    j_j___libc_free_0((unsigned __int64)v9);
  if ( (_QWORD *)v7[0] != v8 )
    j_j___libc_free_0(v7[0]);
}
