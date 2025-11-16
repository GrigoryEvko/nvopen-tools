// Function: sub_3816560
// Address: 0x3816560
//
unsigned __int8 *__fastcall sub_3816560(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 (__fastcall *v4)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v5; // rax
  unsigned __int16 v6; // si
  __int64 v7; // r8
  __int64 v8; // rax
  unsigned int v9; // eax
  __int64 v10; // r12
  _DWORD *v11; // r13
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned int v14; // eax
  __int64 v15; // rsi
  unsigned __int8 *v16; // r12
  __int64 v18; // rdx
  unsigned int v19; // [rsp+0h] [rbp-90h] BYREF
  __int64 v20; // [rsp+8h] [rbp-88h]
  __int64 v21; // [rsp+10h] [rbp-80h] BYREF
  int v22; // [rsp+18h] [rbp-78h]
  __int64 v23; // [rsp+20h] [rbp-70h] BYREF
  char v24; // [rsp+28h] [rbp-68h]
  unsigned __int64 v25; // [rsp+30h] [rbp-60h] BYREF
  unsigned int v26; // [rsp+38h] [rbp-58h]
  __int64 v27; // [rsp+40h] [rbp-50h] BYREF
  __int64 v28; // [rsp+48h] [rbp-48h]
  __int64 v29; // [rsp+50h] [rbp-40h]

  v4 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v5 = *(__int16 **)(a2 + 48);
  v6 = *v5;
  v7 = *((_QWORD *)v5 + 1);
  v8 = a1[1];
  if ( v4 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v27, *a1, *(_QWORD *)(v8 + 64), v6, v7);
    LOWORD(v9) = v28;
    LOWORD(v19) = v28;
    v20 = v29;
  }
  else
  {
    v9 = v4(*a1, *(_QWORD *)(v8 + 64), v6, v7);
    v19 = v9;
    v20 = v18;
  }
  v10 = a1[1];
  v11 = (_DWORD *)(*(_QWORD *)(**(_QWORD **)(a2 + 40) + 96LL) + 24LL);
  if ( (_WORD)v9 )
  {
    if ( (_WORD)v9 == 1 || (unsigned __int16)(v9 - 504) <= 7u )
      BUG();
    v13 = 16LL * ((unsigned __int16)v9 - 1);
    v12 = *(_QWORD *)&byte_444C4A0[v13];
    LOBYTE(v13) = byte_444C4A0[v13 + 8];
  }
  else
  {
    v12 = sub_3007260((__int64)&v19);
    v27 = v12;
    v28 = v13;
  }
  v24 = v13;
  v23 = v12;
  v14 = sub_CA1930(&v23);
  sub_C44830((__int64)&v25, v11, v14);
  v15 = *(_QWORD *)(a2 + 80);
  v21 = v15;
  if ( v15 )
    sub_B96E90((__int64)&v21, v15, 1);
  v22 = *(_DWORD *)(a2 + 72);
  v16 = sub_3401900(v10, (__int64)&v21, v19, v20, (__int64)&v25, 1, a3);
  if ( v21 )
    sub_B91220((__int64)&v21, v21);
  if ( v26 > 0x40 && v25 )
    j_j___libc_free_0_0(v25);
  return v16;
}
