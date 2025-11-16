// Function: sub_2127A00
// Address: 0x2127a00
//
__int64 *__fastcall sub_2127A00(__int64 *a1, unsigned __int64 a2)
{
  __int64 v3; // r11
  __int64 v4; // r10
  __int64 v5; // rsi
  _QWORD *v6; // r14
  __int64 v7; // r9
  __int128 *v8; // rcx
  __int64 v9; // r8
  unsigned __int8 v10; // bl
  unsigned __int16 v11; // si
  const __m128i *v12; // r9
  __int64 *v13; // r14
  __int64 v15; // [rsp+0h] [rbp-80h]
  __int64 v16; // [rsp+8h] [rbp-78h]
  __int64 v17; // [rsp+18h] [rbp-68h]
  __int128 *v18; // [rsp+20h] [rbp-60h]
  __int64 v19; // [rsp+28h] [rbp-58h]
  __int64 v20; // [rsp+30h] [rbp-50h] BYREF
  int v21; // [rsp+38h] [rbp-48h]
  __int64 v22; // [rsp+40h] [rbp-40h]

  sub_1F40D10(
    (__int64)&v20,
    *a1,
    *(_QWORD *)(a1[1] + 48),
    **(unsigned __int8 **)(a2 + 40),
    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v3 = v22;
  v4 = (unsigned __int8)v21;
  v5 = *(_QWORD *)(a2 + 72);
  v6 = (_QWORD *)a1[1];
  v7 = *(_QWORD *)(a2 + 104);
  v8 = *(__int128 **)(a2 + 32);
  v9 = *(_QWORD *)(a2 + 96);
  v10 = *(_BYTE *)(a2 + 88);
  v20 = v5;
  if ( v5 )
  {
    v15 = (unsigned __int8)v21;
    v16 = v22;
    v17 = v7;
    v18 = v8;
    v19 = v9;
    sub_1623A60((__int64)&v20, v5, 2);
    v4 = v15;
    v3 = v16;
    v7 = v17;
    v8 = v18;
    v9 = v19;
  }
  v11 = *(_WORD *)(a2 + 24);
  v21 = *(_DWORD *)(a2 + 64);
  v13 = sub_1D25480(v6, v11, (__int64)&v20, v10, v9, v7, v4, v3, *v8, *(__int128 *)((char *)v8 + 40));
  if ( v20 )
    sub_161E7C0((__int64)&v20, v20);
  sub_2013400((__int64)a1, a2, 1, (__int64)v13, (__m128i *)1, v12);
  return v13;
}
