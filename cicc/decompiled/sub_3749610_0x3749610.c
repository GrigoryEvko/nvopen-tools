// Function: sub_3749610
// Address: 0x3749610
//
__int64 __fastcall sub_3749610(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // r13
  unsigned int v7; // r15d
  int v9; // r14d
  __int64 v10; // rsi
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 *v13; // r15
  __int64 v14; // r14
  _QWORD *v15; // rax
  __int64 v16; // r14
  __int64 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned int v23; // [rsp+4h] [rbp-7Ch]
  __int64 v24; // [rsp+8h] [rbp-78h]
  _QWORD *v25; // [rsp+18h] [rbp-68h]
  __m128i v26; // [rsp+20h] [rbp-60h] BYREF
  __int64 v27; // [rsp+30h] [rbp-50h]
  __int64 v28; // [rsp+38h] [rbp-48h]
  int v29; // [rsp+40h] [rbp-40h]

  v6 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)v6 == 25 )
  {
    v7 = 0;
    if ( !*(_QWORD *)(v6 + 64) )
    {
      v9 = *(unsigned __int8 *)(v6 + 96);
      if ( *(_BYTE *)(v6 + 97) )
        v9 |= 2u;
      if ( (unsigned __int8)sub_A73ED0((_QWORD *)(a2 + 72), 6) || (unsigned __int8)sub_B49560(a2, 6) )
        v9 |= 0x20u;
      v10 = a1[10];
      v23 = v9 | (4 * *(_DWORD *)(v6 + 100));
      v11 = a1[5];
      v12 = *(_QWORD *)(v11 + 744);
      v13 = *(__int64 **)(v11 + 752);
      v14 = *(_QWORD *)(a1[15] + 8) - 40LL;
      v15 = *(_QWORD **)(v12 + 32);
      v24 = v12;
      v26.m128i_i64[0] = v10;
      v25 = v15;
      if ( v10 )
        sub_B96E90((__int64)&v26, v10, 1);
      v16 = (__int64)sub_2E7B380(v25, v14, (unsigned __int8 **)&v26, 0);
      if ( v26.m128i_i64[0] )
        sub_B91220((__int64)&v26, v26.m128i_i64[0]);
      sub_2E31040((__int64 *)(v24 + 40), v16);
      v17 = *v13;
      v18 = *(_QWORD *)v16;
      *(_QWORD *)(v16 + 8) = v13;
      v17 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v16 = v17 | v18 & 7;
      *(_QWORD *)(v17 + 8) = v16;
      *v13 = v16 | *v13 & 7;
      v19 = a1[11];
      if ( v19 )
        sub_2E882B0(v16, (__int64)v25, v19);
      v20 = a1[12];
      if ( v20 )
        sub_2E88680(v16, (__int64)v25, v20);
      v21 = *(_QWORD *)(v6 + 24);
      v26.m128i_i8[0] = 9;
      v26.m128i_i32[0] &= 0xFFF000FF;
      v28 = v21;
      v27 = 0;
      v26.m128i_i32[2] = 0;
      v29 = 0;
      sub_2E8EAD0(v16, (__int64)v25, &v26);
      v26.m128i_i64[0] = 1;
      v27 = 0;
      v28 = v23;
      sub_2E8EAD0(v16, (__int64)v25, &v26);
      if ( *(_QWORD *)(a2 + 48) || (v7 = 1, (*(_BYTE *)(a2 + 7) & 0x20) != 0) )
      {
        v7 = 1;
        v22 = sub_B91F50(a2, "srcloc", 6u);
        if ( v22 )
        {
          v28 = v22;
          v26.m128i_i64[0] = 14;
          v27 = 0;
          sub_2E8EAD0(v16, (__int64)v25, &v26);
        }
      }
    }
    return v7;
  }
  else if ( !*(_BYTE *)v6 && *(_QWORD *)(v6 + 24) == *(_QWORD *)(a2 + 80) && (*(_BYTE *)(v6 + 33) & 0x20) != 0 )
  {
    return sub_3749240(a1, a2, a3, a4, a5);
  }
  else
  {
    return sub_3744640((__int64)a1, (unsigned __int8 *)a2);
  }
}
