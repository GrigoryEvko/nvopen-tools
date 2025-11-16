// Function: sub_36DA4B0
// Address: 0x36da4b0
//
void __fastcall sub_36DA4B0(__int64 a1, __int64 a2)
{
  __int64 v4; // rsi
  __int64 v5; // rax
  unsigned __int16 v6; // dx
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int8 v10; // al
  int v11; // r13d
  char *v12; // rax
  __int64 v13; // rax
  unsigned __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // rdx
  __int64 v17; // rdi
  const __m128i *v18; // rdx
  __m128i v19; // xmm0
  _QWORD *v20; // rdi
  __int64 v21; // r9
  __int64 v22; // r13
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rax
  __int64 v27; // [rsp+0h] [rbp-90h] BYREF
  int v28; // [rsp+8h] [rbp-88h]
  __int64 v29; // [rsp+10h] [rbp-80h]
  __int64 v30; // [rsp+18h] [rbp-78h]
  _QWORD *v31[2]; // [rsp+20h] [rbp-70h] BYREF
  __int64 v32; // [rsp+30h] [rbp-60h] BYREF
  __m128i v33; // [rsp+40h] [rbp-50h] BYREF
  __int64 v34; // [rsp+50h] [rbp-40h]
  __int64 v35; // [rsp+58h] [rbp-38h]
  __m128i v36; // [rsp+60h] [rbp-30h]

  v4 = *(_QWORD *)(a2 + 80);
  v27 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v27, v4, 1);
  v28 = *(_DWORD *)(a2 + 72);
  v5 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 80LL) + 48LL)
     + 16LL * *(unsigned int *)(*(_QWORD *)(a2 + 40) + 88LL);
  v6 = *(_WORD *)v5;
  v7 = *(_QWORD *)(v5 + 8);
  v33.m128i_i16[0] = v6;
  v33.m128i_i64[1] = v7;
  if ( v6 )
  {
    if ( v6 == 1 || (unsigned __int16)(v6 - 504) <= 7u )
      BUG();
    v26 = 16LL * (v6 - 1);
    v9 = *(_QWORD *)&byte_444C4A0[v26];
    v10 = byte_444C4A0[v26 + 8];
  }
  else
  {
    v29 = sub_3007260((__int64)&v33);
    v30 = v8;
    v9 = v29;
    v10 = v30;
  }
  v33.m128i_i64[0] = v9;
  v33.m128i_i8[8] = v10;
  v11 = (sub_CA1930(&v33) == 64) + 1048;
  sub_36DA3C0((__int64 *)v31, *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a2 + 40) + 120LL) + 96LL));
  v12 = sub_C94910(*(_QWORD *)(a1 + 952) + 539408LL, v31[0], (size_t)v31[1]);
  v13 = sub_33F8870(*(_QWORD **)(a1 + 64), v12, 7u, 0, 0);
  v14 = *(_QWORD *)(a2 + 48);
  v15 = *(unsigned int *)(a2 + 68);
  v17 = v16;
  v18 = *(const __m128i **)(a2 + 40);
  v19 = _mm_loadu_si128(v18 + 5);
  v34 = v13;
  v35 = v17;
  v20 = *(_QWORD **)(a1 + 64);
  v33 = v19;
  v36 = _mm_loadu_si128(v18);
  v22 = sub_33E66D0(v20, v11, (__int64)&v27, v14, v15, v21, (unsigned __int64 *)&v33, 3);
  sub_34158F0(*(_QWORD *)(a1 + 64), a2, v22, v23, v24, v25);
  sub_3421DB0(v22);
  sub_33ECEA0(*(const __m128i **)(a1 + 64), a2);
  if ( v31[0] != &v32 )
    j_j___libc_free_0((unsigned __int64)v31[0]);
  if ( v27 )
    sub_B91220((__int64)&v27, v27);
}
