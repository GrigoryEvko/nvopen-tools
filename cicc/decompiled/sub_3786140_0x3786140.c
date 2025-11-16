// Function: sub_3786140
// Address: 0x3786140
//
__int64 __fastcall sub_3786140(__int64 a1, __int64 a2)
{
  unsigned int v3; // ebx
  __int64 v4; // r14
  const __m128i *v5; // rax
  __int64 v6; // rsi
  __int128 v7; // xmm0
  unsigned __int64 v8; // r9
  __int64 v9; // rcx
  int v10; // eax
  __int64 v11; // r9
  __int64 v12; // rax
  __int64 v13; // r15
  unsigned __int16 *v14; // rax
  int v15; // edx
  __int64 v16; // rax
  __int128 v17; // rax
  _QWORD *v18; // r12
  __int128 v19; // rax
  __int64 v20; // r14
  __int64 v22; // [rsp+8h] [rbp-B8h]
  __int128 v23; // [rsp+10h] [rbp-B0h]
  __int128 v24; // [rsp+20h] [rbp-A0h]
  unsigned __int64 v25; // [rsp+30h] [rbp-90h]
  __int64 v26; // [rsp+38h] [rbp-88h]
  unsigned int v27; // [rsp+38h] [rbp-88h]
  __int64 v28; // [rsp+50h] [rbp-70h] BYREF
  int v29; // [rsp+58h] [rbp-68h]
  __int128 v30; // [rsp+60h] [rbp-60h] BYREF
  __int128 v31; // [rsp+70h] [rbp-50h] BYREF
  __int16 v32; // [rsp+80h] [rbp-40h] BYREF
  __int64 v33; // [rsp+88h] [rbp-38h]

  v3 = **(unsigned __int16 **)(a2 + 48);
  v4 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
  v5 = *(const __m128i **)(a2 + 40);
  v6 = *(_QWORD *)(a2 + 80);
  v7 = (__int128)_mm_loadu_si128(v5);
  v28 = v6;
  v8 = v5[2].m128i_u64[1];
  v9 = v5[3].m128i_i64[0];
  v23 = (__int128)_mm_loadu_si128(v5 + 5);
  v26 = v5[5].m128i_i64[0];
  if ( v6 )
  {
    v22 = v5[3].m128i_i64[0];
    v25 = v5[2].m128i_u64[1];
    sub_B96E90((__int64)&v28, v6, 1);
    v9 = v22;
    v8 = v25;
  }
  v10 = *(_DWORD *)(a2 + 72);
  DWORD2(v30) = 0;
  v29 = v10;
  *(_QWORD *)&v30 = 0;
  *(_QWORD *)&v31 = 0;
  DWORD2(v31) = 0;
  sub_375E8D0(a1, v8, v9, (__int64)&v30, (__int64)&v31);
  v12 = *(_QWORD *)(v26 + 96);
  if ( *(_DWORD *)(v12 + 32) <= 0x40u )
    v13 = *(_QWORD *)(v12 + 24);
  else
    v13 = **(_QWORD **)(v12 + 24);
  v14 = (unsigned __int16 *)(*(_QWORD *)(v30 + 48) + 16LL * DWORD2(v30));
  v15 = *v14;
  v16 = *((_QWORD *)v14 + 1);
  v32 = v15;
  v33 = v16;
  if ( (_WORD)v15 )
    v27 = word_4456340[v15 - 1];
  else
    v27 = sub_3007240((__int64)&v32);
  *(_QWORD *)&v17 = sub_340F900(*(_QWORD **)(a1 + 8), 0xA0u, (__int64)&v28, v3, v4, v11, v7, v30, v23);
  v18 = *(_QWORD **)(a1 + 8);
  v24 = v17;
  *(_QWORD *)&v19 = sub_3400EE0((__int64)v18, v13 + v27, (__int64)&v28, 0, (__m128i)v7);
  v20 = sub_340F900(v18, 0xA0u, (__int64)&v28, v3, v4, *((__int64 *)&v24 + 1), v24, v31, v19);
  if ( v28 )
    sub_B91220((__int64)&v28, v28);
  return v20;
}
