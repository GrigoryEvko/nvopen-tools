// Function: sub_37982E0
// Address: 0x37982e0
//
unsigned __int8 *__fastcall sub_37982E0(__int64 *a1, __int64 a2)
{
  unsigned int v2; // ebx
  __int64 v5; // rsi
  const __m128i *v6; // rax
  __int64 v7; // rdx
  __m128i v8; // xmm0
  __int32 v9; // ecx
  __int64 v10; // rax
  unsigned __int16 *v11; // rcx
  unsigned __int16 *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // r8
  int v15; // eax
  __int64 v16; // rcx
  __int64 v17; // rsi
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r8
  int v22; // eax
  unsigned int v23; // edi
  __int128 v24; // rax
  __int64 v25; // r9
  __int64 v26; // rax
  unsigned int v27; // edx
  _QWORD *v28; // rdi
  __m128i v29; // xmm1
  int v30; // r9d
  __m128i v31; // rax
  __int32 v32; // r9d
  _DWORD *v33; // r15
  unsigned __int16 v34; // si
  unsigned int v35; // eax
  unsigned __int8 *v36; // r14
  __int64 v38; // rdx
  bool v39; // al
  __int64 v40; // [rsp+0h] [rbp-B0h]
  _QWORD *v41; // [rsp+8h] [rbp-A8h]
  __int64 v42; // [rsp+10h] [rbp-A0h]
  __int64 v43; // [rsp+18h] [rbp-98h]
  __int32 v44; // [rsp+24h] [rbp-8Ch]
  char v45; // [rsp+28h] [rbp-88h]
  __int16 v46; // [rsp+2Ah] [rbp-86h]
  __m128i v47; // [rsp+30h] [rbp-80h] BYREF
  __int64 v48; // [rsp+40h] [rbp-70h] BYREF
  int v49; // [rsp+48h] [rbp-68h]
  __m128i v50; // [rsp+50h] [rbp-60h] BYREF
  __m128i v51; // [rsp+60h] [rbp-50h] BYREF
  __int64 v52; // [rsp+70h] [rbp-40h]
  __int32 v53; // [rsp+78h] [rbp-38h]

  v5 = *(_QWORD *)(a2 + 80);
  v48 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v48, v5, 1);
  v49 = *(_DWORD *)(a2 + 72);
  v6 = *(const __m128i **)(a2 + 40);
  v7 = v6->m128i_i64[0];
  v8 = _mm_loadu_si128(v6);
  v43 = v6[2].m128i_i64[1];
  v9 = v6[3].m128i_i32[0];
  v10 = v6->m128i_u32[2];
  v47 = v8;
  v44 = v9;
  v11 = *(unsigned __int16 **)(a2 + 48);
  v12 = (unsigned __int16 *)(*(_QWORD *)(v7 + 48) + 16 * v10);
  v13 = *v12;
  v14 = *((_QWORD *)v12 + 1);
  v50.m128i_i16[0] = *v12;
  v50.m128i_i64[1] = v14;
  v15 = *v11;
  v16 = *((_QWORD *)v11 + 1);
  v51.m128i_i16[0] = v15;
  v51.m128i_i64[1] = v16;
  if ( (_WORD)v15 )
  {
    v42 = 0;
    LOWORD(v15) = word_4456580[v15 - 1];
  }
  else
  {
    v15 = sub_3009970((__int64)&v51, v5, v13, v16, v14);
    v14 = v50.m128i_i64[1];
    v42 = v13;
    HIWORD(v2) = HIWORD(v15);
    LOWORD(v13) = v50.m128i_i16[0];
  }
  LOWORD(v2) = v15;
  v17 = *a1;
  sub_2FE6CC0((__int64)&v51, *a1, *(_QWORD *)(a1[1] + 64), (unsigned __int16)v13, v14);
  if ( v51.m128i_i8[0] == 5 )
  {
    v26 = sub_37946F0((__int64)a1, v47.m128i_u64[0], v47.m128i_i64[1]);
  }
  else
  {
    if ( v50.m128i_i16[0] )
    {
      v21 = 0;
      LOWORD(v22) = word_4456580[v50.m128i_u16[0] - 1];
    }
    else
    {
      v22 = sub_3009970((__int64)&v50, v17, v18, v19, v20);
      v46 = HIWORD(v22);
      v21 = v38;
    }
    HIWORD(v23) = v46;
    v40 = v21;
    LOWORD(v23) = v22;
    v41 = (_QWORD *)a1[1];
    *(_QWORD *)&v24 = sub_3400EE0((__int64)v41, 0, (__int64)&v48, 0, v8);
    v26 = (__int64)sub_3406EB0(v41, 0x9Eu, (__int64)&v48, v23, v40, v25, *(_OWORD *)&v47, v24);
  }
  v47.m128i_i64[0] = v26;
  v28 = (_QWORD *)a1[1];
  v47.m128i_i64[1] = v27 | v47.m128i_i64[1] & 0xFFFFFFFF00000000LL;
  v29 = _mm_load_si128(&v47);
  v30 = *(_DWORD *)(a2 + 28);
  v52 = v43;
  v51 = v29;
  v53 = v44;
  v31.m128i_i64[0] = (__int64)sub_33FBA10(v28, 155, (__int64)&v48, 2, 0, v30, (__int64)&v51, 2);
  v32 = v31.m128i_i32[2];
  v33 = (_DWORD *)*a1;
  v51 = _mm_loadu_si128(&v50);
  if ( v50.m128i_i16[0] )
  {
    v34 = v50.m128i_i16[0] - 17;
    if ( (unsigned __int16)(v50.m128i_i16[0] - 10) > 6u && (unsigned __int16)(v50.m128i_i16[0] - 126) > 0x31u )
    {
      if ( v34 > 0xD3u )
      {
LABEL_13:
        v35 = v33[15];
        goto LABEL_14;
      }
      goto LABEL_20;
    }
    if ( v34 <= 0xD3u )
    {
LABEL_20:
      v35 = v33[17];
      goto LABEL_14;
    }
  }
  else
  {
    v47 = v31;
    v45 = sub_3007030((__int64)&v51);
    v39 = sub_30070B0((__int64)&v51);
    v32 = v47.m128i_i32[2];
    if ( v39 )
      goto LABEL_20;
    if ( !v45 )
      goto LABEL_13;
  }
  v35 = v33[16];
LABEL_14:
  if ( v35 > 2 )
    BUG();
  v36 = sub_33FAF80(a1[1], 215 - v35, (__int64)&v48, v2, v42, v32, v8);
  if ( v48 )
    sub_B91220((__int64)&v48, v48);
  return v36;
}
