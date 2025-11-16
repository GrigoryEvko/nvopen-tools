// Function: sub_38319D0
// Address: 0x38319d0
//
unsigned __int8 *__fastcall sub_38319D0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r14d
  __int64 v4; // rsi
  const __m128i *v5; // rax
  __m128i v6; // xmm0
  __int64 v7; // r15
  __int64 v8; // r8
  unsigned __int16 *v9; // r15
  __int64 v10; // rdx
  __int64 v11; // rdx
  int v12; // eax
  __int64 v13; // rcx
  unsigned __int64 v14; // rsi
  unsigned __int16 *v15; // rdx
  int v16; // eax
  __int64 v17; // rdx
  unsigned __int16 v18; // ax
  int v19; // eax
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // r15
  unsigned int v23; // edx
  unsigned int v24; // edx
  __int64 v25; // r9
  __int64 v26; // rax
  unsigned int v27; // edx
  unsigned __int8 *v28; // r14
  __int64 v30; // rdx
  __int64 v31; // rdx
  unsigned int v32; // [rsp+8h] [rbp-C8h]
  char v33; // [rsp+8h] [rbp-C8h]
  __int64 v34; // [rsp+10h] [rbp-C0h]
  __int64 v35; // [rsp+18h] [rbp-B8h]
  __int64 *v36; // [rsp+18h] [rbp-B8h]
  __int128 v37; // [rsp+20h] [rbp-B0h]
  __int128 v38; // [rsp+30h] [rbp-A0h]
  unsigned __int8 *v39; // [rsp+50h] [rbp-80h]
  __int64 v40; // [rsp+60h] [rbp-70h]
  __int64 v41; // [rsp+70h] [rbp-60h] BYREF
  int v42; // [rsp+78h] [rbp-58h]
  __int16 v43; // [rsp+80h] [rbp-50h] BYREF
  __int64 v44; // [rsp+88h] [rbp-48h]
  __int16 v45; // [rsp+90h] [rbp-40h] BYREF
  __int64 v46; // [rsp+98h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 80);
  v41 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v41, v4, 1);
  v42 = *(_DWORD *)(a2 + 72);
  v5 = *(const __m128i **)(a2 + 40);
  v6 = _mm_loadu_si128(v5);
  v35 = v5->m128i_i64[0];
  v7 = v5->m128i_u32[2];
  *(_QWORD *)&v38 = sub_37AE0F0(a1, v5[2].m128i_u64[1], v5[3].m128i_i64[0]);
  v9 = (unsigned __int16 *)(*(_QWORD *)(v35 + 48) + 16 * v7);
  *((_QWORD *)&v38 + 1) = v10;
  v11 = *((_QWORD *)v9 + 1);
  v12 = *v9;
  v37 = (__int128)_mm_loadu_si128((const __m128i *)(*(_QWORD *)(a2 + 40) + 80LL));
  v45 = v12;
  v46 = v11;
  if ( (_WORD)v12 )
  {
    LOBYTE(v9) = (unsigned __int16)(v12 - 176) <= 0x34u;
    v13 = (unsigned int)v9;
    v14 = word_4456340[v12 - 1];
  }
  else
  {
    v14 = sub_3007240((__int64)&v45);
    v13 = HIDWORD(v14);
    LOBYTE(v9) = BYTE4(v14);
  }
  v15 = (unsigned __int16 *)(*(_QWORD *)(v38 + 48) + 16LL * DWORD2(v38));
  v16 = *v15;
  v17 = *((_QWORD *)v15 + 1);
  v43 = v16;
  v44 = v17;
  if ( (_WORD)v16 )
  {
    v34 = 0;
    v18 = word_4456580[v16 - 1];
  }
  else
  {
    v33 = v13;
    v18 = sub_3009970((__int64)&v43, v14, v17, v13, v8);
    LOBYTE(v13) = v33;
    v34 = v31;
  }
  LODWORD(v40) = v14;
  BYTE4(v40) = v13;
  v36 = *(__int64 **)(*(_QWORD *)(a1 + 8) + 64LL);
  v32 = v18;
  if ( (_BYTE)v9 )
  {
    LOWORD(v19) = sub_2D43AD0(v18, v14);
    v22 = 0;
    v23 = v32;
    if ( (_WORD)v19 )
      goto LABEL_9;
  }
  else
  {
    LOWORD(v19) = sub_2D43050(v18, v14);
    v22 = 0;
    v23 = v32;
    if ( (_WORD)v19 )
      goto LABEL_9;
  }
  v19 = sub_3009450(v36, v23, v34, v40, v20, v21);
  HIWORD(v2) = HIWORD(v19);
  v22 = v30;
LABEL_9:
  LOWORD(v2) = v19;
  v39 = sub_33FAFB0(*(_QWORD *)(a1 + 8), v6.m128i_i64[0], v6.m128i_u32[2], (__int64)&v41, v2, v22, v6);
  v26 = sub_340F900(
          *(_QWORD **)(a1 + 8),
          0xA0u,
          (__int64)&v41,
          v2,
          v22,
          v25,
          __PAIR128__(v24 | v6.m128i_i64[1] & 0xFFFFFFFF00000000LL, (unsigned __int64)v39),
          v38,
          v37);
  v28 = sub_33FAFB0(
          *(_QWORD *)(a1 + 8),
          v26,
          v27,
          (__int64)&v41,
          **(unsigned __int16 **)(a2 + 48),
          *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
          v6);
  if ( v41 )
    sub_B91220((__int64)&v41, v41);
  return v28;
}
