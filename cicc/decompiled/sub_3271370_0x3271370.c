// Function: sub_3271370
// Address: 0x3271370
//
__int64 __fastcall sub_3271370(char a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  int v11; // eax
  char v13; // al
  int v14; // r9d
  __int64 v15; // rax
  __int16 v16; // cx
  __int64 v17; // rsi
  const __m128i *v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rdx
  __m128i v21; // xmm0
  __int64 v22; // rax
  __int64 v23; // rsi
  unsigned int v24; // eax
  unsigned __int16 v25; // dx
  int v26; // r9d
  __int64 v27; // r15
  __int64 v28; // rax
  __int64 v29; // rax
  unsigned int *v30; // rax
  __int64 v31; // rax
  bool v32; // al
  __int64 v33; // rcx
  __int64 v34; // r8
  unsigned __int16 v35; // ax
  __int64 v36; // rdx
  __int64 v37; // r11
  __int128 v38; // [rsp-20h] [rbp-B0h]
  int v39; // [rsp+18h] [rbp-78h]
  int v40; // [rsp+18h] [rbp-78h]
  int v41; // [rsp+20h] [rbp-70h]
  __int128 v42; // [rsp+20h] [rbp-70h]
  unsigned __int64 v43; // [rsp+28h] [rbp-68h]
  int v44; // [rsp+30h] [rbp-60h] BYREF
  __int64 v45; // [rsp+38h] [rbp-58h]
  unsigned __int16 v46; // [rsp+40h] [rbp-50h] BYREF
  __int64 v47; // [rsp+48h] [rbp-48h]

  v6 = a4;
  v11 = *(_DWORD *)(a4 + 24);
  if ( v11 == 214 )
  {
    v6 = **(_QWORD **)(a4 + 40);
    v11 = *(_DWORD *)(v6 + 24);
  }
  if ( v11 != 186 )
    return 0;
  v41 = a6;
  v13 = sub_33E0780(*(_QWORD *)(*(_QWORD *)(v6 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(v6 + 40) + 48LL), 0, a4, a5, a6);
  v14 = v41;
  if ( !v13 )
    return 0;
  v15 = *(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3;
  v16 = *(_WORD *)v15;
  v17 = *(_QWORD *)(v15 + 8);
  v18 = *(const __m128i **)(v6 + 40);
  v45 = v17;
  v19 = v18->m128i_u32[2];
  LOWORD(v44) = v16;
  v20 = v18->m128i_i64[0];
  v21 = _mm_loadu_si128(v18);
  v22 = *(_QWORD *)(v18->m128i_i64[0] + 48) + 16 * v19;
  v43 = v21.m128i_u64[1];
  if ( v16 != *(_WORD *)v22 || *(_QWORD *)(v22 + 8) != v17 && !v16 )
  {
    if ( *(_DWORD *)(v20 + 24) != 216 )
      return 0;
    v30 = *(unsigned int **)(v20 + 40);
    v19 = v30[2];
    v43 = v19 | v21.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    v20 = *(_QWORD *)v30;
    v31 = *(_QWORD *)(*(_QWORD *)v30 + 48LL) + 16 * v19;
    if ( v16 != *(_WORD *)v31 || !v16 && v17 != *(_QWORD *)(v31 + 8) )
      return 0;
  }
  *(_QWORD *)&v42 = v20;
  v23 = v20;
  v39 = v14;
  *((_QWORD *)&v42 + 1) = v19 | v43 & 0xFFFFFFFF00000000LL;
  v24 = sub_33D4D80(a5, v20, *((_QWORD *)&v42 + 1), 0);
  v25 = v44;
  v26 = v39;
  v27 = v24;
  if ( (_WORD)v44 )
  {
    if ( (unsigned __int16)(v44 - 17) > 0xD3u )
    {
LABEL_11:
      v28 = v45;
      goto LABEL_12;
    }
    v25 = word_4456580[(unsigned __int16)v44 - 1];
    v28 = 0;
  }
  else
  {
    v32 = sub_30070B0((__int64)&v44);
    v25 = 0;
    v26 = v39;
    if ( !v32 )
      goto LABEL_11;
    v35 = sub_3009970((__int64)&v44, v23, 0, v33, v34);
    v26 = v39;
    v37 = v36;
    v25 = v35;
    v28 = v37;
  }
LABEL_12:
  v46 = v25;
  v47 = v28;
  if ( v25 )
  {
    if ( v25 == 1 || (unsigned __int16)(v25 - 504) <= 7u )
      BUG();
    v29 = *(_QWORD *)&byte_444C4A0[16 * v25 - 16];
  }
  else
  {
    v40 = v26;
    v29 = sub_3007260((__int64)&v46);
    v26 = v40;
  }
  if ( v27 != v29 )
    return 0;
  *((_QWORD *)&v38 + 1) = a3;
  *(_QWORD *)&v38 = a2;
  return sub_3406EB0(a5, 56 - ((unsigned int)(a1 == 0) - 1), v26, v44, v45, v26, v38, v42);
}
