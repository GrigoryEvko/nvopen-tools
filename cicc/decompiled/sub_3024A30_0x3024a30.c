// Function: sub_3024A30
// Address: 0x3024a30
//
void __fastcall sub_3024A30(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r13
  _BYTE *v12; // rax
  __m128i *v13; // rdx
  __m128i si128; // xmm0
  _QWORD *v15; // rdx
  __int64 v16; // rax
  __int64 v17; // r14
  __int16 v18; // cx
  char v19; // cl
  int v20; // eax
  _WORD *v21; // rdx
  __int64 v22; // r14
  __int64 v23; // rdi
  _BYTE *v24; // rax
  __int64 v25; // rax
  __int64 v26; // rax
  const char *v27; // rsi
  __int64 v28; // rax
  size_t v29; // rdx
  signed __int64 v30; // r13
  __int64 v31; // rax
  char v32; // [rsp+7h] [rbp-59h]
  __int64 v33; // [rsp+8h] [rbp-58h]
  unsigned __int8 *v34; // [rsp+10h] [rbp-50h] BYREF
  size_t v35; // [rsp+18h] [rbp-48h]
  __int64 v36; // [rsp+20h] [rbp-40h] BYREF

  v7 = sub_31DA930();
  v11 = *(_QWORD *)(a2 + 24);
  v33 = v7;
  v12 = *(_BYTE **)(a3 + 32);
  if ( *(_BYTE **)(a3 + 24) == v12 )
  {
    sub_CB6200(a3, (unsigned __int8 *)".", 1u);
  }
  else
  {
    *v12 = 46;
    ++*(_QWORD *)(a3 + 32);
  }
  sub_3024560(a1, *(_DWORD *)(*(_QWORD *)(a2 + 8) + 8LL) >> 8, a3, v8, v9, v10);
  if ( (unsigned __int8)sub_CE8AD0((_BYTE *)a2) )
  {
    if ( *(_DWORD *)(a4 + 336) <= 0x27u || *(_DWORD *)(a4 + 340) <= 0x12Bu )
      sub_C64ED0(".attribute(.managed) requires PTX version >= 4.0 and sm_30", 1u);
    v13 = *(__m128i **)(a3 + 32);
    if ( *(_QWORD *)(a3 + 24) - (_QWORD)v13 > 0x14u )
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_44C2DC0);
      v13[1].m128i_i32[0] = 1684367201;
      v13[1].m128i_i8[4] = 41;
      *v13 = si128;
      v15 = (_QWORD *)(*(_QWORD *)(a3 + 32) + 21LL);
      v16 = *(_QWORD *)(a3 + 24);
      *(_QWORD *)(a3 + 32) = v15;
      if ( (unsigned __int64)(v16 - (_QWORD)v15) <= 7 )
        goto LABEL_8;
LABEL_23:
      v17 = a3;
      *v15 = 0x206E67696C612E20LL;
      *(_QWORD *)(a3 + 32) += 8LL;
      v18 = (*(_WORD *)(a2 + 34) >> 1) & 0x3F;
      if ( v18 )
        goto LABEL_9;
LABEL_24:
      v19 = sub_AE5260(v33, v11);
      goto LABEL_10;
    }
    sub_CB6200(a3, " .attribute(.managed)", 0x15u);
  }
  v15 = *(_QWORD **)(a3 + 32);
  if ( *(_QWORD *)(a3 + 24) - (_QWORD)v15 > 7u )
    goto LABEL_23;
LABEL_8:
  v17 = sub_CB6200(a3, (unsigned __int8 *)" .align ", 8u);
  v18 = (*(_WORD *)(a2 + 34) >> 1) & 0x3F;
  if ( !v18 )
    goto LABEL_24;
LABEL_9:
  v32 = v18 - 1;
  sub_AE5260(v33, v11);
  v19 = v32;
LABEL_10:
  sub_CB59D0(v17, 1LL << v19);
  if ( sub_BCAC40(v11, 128) || (v20 = *(unsigned __int8 *)(v11 + 8), (_BYTE)v20 == 5) )
  {
    sub_904010(a3, " .b8 ");
    v26 = sub_31DB510(a1, a2);
    sub_EA12C0(v26, a3, *(_BYTE **)(a1 + 208));
    v27 = "[16]";
LABEL_26:
    sub_904010(a3, v27);
    return;
  }
  if ( (unsigned __int8)v20 > 3u && (v20 & 0xF5) != 4 )
  {
    if ( (unsigned int)(v20 - 15) > 2 )
      BUG();
    v28 = sub_9208B0(v33, v11);
    v35 = v29;
    v34 = (unsigned __int8 *)((unsigned __int64)(v28 + 7) >> 3);
    v30 = sub_CA1930(&v34);
    sub_904010(a3, " .b8 ");
    v31 = sub_31DB510(a1, a2);
    sub_EA12C0(v31, a3, *(_BYTE **)(a1 + 208));
    sub_904010(a3, "[");
    if ( v30 )
      sub_CB59F0(a3, v30);
    v27 = "]";
    goto LABEL_26;
  }
  v21 = *(_WORD **)(a3 + 32);
  if ( *(_QWORD *)(a3 + 24) - (_QWORD)v21 <= 1u )
  {
    v22 = sub_CB6200(a3, (unsigned __int8 *)" .", 2u);
  }
  else
  {
    v22 = a3;
    *v21 = 11808;
    *(_QWORD *)(a3 + 32) += 2LL;
  }
  sub_30246F0((__int64)&v34, a1, v11, 1);
  v23 = sub_CB6200(v22, v34, v35);
  v24 = *(_BYTE **)(v23 + 32);
  if ( *(_BYTE **)(v23 + 24) == v24 )
  {
    sub_CB6200(v23, (unsigned __int8 *)" ", 1u);
  }
  else
  {
    *v24 = 32;
    ++*(_QWORD *)(v23 + 32);
  }
  if ( v34 != (unsigned __int8 *)&v36 )
    j_j___libc_free_0((unsigned __int64)v34);
  v25 = sub_31DB510(a1, a2);
  sub_EA12C0(v25, a3, *(_BYTE **)(a1 + 208));
}
