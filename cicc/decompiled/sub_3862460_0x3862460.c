// Function: sub_3862460
// Address: 0x3862460
//
__int64 __fastcall sub_3862460(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  _QWORD *v9; // rdx
  __int64 v10; // r13
  __int64 v11; // rdi
  _BYTE *v12; // rax
  __int64 v13; // r13
  __int64 v14; // rax
  void *v15; // rdx
  __int64 v16; // rax
  unsigned int *v17; // r13
  _BYTE *v18; // rax
  _BYTE *v19; // rax
  __int64 v20; // rax
  __m128i *v21; // rdx
  __int64 v22; // rdi
  __m128i v23; // xmm0
  _DWORD *v24; // rax
  __int64 v25; // rax
  __m128i *v26; // rdx
  __m128i v27; // xmm0
  __int64 v28; // rax
  _BYTE *v29; // rax
  __int64 v30; // rax
  __m128i *v31; // rdx
  __m128i v32; // xmm0
  __int64 v34; // rax
  __m128i *v35; // rdx
  __m128i si128; // xmm0
  _BYTE *v37; // rax
  __int64 v38; // rax
  __m128i *v39; // rdx
  __m128i v40; // xmm0
  __int64 v41; // rax
  __int64 v42; // rax
  unsigned int *v43; // [rsp+8h] [rbp-58h]
  char *v44[2]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v45; // [rsp+20h] [rbp-40h] BYREF

  if ( !*(_BYTE *)(a1 + 48) )
    goto LABEL_2;
  v34 = sub_16E8750(a2, a3);
  v35 = *(__m128i **)(v34 + 24);
  if ( *(_QWORD *)(v34 + 16) - (_QWORD)v35 <= 0x1Au )
  {
    sub_16E7EE0(v34, "Memory dependences are safe", 0x1Bu);
    if ( *(_QWORD *)(a1 + 40) == -1 )
    {
LABEL_42:
      if ( !**(_BYTE **)(a1 + 8) )
        goto LABEL_43;
LABEL_50:
      sub_1263B40(a2, " with run-time checks");
      v37 = *(_BYTE **)(a2 + 24);
      if ( *(_BYTE **)(a2 + 16) != v37 )
        goto LABEL_44;
LABEL_51:
      sub_16E7EE0(a2, "\n", 1u);
      goto LABEL_2;
    }
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F717F0);
    qmemcpy(&v35[1], "es are safe", 11);
    *v35 = si128;
    *(_QWORD *)(v34 + 24) += 27LL;
    if ( *(_QWORD *)(a1 + 40) == -1 )
      goto LABEL_42;
  }
  v41 = sub_1263B40(a2, " with a maximum dependence distance of ");
  v42 = sub_16E7A90(v41, *(_QWORD *)(a1 + 40));
  sub_1263B40(v42, " bytes");
  if ( **(_BYTE **)(a1 + 8) )
    goto LABEL_50;
LABEL_43:
  v37 = *(_BYTE **)(a2 + 24);
  if ( *(_BYTE **)(a2 + 16) == v37 )
    goto LABEL_51;
LABEL_44:
  *v37 = 10;
  ++*(_QWORD *)(a2 + 24);
LABEL_2:
  if ( *(_QWORD *)(a1 + 56) )
  {
    v5 = sub_16E8750(a2, a3);
    v9 = *(_QWORD **)(v5 + 24);
    v10 = v5;
    if ( *(_QWORD *)(v5 + 16) - (_QWORD)v9 <= 7u )
    {
      v10 = sub_16E7EE0(v5, "Report: ", 8u);
    }
    else
    {
      *v9 = 0x203A74726F706552LL;
      *(_QWORD *)(v5 + 24) += 8LL;
    }
    sub_15CA8E0((__int64 *)v44, *(_QWORD *)(a1 + 56), (__int64)v9, v6, v7, v8);
    v11 = sub_16E7EE0(v10, v44[0], (size_t)v44[1]);
    v12 = *(_BYTE **)(v11 + 24);
    if ( *(_BYTE **)(v11 + 16) == v12 )
    {
      sub_16E7EE0(v11, "\n", 1u);
    }
    else
    {
      *v12 = 10;
      ++*(_QWORD *)(v11 + 24);
    }
    if ( (__int64 *)v44[0] != &v45 )
      j_j___libc_free_0((unsigned __int64)v44[0]);
  }
  v13 = *(_QWORD *)(a1 + 16);
  if ( *(_BYTE *)(v13 + 218) )
  {
    v14 = sub_16E8750(a2, a3);
    v15 = *(void **)(v14 + 24);
    if ( *(_QWORD *)(v14 + 16) - (_QWORD)v15 <= 0xCu )
    {
      sub_16E7EE0(v14, "Dependences:\n", 0xDu);
    }
    else
    {
      qmemcpy(v15, "Dependences:\n", 13);
      *(_QWORD *)(v14 + 24) += 13LL;
    }
    v16 = 3LL * *(unsigned int *)(v13 + 232);
    v17 = *(unsigned int **)(v13 + 224);
    v43 = &v17[v16];
    while ( v43 != v17 )
    {
      while ( 1 )
      {
        sub_3860390(v17, a2, a3 + 2, (_QWORD *)(*(_QWORD *)(a1 + 16) + 48LL));
        v18 = *(_BYTE **)(a2 + 24);
        if ( *(_BYTE **)(a2 + 16) == v18 )
          break;
        *v18 = 10;
        v17 += 3;
        ++*(_QWORD *)(a2 + 24);
        if ( v43 == v17 )
          goto LABEL_18;
      }
      v17 += 3;
      sub_16E7EE0(a2, "\n", 1u);
    }
  }
  else
  {
    v38 = sub_16E8750(a2, a3);
    v39 = *(__m128i **)(v38 + 24);
    if ( *(_QWORD *)(v38 + 16) - (_QWORD)v39 <= 0x22u )
    {
      sub_16E7EE0(v38, "Too many dependences, not recorded\n", 0x23u);
    }
    else
    {
      v40 = _mm_load_si128((const __m128i *)&xmmword_3F71850);
      v39[2].m128i_i8[2] = 10;
      v39[2].m128i_i16[0] = 25701;
      *v39 = v40;
      v39[1] = _mm_load_si128((const __m128i *)&xmmword_3F71860);
      *(_QWORD *)(v38 + 24) += 35LL;
    }
  }
LABEL_18:
  sub_385E1B0(*(_QWORD *)(a1 + 8), a2, a3);
  v19 = *(_BYTE **)(a2 + 24);
  if ( *(_BYTE **)(a2 + 16) == v19 )
  {
    sub_16E7EE0(a2, "\n", 1u);
  }
  else
  {
    *v19 = 10;
    ++*(_QWORD *)(a2 + 24);
  }
  v20 = sub_16E8750(a2, a3);
  v21 = *(__m128i **)(v20 + 24);
  v22 = v20;
  if ( *(_QWORD *)(v20 + 16) - (_QWORD)v21 <= 0x1Eu )
  {
    v22 = sub_16E7EE0(v20, "Store to invariant address was ", 0x1Fu);
    v24 = *(_DWORD **)(v22 + 24);
  }
  else
  {
    v23 = _mm_load_si128((const __m128i *)&xmmword_452B790);
    qmemcpy(&v21[1], "nt address was ", 15);
    *v21 = v23;
    v24 = (_DWORD *)(*(_QWORD *)(v20 + 24) + 31LL);
    *(_QWORD *)(v22 + 24) = v24;
  }
  if ( !*(_BYTE *)(a1 + 49) )
  {
    if ( *(_QWORD *)(v22 + 16) - (_QWORD)v24 > 3u )
    {
      *v24 = 544501614;
      v24 = (_DWORD *)(*(_QWORD *)(v22 + 24) + 4LL);
      *(_QWORD *)(v22 + 24) = v24;
    }
    else
    {
      v22 = sub_16E7EE0(v22, "not ", 4u);
      v24 = *(_DWORD **)(v22 + 24);
    }
  }
  if ( *(_QWORD *)(v22 + 16) - (_QWORD)v24 <= 0xEu )
  {
    sub_16E7EE0(v22, "found in loop.\n", 0xFu);
  }
  else
  {
    qmemcpy(v24, "found in loop.\n", 15);
    *(_QWORD *)(v22 + 24) += 15LL;
  }
  v25 = sub_16E8750(a2, a3);
  v26 = *(__m128i **)(v25 + 24);
  if ( *(_QWORD *)(v25 + 16) - (_QWORD)v26 <= 0x11u )
  {
    sub_16E7EE0(v25, "SCEV assumptions:\n", 0x12u);
  }
  else
  {
    v27 = _mm_load_si128((const __m128i *)&xmmword_3F718A0);
    v26[1].m128i_i16[0] = 2618;
    *v26 = v27;
    *(_QWORD *)(v25 + 24) += 18LL;
  }
  v28 = sub_1458800(*(_QWORD *)a1);
  sub_1452580(v28, a2, a3);
  v29 = *(_BYTE **)(a2 + 24);
  if ( *(_BYTE **)(a2 + 16) == v29 )
  {
    sub_16E7EE0(a2, "\n", 1u);
  }
  else
  {
    *v29 = 10;
    ++*(_QWORD *)(a2 + 24);
  }
  v30 = sub_16E8750(a2, a3);
  v31 = *(__m128i **)(v30 + 24);
  if ( *(_QWORD *)(v30 + 16) - (_QWORD)v31 <= 0x17u )
  {
    sub_16E7EE0(v30, "Expressions re-written:\n", 0x18u);
  }
  else
  {
    v32 = _mm_load_si128((const __m128i *)&xmmword_3F718B0);
    v31[1].m128i_i64[0] = 0xA3A6E6574746972LL;
    *v31 = v32;
    *(_QWORD *)(v30 + 24) += 24LL;
  }
  return sub_1471620(*(_QWORD *)a1, a2, a3);
}
