// Function: sub_D36E30
// Address: 0xd36e30
//
__int64 __fastcall sub_D36E30(_QWORD *a1, __int64 a2, unsigned int a3)
{
  __int64 v5; // rax
  _QWORD *v6; // rdx
  __int64 v7; // r13
  __int64 v8; // rdi
  _BYTE *v9; // rax
  __int64 v10; // r13
  __int64 v11; // rax
  void *v12; // rdx
  __int64 v13; // rax
  unsigned int *v14; // r13
  _BYTE *v15; // rax
  _BYTE *v16; // rax
  __int64 v17; // rax
  __m128i *v18; // rdx
  __int64 v19; // rdi
  __m128i v20; // xmm0
  _DWORD *v21; // rax
  __int64 v22; // rax
  __m128i *v23; // rdx
  __m128i v24; // xmm0
  __int64 v25; // rax
  _BYTE *v26; // rax
  __int64 v27; // rax
  __m128i *v28; // rdx
  __m128i v29; // xmm0
  __int64 v31; // rax
  __m128i *v32; // rdx
  __m128i v33; // xmm0
  __int64 v34; // rax
  __m128i *v35; // rdx
  __m128i si128; // xmm0
  __int64 v37; // r13
  __int64 v38; // rax
  __int64 v39; // rax
  __m128i *v40; // rdx
  __m128i v41; // xmm0
  __m128i *v42; // rdx
  __m128i v43; // xmm0
  __int64 v44; // rdi
  __int64 v45; // rax
  __int64 v46; // rdx
  __m128i v47; // xmm0
  unsigned int *v48; // [rsp+8h] [rbp-58h]
  unsigned __int8 *v49[2]; // [rsp+10h] [rbp-50h] BYREF
  __int64 v50; // [rsp+20h] [rbp-40h] BYREF

  if ( *((_BYTE *)a1 + 40) )
  {
    v34 = sub_CB69B0(a2, a3);
    v35 = *(__m128i **)(v34 + 32);
    if ( *(_QWORD *)(v34 + 24) - (_QWORD)v35 <= 0x1Au )
    {
      sub_CB6200(v34, "Memory dependences are safe", 0x1Bu);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F717F0);
      qmemcpy(&v35[1], "es are safe", 11);
      *v35 = si128;
      *(_QWORD *)(v34 + 32) += 27LL;
    }
    v37 = a1[2];
    if ( *(_QWORD *)(v37 + 216) != 0xFFFFFFFFLL )
    {
      v42 = *(__m128i **)(a2 + 32);
      if ( *(_QWORD *)(a2 + 24) - (_QWORD)v42 <= 0x24u )
      {
        v44 = sub_CB6200(a2, " with a maximum safe vector width of ", 0x25u);
      }
      else
      {
        v43 = _mm_load_si128((const __m128i *)&xmmword_3F71800);
        v42[2].m128i_i32[0] = 1718558824;
        v44 = a2;
        v42[2].m128i_i8[4] = 32;
        *v42 = v43;
        v42[1] = _mm_load_si128((const __m128i *)&xmmword_3F71810);
        *(_QWORD *)(a2 + 32) += 37LL;
      }
      v45 = sub_CB59D0(v44, *(_QWORD *)(v37 + 216));
      v46 = *(_QWORD *)(v45 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(v45 + 24) - v46) <= 4 )
      {
        sub_CB6200(v45, " bits", 5u);
      }
      else
      {
        *(_DWORD *)v46 = 1953063456;
        *(_BYTE *)(v46 + 4) = 115;
        *(_QWORD *)(v45 + 32) += 5LL;
      }
    }
    v38 = *(_QWORD *)(a2 + 32);
    if ( *(_BYTE *)a1[1] )
    {
      if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v38) <= 0x14 )
      {
        sub_CB6200(a2, " with run-time checks", 0x15u);
        v38 = *(_QWORD *)(a2 + 32);
      }
      else
      {
        v47 = _mm_load_si128((const __m128i *)&xmmword_3F71820);
        *(_DWORD *)(v38 + 16) = 1801676136;
        *(_BYTE *)(v38 + 20) = 115;
        *(__m128i *)v38 = v47;
        v38 = *(_QWORD *)(a2 + 32) + 21LL;
        *(_QWORD *)(a2 + 32) = v38;
      }
    }
    if ( *(_QWORD *)(a2 + 24) == v38 )
    {
      sub_CB6200(a2, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *(_BYTE *)v38 = 10;
      ++*(_QWORD *)(a2 + 32);
    }
  }
  if ( *((_BYTE *)a1 + 41) )
  {
    v31 = sub_CB69B0(a2, a3);
    v32 = *(__m128i **)(v31 + 32);
    if ( *(_QWORD *)(v31 + 24) - (_QWORD)v32 <= 0x20u )
    {
      sub_CB6200(v31, "Has convergent operation in loop\n", 0x21u);
    }
    else
    {
      v33 = _mm_load_si128((const __m128i *)&xmmword_3F71830);
      v32[2].m128i_i8[0] = 10;
      *v32 = v33;
      v32[1] = _mm_load_si128((const __m128i *)&xmmword_3F71840);
      *(_QWORD *)(v31 + 32) += 33LL;
    }
  }
  if ( a1[14] )
  {
    v5 = sub_CB69B0(a2, a3);
    v6 = *(_QWORD **)(v5 + 32);
    v7 = v5;
    if ( *(_QWORD *)(v5 + 24) - (_QWORD)v6 <= 7u )
    {
      v7 = sub_CB6200(v5, "Report: ", 8u);
    }
    else
    {
      *v6 = 0x203A74726F706552LL;
      *(_QWORD *)(v5 + 32) += 8LL;
    }
    sub_B17B60((__int64)v49, a1[14]);
    v8 = sub_CB6200(v7, v49[0], (size_t)v49[1]);
    v9 = *(_BYTE **)(v8 + 32);
    if ( *(_BYTE **)(v8 + 24) == v9 )
    {
      sub_CB6200(v8, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v9 = 10;
      ++*(_QWORD *)(v8 + 32);
    }
    if ( (__int64 *)v49[0] != &v50 )
      j_j___libc_free_0(v49[0], v50 + 1);
  }
  v10 = a1[2];
  if ( *(_BYTE *)(v10 + 232) )
  {
    v11 = sub_CB69B0(a2, a3);
    v12 = *(void **)(v11 + 32);
    if ( *(_QWORD *)(v11 + 24) - (_QWORD)v12 <= 0xCu )
    {
      sub_CB6200(v11, "Dependences:\n", 0xDu);
    }
    else
    {
      qmemcpy(v12, "Dependences:\n", 13);
      *(_QWORD *)(v11 + 32) += 13LL;
    }
    v13 = 3LL * *(unsigned int *)(v10 + 248);
    v14 = *(unsigned int **)(v10 + 240);
    v48 = &v14[v13];
    while ( v48 != v14 )
    {
      while ( 1 )
      {
        sub_D362E0(v14, a2, a3 + 2, (_QWORD *)(a1[2] + 56LL));
        v15 = *(_BYTE **)(a2 + 32);
        if ( *(_BYTE **)(a2 + 24) == v15 )
          break;
        *v15 = 10;
        v14 += 3;
        ++*(_QWORD *)(a2 + 32);
        if ( v48 == v14 )
          goto LABEL_19;
      }
      v14 += 3;
      sub_CB6200(a2, (unsigned __int8 *)"\n", 1u);
    }
  }
  else
  {
    v39 = sub_CB69B0(a2, a3);
    v40 = *(__m128i **)(v39 + 32);
    if ( *(_QWORD *)(v39 + 24) - (_QWORD)v40 <= 0x22u )
    {
      sub_CB6200(v39, "Too many dependences, not recorded\n", 0x23u);
    }
    else
    {
      v41 = _mm_load_si128((const __m128i *)&xmmword_3F71850);
      v40[2].m128i_i8[2] = 10;
      v40[2].m128i_i16[0] = 25701;
      *v40 = v41;
      v40[1] = _mm_load_si128((const __m128i *)&xmmword_3F71860);
      *(_QWORD *)(v39 + 32) += 35LL;
    }
  }
LABEL_19:
  sub_D34AE0(a1[1], a2, a3);
  v16 = *(_BYTE **)(a2 + 32);
  if ( *(_BYTE **)(a2 + 24) == v16 )
  {
    sub_CB6200(a2, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v16 = 10;
    ++*(_QWORD *)(a2 + 32);
  }
  v17 = sub_CB69B0(a2, a3);
  v18 = *(__m128i **)(v17 + 32);
  v19 = v17;
  if ( *(_QWORD *)(v17 + 24) - (_QWORD)v18 <= 0x31u )
  {
    v19 = sub_CB6200(v17, "Non vectorizable stores to invariant address were ", 0x32u);
    v21 = *(_DWORD **)(v19 + 32);
  }
  else
  {
    v20 = _mm_load_si128((const __m128i *)&xmmword_3F71870);
    v18[3].m128i_i16[0] = 8293;
    *v18 = v20;
    v18[1] = _mm_load_si128((const __m128i *)&xmmword_3F71880);
    v18[2] = _mm_load_si128((const __m128i *)&xmmword_3F71890);
    v21 = (_DWORD *)(*(_QWORD *)(v17 + 32) + 50LL);
    *(_QWORD *)(v19 + 32) = v21;
  }
  if ( !*((_BYTE *)a1 + 42) && !*((_BYTE *)a1 + 43) )
  {
    if ( *(_QWORD *)(v19 + 24) - (_QWORD)v21 > 3u )
    {
      *v21 = 544501614;
      v21 = (_DWORD *)(*(_QWORD *)(v19 + 32) + 4LL);
      *(_QWORD *)(v19 + 32) = v21;
    }
    else
    {
      v19 = sub_CB6200(v19, "not ", 4u);
      v21 = *(_DWORD **)(v19 + 32);
    }
  }
  if ( *(_QWORD *)(v19 + 24) - (_QWORD)v21 <= 0xEu )
  {
    sub_CB6200(v19, "found in loop.\n", 0xFu);
  }
  else
  {
    qmemcpy(v21, "found in loop.\n", 15);
    *(_QWORD *)(v19 + 32) += 15LL;
  }
  v22 = sub_CB69B0(a2, a3);
  v23 = *(__m128i **)(v22 + 32);
  if ( *(_QWORD *)(v22 + 24) - (_QWORD)v23 <= 0x11u )
  {
    sub_CB6200(v22, "SCEV assumptions:\n", 0x12u);
  }
  else
  {
    v24 = _mm_load_si128((const __m128i *)&xmmword_3F718A0);
    v23[1].m128i_i16[0] = 2618;
    *v23 = v24;
    *(_QWORD *)(v22 + 32) += 18LL;
  }
  v25 = sub_D9B120(*a1);
  (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v25 + 24LL))(v25, a2, a3);
  v26 = *(_BYTE **)(a2 + 32);
  if ( *(_BYTE **)(a2 + 24) == v26 )
  {
    sub_CB6200(a2, (unsigned __int8 *)"\n", 1u);
  }
  else
  {
    *v26 = 10;
    ++*(_QWORD *)(a2 + 32);
  }
  v27 = sub_CB69B0(a2, a3);
  v28 = *(__m128i **)(v27 + 32);
  if ( *(_QWORD *)(v27 + 24) - (_QWORD)v28 <= 0x17u )
  {
    sub_CB6200(v27, "Expressions re-written:\n", 0x18u);
  }
  else
  {
    v29 = _mm_load_si128((const __m128i *)&xmmword_3F718B0);
    v28[1].m128i_i64[0] = 0xA3A6E6574746972LL;
    *v28 = v29;
    *(_QWORD *)(v27 + 32) += 24LL;
  }
  return sub_DD87A0(*a1, a2, a3);
}
