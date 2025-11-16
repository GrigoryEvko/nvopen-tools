// Function: sub_1A75460
// Address: 0x1a75460
//
__int64 __fastcall sub_1A75460(__int64 a1)
{
  __m128i *v2; // r15
  __int64 v3; // r12
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // r12
  int v6; // esi
  int v7; // eax
  __int64 result; // rax
  unsigned __int64 v9; // r13
  unsigned __int64 v10; // rsi
  __int64 v11; // r14
  __int64 *v12; // rax
  char v13; // dl
  unsigned __int64 v14; // r12
  unsigned __int64 v15; // rax
  __int64 v16; // r13
  unsigned __int64 v17; // rdi
  unsigned int i; // r15d
  int v19; // eax
  unsigned int v20; // r14d
  __int32 v21; // r12d
  __int64 v22; // r8
  unsigned int v23; // r12d
  unsigned __int64 v24; // rdi
  int v25; // eax
  __int64 *v26; // rsi
  unsigned int v27; // r8d
  __int64 *v28; // rdi
  __m128i *v29; // rsi
  unsigned __int64 v30; // [rsp+0h] [rbp-60h]
  __int64 v31; // [rsp+8h] [rbp-58h]
  __m128i v32; // [rsp+10h] [rbp-50h] BYREF
  __m128i v33; // [rsp+20h] [rbp-40h] BYREF

LABEL_1:
  v2 = *(__m128i **)(a1 + 112);
  while ( 1 )
  {
    v3 = *(_QWORD *)v2[-2].m128i_i64[0];
    v4 = sub_157EBA0(v3 & 0xFFFFFFFFFFFFFFF8LL);
    v5 = v3 & 4;
    v6 = 0;
    if ( v4 )
    {
      v7 = sub_15F4D60(v4);
      v2 = *(__m128i **)(a1 + 112);
      v6 = v7;
    }
    result = v2[-2].m128i_i64[1];
    if ( ((result >> 1) & 3) != 0 )
    {
      if ( ((result >> 1) & 3) == (unsigned int)(v5 >> 1) )
        return result;
      v9 = result & 0xFFFFFFFFFFFFFFF8LL;
      v2[-2].m128i_i64[1] = v2[-2].m128i_i64[1] & 0xFFFFFFFFFFFFFFF9LL | 4;
      v10 = *(_QWORD *)((result & 0xFFFFFFFFFFFFFFF8LL) + 32);
      goto LABEL_7;
    }
    v20 = v2[-1].m128i_u32[2];
    if ( v20 == v6 )
      return result;
    v21 = v2[-1].m128i_i32[2];
    v9 = result & 0xFFFFFFFFFFFFFFF8LL;
    v31 = v2[-1].m128i_i64[0];
    while ( 1 )
    {
      v23 = v21 + 1;
      v2[-1].m128i_i32[2] = v23;
      v24 = sub_157EBA0(*(_QWORD *)(result & 0xFFFFFFFFFFFFFFF8LL) & 0xFFFFFFFFFFFFFFF8LL);
      v25 = 0;
      if ( v24 )
      {
        v25 = sub_15F4D60(v24);
        v23 = v2[-1].m128i_u32[2];
      }
      if ( v25 == v23 )
        break;
      v22 = sub_15F4DF0(v2[-1].m128i_i64[0], v23);
      result = v2[-2].m128i_i64[1];
      if ( *(_QWORD *)(*(_QWORD *)((result & 0xFFFFFFFFFFFFFFF8LL) + 8) + 32LL) != v22 )
        break;
      v21 = v2[-1].m128i_i32[2];
    }
    v10 = sub_15F4DF0(v31, v20);
LABEL_7:
    v11 = (__int64)sub_1444E60(*(_QWORD **)(v9 + 8), v10);
    v12 = *(__int64 **)(a1 + 8);
    if ( *(__int64 **)(a1 + 16) != v12 )
      goto LABEL_8;
    v26 = &v12[*(unsigned int *)(a1 + 28)];
    v27 = *(_DWORD *)(a1 + 28);
    if ( v12 != v26 )
    {
      v28 = 0;
      while ( v11 != *v12 )
      {
        if ( *v12 == -2 )
        {
          v28 = v12;
          if ( v12 + 1 == v26 )
            goto LABEL_31;
          ++v12;
        }
        else if ( v26 == ++v12 )
        {
          if ( !v28 )
            goto LABEL_34;
LABEL_31:
          *v28 = v11;
          --*(_DWORD *)(a1 + 32);
          ++*(_QWORD *)a1;
          goto LABEL_9;
        }
      }
      goto LABEL_1;
    }
LABEL_34:
    if ( v27 < *(_DWORD *)(a1 + 24) )
    {
      *(_DWORD *)(a1 + 28) = v27 + 1;
      *v26 = v11;
      ++*(_QWORD *)a1;
    }
    else
    {
LABEL_8:
      sub_16CCBA0(a1, v11);
      if ( !v13 )
        goto LABEL_1;
    }
LABEL_9:
    v14 = v11 & 0xFFFFFFFFFFFFFFF9LL | (*(_QWORD *)v11 >> 1) & 2LL;
    v30 = v14;
    v15 = sub_157EBA0(*(_QWORD *)v11 & 0xFFFFFFFFFFFFFFF8LL);
    v16 = v15;
    if ( (v14 & 2) != 0 )
    {
      if ( *(_QWORD *)((v14 & 0xFFFFFFFFFFFFFFF8LL) + 32) == *(_QWORD *)(*(_QWORD *)((v14 & 0xFFFFFFFFFFFFFFF8LL) + 8)
                                                                       + 32LL) )
        v30 = v14 & 0xFFFFFFFFFFFFFFF9LL | 4;
      i = 0;
    }
    else
    {
      v17 = v15;
      for ( i = 0; ; ++i )
      {
        v19 = 0;
        if ( v17 )
          v19 = sub_15F4D60(v17);
        if ( v19 == i || *(_QWORD *)(*(_QWORD *)((v14 & 0xFFFFFFFFFFFFFFF8LL) + 8) + 32LL) != sub_15F4DF0(v16, i) )
          break;
        v17 = sub_157EBA0(*(_QWORD *)v11 & 0xFFFFFFFFFFFFFFF8LL);
      }
    }
    v29 = *(__m128i **)(a1 + 112);
    v32.m128i_i64[0] = v11;
    v33.m128i_i64[0] = v16;
    v32.m128i_i64[1] = v30;
    v33.m128i_i32[2] = i;
    if ( v29 == *(__m128i **)(a1 + 120) )
    {
      sub_1A752D0((const __m128i **)(a1 + 104), v29, &v32);
      v2 = *(__m128i **)(a1 + 112);
    }
    else
    {
      if ( v29 )
      {
        *v29 = _mm_loadu_si128(&v32);
        v29[1] = _mm_loadu_si128(&v33);
        v29 = *(__m128i **)(a1 + 112);
      }
      v2 = v29 + 2;
      *(_QWORD *)(a1 + 112) = v29 + 2;
    }
  }
}
