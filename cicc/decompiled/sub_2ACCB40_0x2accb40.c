// Function: sub_2ACCB40
// Address: 0x2accb40
//
__int64 __fastcall sub_2ACCB40(const __m128i *a1, __int64 a2)
{
  __m128i *v3; // rbx
  __int64 v4; // r8
  unsigned int v5; // edi
  __int64 v6; // rcx
  __int64 *v7; // r9
  int v8; // r13d
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r10
  unsigned int v12; // r13d
  int v13; // r15d
  __int64 *v14; // r9
  unsigned int v15; // r8d
  __int64 *v16; // rdx
  __int64 v17; // r11
  __int64 v18; // rdi
  int v19; // r13d
  __int64 *v20; // r9
  unsigned int v21; // edx
  __int64 *v22; // rax
  __int64 v23; // r10
  unsigned int v24; // r8d
  __int64 v25; // rax
  int v26; // r15d
  __int64 *v27; // r10
  unsigned int v28; // edi
  __int64 *v29; // rdx
  __int64 v30; // r13
  __int32 v31; // eax
  unsigned int v32; // esi
  int v33; // edx
  int v34; // eax
  int v35; // eax
  int v36; // eax
  __int64 v37; // rax
  int v38; // eax
  int v39; // eax
  __int64 v40; // rax
  __int64 result; // rax
  int v42; // eax
  int v43; // eax
  __int64 *v44; // [rsp+8h] [rbp-48h] BYREF
  __m128i v45; // [rsp+10h] [rbp-40h] BYREF

  v3 = (__m128i *)&a1[-1];
  v45 = _mm_loadu_si128(a1);
  while ( 1 )
  {
    v32 = *(_DWORD *)(a2 + 24);
    if ( v32 )
    {
      v4 = v45.m128i_i64[0];
      v5 = v32 - 1;
      v6 = *(_QWORD *)(a2 + 8);
      v7 = 0;
      v8 = 1;
      v9 = (v32 - 1) & (((unsigned __int32)v45.m128i_i32[0] >> 9) ^ ((unsigned __int32)v45.m128i_i32[0] >> 4));
      v10 = (__int64 *)(v6 + 16LL * v9);
      v11 = *v10;
      if ( v45.m128i_i64[0] == *v10 )
      {
LABEL_3:
        v12 = *((_DWORD *)v10 + 2);
        goto LABEL_4;
      }
      while ( v11 != -4096 )
      {
        if ( v11 == -8192 && !v7 )
          v7 = v10;
        v9 = v5 & (v8 + v9);
        v10 = (__int64 *)(v6 + 16LL * v9);
        v11 = *v10;
        if ( v45.m128i_i64[0] == *v10 )
          goto LABEL_3;
        ++v8;
      }
      if ( !v7 )
        v7 = v10;
      v34 = *(_DWORD *)(a2 + 16);
      ++*(_QWORD *)a2;
      v33 = v34 + 1;
      v44 = v7;
      if ( 4 * (v34 + 1) < 3 * v32 )
      {
        if ( v32 - *(_DWORD *)(a2 + 20) - v33 > v32 >> 3 )
          goto LABEL_25;
        goto LABEL_14;
      }
    }
    else
    {
      ++*(_QWORD *)a2;
      v44 = 0;
    }
    v32 *= 2;
LABEL_14:
    sub_2ACC850(a2, v32);
    sub_2AC1490(a2, v45.m128i_i64, &v44);
    v4 = v45.m128i_i64[0];
    v7 = v44;
    v33 = *(_DWORD *)(a2 + 16) + 1;
LABEL_25:
    *(_DWORD *)(a2 + 16) = v33;
    if ( *v7 != -4096 )
      --*(_DWORD *)(a2 + 20);
    *v7 = v4;
    *((_DWORD *)v7 + 2) = 0;
    v32 = *(_DWORD *)(a2 + 24);
    v12 = v32;
    if ( !v32 )
    {
      ++*(_QWORD *)a2;
      v32 = 0;
      v44 = 0;
      goto LABEL_29;
    }
    v6 = *(_QWORD *)(a2 + 8);
    v5 = v32 - 1;
    v12 = 0;
LABEL_4:
    v13 = 1;
    v14 = 0;
    v15 = v5 & (((unsigned int)v3->m128i_i64[0] >> 9) ^ ((unsigned int)v3->m128i_i64[0] >> 4));
    v16 = (__int64 *)(v6 + 16LL * v15);
    v17 = *v16;
    if ( v3->m128i_i64[0] != *v16 )
      break;
LABEL_5:
    if ( *((_DWORD *)v16 + 2) != v12 )
      goto LABEL_6;
LABEL_57:
    if ( v45.m128i_i8[12] && !v3->m128i_i8[12] || v45.m128i_i32[2] >= (unsigned __int32)v3->m128i_i32[2] )
      goto LABEL_59;
    v25 = v3->m128i_i64[0];
LABEL_10:
    v3[1].m128i_i64[0] = v25;
    v31 = v3->m128i_i32[2];
    --v3;
    v3[2].m128i_i32[2] = v31;
    v3[2].m128i_i8[12] = v3[1].m128i_i8[12];
  }
  while ( v17 != -4096 )
  {
    if ( !v14 && v17 == -8192 )
      v14 = v16;
    v15 = v5 & (v13 + v15);
    v16 = (__int64 *)(v6 + 16LL * v15);
    v17 = *v16;
    if ( v3->m128i_i64[0] == *v16 )
      goto LABEL_5;
    ++v13;
  }
  v36 = *(_DWORD *)(a2 + 16);
  if ( !v14 )
    v14 = v16;
  ++*(_QWORD *)a2;
  v35 = v36 + 1;
  v44 = v14;
  if ( 4 * v35 >= 3 * v32 )
  {
LABEL_29:
    v32 *= 2;
LABEL_30:
    sub_2ACC850(a2, v32);
    sub_2AC1490(a2, v3->m128i_i64, &v44);
    v14 = v44;
    v35 = *(_DWORD *)(a2 + 16) + 1;
    goto LABEL_41;
  }
  if ( v32 - (v35 + *(_DWORD *)(a2 + 20)) <= v32 >> 3 )
    goto LABEL_30;
LABEL_41:
  *(_DWORD *)(a2 + 16) = v35;
  if ( *v14 != -4096 )
    --*(_DWORD *)(a2 + 20);
  v37 = v3->m128i_i64[0];
  *((_DWORD *)v14 + 2) = 0;
  *v14 = v37;
  if ( !v12 )
    goto LABEL_57;
  v32 = *(_DWORD *)(a2 + 24);
  if ( !v32 )
  {
    ++*(_QWORD *)a2;
    v44 = 0;
    goto LABEL_46;
  }
  v6 = *(_QWORD *)(a2 + 8);
LABEL_6:
  v18 = v45.m128i_i64[0];
  v19 = 1;
  v20 = 0;
  v21 = (v32 - 1) & (((unsigned __int32)v45.m128i_i32[0] >> 9) ^ ((unsigned __int32)v45.m128i_i32[0] >> 4));
  v22 = (__int64 *)(v6 + 16LL * v21);
  v23 = *v22;
  if ( v45.m128i_i64[0] == *v22 )
  {
LABEL_7:
    v24 = *((_DWORD *)v22 + 2);
    goto LABEL_8;
  }
  while ( v23 != -4096 )
  {
    if ( !v20 && v23 == -8192 )
      v20 = v22;
    v21 = (v32 - 1) & (v19 + v21);
    v22 = (__int64 *)(v6 + 16LL * v21);
    v23 = *v22;
    if ( v45.m128i_i64[0] == *v22 )
      goto LABEL_7;
    ++v19;
  }
  if ( !v20 )
    v20 = v22;
  v43 = *(_DWORD *)(a2 + 16);
  ++*(_QWORD *)a2;
  v38 = v43 + 1;
  v44 = v20;
  if ( 4 * v38 >= 3 * v32 )
  {
LABEL_46:
    v32 *= 2;
LABEL_47:
    sub_2ACC850(a2, v32);
    sub_2AC1490(a2, v45.m128i_i64, &v44);
    v18 = v45.m128i_i64[0];
    v20 = v44;
    v38 = *(_DWORD *)(a2 + 16) + 1;
    goto LABEL_48;
  }
  if ( v32 - (v38 + *(_DWORD *)(a2 + 20)) <= v32 >> 3 )
    goto LABEL_47;
LABEL_48:
  *(_DWORD *)(a2 + 16) = v38;
  if ( *v20 != -4096 )
    --*(_DWORD *)(a2 + 20);
  *v20 = v18;
  *((_DWORD *)v20 + 2) = 0;
  v32 = *(_DWORD *)(a2 + 24);
  if ( !v32 )
  {
    ++*(_QWORD *)a2;
    v44 = 0;
    goto LABEL_52;
  }
  v6 = *(_QWORD *)(a2 + 8);
  v24 = 0;
LABEL_8:
  v25 = v3->m128i_i64[0];
  v26 = 1;
  v27 = 0;
  v28 = (v32 - 1) & (((unsigned int)v3->m128i_i64[0] >> 9) ^ ((unsigned int)v3->m128i_i64[0] >> 4));
  v29 = (__int64 *)(v6 + 16LL * v28);
  v30 = *v29;
  if ( *v29 == v3->m128i_i64[0] )
  {
LABEL_9:
    if ( v24 >= *((_DWORD *)v29 + 2) )
      goto LABEL_59;
    goto LABEL_10;
  }
  while ( v30 != -4096 )
  {
    if ( v30 == -8192 && !v27 )
      v27 = v29;
    v28 = (v32 - 1) & (v26 + v28);
    v29 = (__int64 *)(v6 + 16LL * v28);
    v30 = *v29;
    if ( v25 == *v29 )
      goto LABEL_9;
    ++v26;
  }
  v42 = *(_DWORD *)(a2 + 16);
  if ( !v27 )
    v27 = v29;
  ++*(_QWORD *)a2;
  v39 = v42 + 1;
  v44 = v27;
  if ( 4 * v39 >= 3 * v32 )
  {
LABEL_52:
    v32 *= 2;
LABEL_53:
    sub_2ACC850(a2, v32);
    sub_2AC1490(a2, v3->m128i_i64, &v44);
    v27 = v44;
    v39 = *(_DWORD *)(a2 + 16) + 1;
    goto LABEL_54;
  }
  if ( v32 - (v39 + *(_DWORD *)(a2 + 20)) <= v32 >> 3 )
    goto LABEL_53;
LABEL_54:
  *(_DWORD *)(a2 + 16) = v39;
  if ( *v27 != -4096 )
    --*(_DWORD *)(a2 + 20);
  v40 = v3->m128i_i64[0];
  *((_DWORD *)v27 + 2) = 0;
  *v27 = v40;
LABEL_59:
  v3[1].m128i_i64[0] = v45.m128i_i64[0];
  v3[1].m128i_i32[2] = v45.m128i_i32[2];
  result = v45.m128i_u8[12];
  v3[1].m128i_i8[12] = v45.m128i_i8[12];
  return result;
}
