// Function: sub_2162080
// Address: 0x2162080
//
__int64 __fastcall sub_2162080(__int64 a1, __int64 a2, _QWORD *a3, _QWORD *a4, __int64 a5, char a6)
{
  unsigned __int64 v7; // r12
  __int64 v8; // rax
  unsigned __int64 v10; // rax
  unsigned int v11; // r8d
  int v12; // r9d
  unsigned __int64 v13; // rdx
  __int64 v14; // rax
  unsigned __int64 v15; // rax
  char v16; // al
  _QWORD *v17; // rdx
  unsigned __int64 v18; // rsi
  __int64 v19; // rax
  unsigned __int64 v20; // rax
  char v21; // al
  int v22; // eax
  const __m128i *v23; // r13
  __int64 v24; // rax
  __m128i *v25; // rax
  int v27; // eax
  const __m128i *v28; // r12
  __int64 v29; // rax
  __m128i *v30; // rax
  _QWORD *v31; // [rsp+0h] [rbp-50h]
  unsigned __int8 v32; // [rsp+0h] [rbp-50h]
  unsigned __int8 v33; // [rsp+8h] [rbp-48h]
  _QWORD *v34; // [rsp+8h] [rbp-48h]

  if ( *(_QWORD *)(a2 + 32) == a2 + 24 )
    return 0;
  v7 = *(_QWORD *)(a2 + 24) & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v7 )
    BUG();
  v8 = *(_QWORD *)v7;
  if ( (*(_QWORD *)v7 & 4) == 0 && (*(_BYTE *)(v7 + 46) & 4) != 0 )
  {
    while ( 1 )
    {
      v10 = v8 & 0xFFFFFFFFFFFFFFF8LL;
      v7 = v10;
      if ( (*(_BYTE *)(v10 + 46) & 4) == 0 )
        break;
      v8 = *(_QWORD *)v10;
    }
  }
  v11 = (*(__int64 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)a1 + 664LL))(a1, v7);
  if ( !(_BYTE)v11 )
    return 0;
  if ( *(_QWORD *)(a2 + 32) == v7 )
    goto LABEL_36;
  v13 = *(_QWORD *)v7 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v13 )
    BUG();
  v14 = *(_QWORD *)v13;
  if ( (*(_QWORD *)v13 & 4) == 0 && (*(_BYTE *)(v13 + 46) & 4) != 0 )
  {
    while ( 1 )
    {
      v15 = v14 & 0xFFFFFFFFFFFFFFF8LL;
      v13 = v15;
      if ( (*(_BYTE *)(v15 + 46) & 4) == 0 )
        break;
      v14 = *(_QWORD *)v15;
    }
  }
  v33 = v11;
  v31 = (_QWORD *)v13;
  v16 = (*(__int64 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)a1 + 664LL))(a1, v13);
  v11 = v33;
  if ( !v16 )
  {
LABEL_36:
    v27 = **(unsigned __int16 **)(v7 + 16);
    if ( v27 == 533 )
    {
      v11 = 0;
      *a3 = *(_QWORD *)(*(_QWORD *)(v7 + 32) + 24LL);
    }
    else if ( v27 == 190 )
    {
      *a3 = *(_QWORD *)(*(_QWORD *)(v7 + 32) + 64LL);
      v28 = *(const __m128i **)(v7 + 32);
      v29 = *(unsigned int *)(a5 + 8);
      if ( (unsigned int)v29 >= *(_DWORD *)(a5 + 12) )
      {
        sub_16CD150(a5, (const void *)(a5 + 16), 0, 40, v11, v12);
        v29 = *(unsigned int *)(a5 + 8);
      }
      v11 = 0;
      v30 = (__m128i *)(*(_QWORD *)a5 + 40 * v29);
      *v30 = _mm_loadu_si128(v28);
      v30[1] = _mm_loadu_si128(v28 + 1);
      v30[2].m128i_i64[0] = v28[2].m128i_i64[0];
      ++*(_DWORD *)(a5 + 8);
    }
  }
  else
  {
    v17 = v31;
    if ( *(_QWORD **)(a2 + 32) == v31 )
      goto LABEL_25;
    v18 = *v31 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v18 )
      BUG();
    v19 = *(_QWORD *)v18;
    if ( (*(_QWORD *)v18 & 4) == 0 && (*(_BYTE *)(v18 + 46) & 4) != 0 )
    {
      while ( 1 )
      {
        v20 = v19 & 0xFFFFFFFFFFFFFFF8LL;
        v18 = v20;
        if ( (*(_BYTE *)(v20 + 46) & 4) == 0 )
          break;
        v19 = *(_QWORD *)v20;
      }
    }
    v32 = v33;
    v34 = v17;
    v21 = (*(__int64 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)a1 + 664LL))(a1, v18);
    v17 = v34;
    v11 = v32;
    if ( !v21 )
    {
LABEL_25:
      v22 = *(unsigned __int16 *)v17[2];
      if ( v22 == 190 )
      {
        if ( **(_WORD **)(v7 + 16) != 533 )
          return v11;
        *a3 = *(_QWORD *)(v17[4] + 64LL);
        v23 = (const __m128i *)v17[4];
        v24 = *(unsigned int *)(a5 + 8);
        if ( (unsigned int)v24 >= *(_DWORD *)(a5 + 12) )
        {
          sub_16CD150(a5, (const void *)(a5 + 16), 0, 40, v11, v12);
          v24 = *(unsigned int *)(a5 + 8);
        }
        v25 = (__m128i *)(*(_QWORD *)a5 + 40 * v24);
        *v25 = _mm_loadu_si128(v23);
        v25[1] = _mm_loadu_si128(v23 + 1);
        v25[2].m128i_i64[0] = v23[2].m128i_i64[0];
        ++*(_DWORD *)(a5 + 8);
        *a4 = *(_QWORD *)(*(_QWORD *)(v7 + 32) + 24LL);
        return 0;
      }
      if ( v22 == 533 && **(_WORD **)(v7 + 16) == 533 )
      {
        *a3 = *(_QWORD *)(v17[4] + 24LL);
        if ( a6 )
        {
          sub_1E16240(v7);
          return 0;
        }
        return 0;
      }
    }
  }
  return v11;
}
