// Function: sub_36D5820
// Address: 0x36d5820
//
__int64 __fastcall sub_36D5820(__int64 a1, __int64 a2, _QWORD *a3, _QWORD *a4, __int64 a5, char a6)
{
  unsigned __int64 v6; // r12
  __int64 v7; // rax
  unsigned __int64 v10; // rax
  unsigned int v11; // eax
  __int64 v12; // r9
  __int64 v13; // r8
  unsigned __int64 v14; // rdx
  __int64 v15; // rax
  unsigned __int64 v16; // rax
  char v17; // al
  _QWORD *v18; // rdx
  unsigned __int64 v19; // rsi
  __int64 v20; // rax
  unsigned __int64 v21; // rax
  char v22; // al
  int v23; // eax
  __int64 v24; // rax
  const __m128i *v25; // r13
  unsigned __int64 v26; // rdx
  __int64 v27; // rdx
  __m128i *v28; // rax
  int v30; // eax
  __int64 v31; // rax
  const __m128i *v32; // r12
  unsigned __int64 v33; // rdx
  __int64 v34; // rdx
  __m128i *v35; // rax
  __int64 v36; // rcx
  const void *v37; // rsi
  __int8 *v38; // r12
  __int64 v39; // rcx
  const void *v40; // rsi
  __int8 *v41; // r13
  _QWORD *v42; // [rsp+0h] [rbp-50h]
  unsigned __int8 v43; // [rsp+0h] [rbp-50h]
  unsigned __int8 v44; // [rsp+8h] [rbp-48h]
  _QWORD *v45; // [rsp+8h] [rbp-48h]

  if ( *(_QWORD *)(a2 + 56) == a2 + 48 )
    goto LABEL_34;
  v6 = *(_QWORD *)(a2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v6 )
    BUG();
  v7 = *(_QWORD *)v6;
  if ( (*(_QWORD *)v6 & 4) == 0 && (*(_BYTE *)(v6 + 44) & 4) != 0 )
  {
    while ( 1 )
    {
      v10 = v7 & 0xFFFFFFFFFFFFFFF8LL;
      v6 = v10;
      if ( (*(_BYTE *)(v10 + 44) & 4) == 0 )
        break;
      v7 = *(_QWORD *)v10;
    }
  }
  v11 = sub_2FDF4D0(a1, v6);
  v13 = v11;
  if ( !(_BYTE)v11 )
    goto LABEL_34;
  if ( *(_QWORD *)(a2 + 56) == v6 )
    goto LABEL_36;
  v14 = *(_QWORD *)v6 & 0xFFFFFFFFFFFFFFF8LL;
  if ( !v14 )
    BUG();
  v15 = *(_QWORD *)v14;
  if ( (*(_QWORD *)v14 & 4) == 0 && (*(_BYTE *)(v14 + 44) & 4) != 0 )
  {
    while ( 1 )
    {
      v16 = v15 & 0xFFFFFFFFFFFFFFF8LL;
      v14 = v16;
      if ( (*(_BYTE *)(v16 + 44) & 4) == 0 )
        break;
      v15 = *(_QWORD *)v16;
    }
  }
  v44 = v13;
  v42 = (_QWORD *)v14;
  v17 = sub_2FDF4D0(a1, v14);
  v13 = v44;
  if ( !v17 )
  {
LABEL_36:
    v30 = *(unsigned __int16 *)(v6 + 68);
    if ( v30 == 1513 )
    {
      LODWORD(v13) = 0;
      *a3 = *(_QWORD *)(*(_QWORD *)(v6 + 32) + 24LL);
    }
    else if ( v30 == 396 )
    {
      *a3 = *(_QWORD *)(*(_QWORD *)(v6 + 32) + 64LL);
      v31 = *(unsigned int *)(a5 + 8);
      v32 = *(const __m128i **)(v6 + 32);
      v33 = v31 + 1;
      if ( v31 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
      {
        v36 = *(_QWORD *)a5;
        v37 = (const void *)(a5 + 16);
        if ( *(_QWORD *)a5 > (unsigned __int64)v32 || (unsigned __int64)v32 >= v36 + 40 * v31 )
        {
          sub_C8D5F0(a5, v37, v33, 0x28u, v13, v12);
          v34 = *(_QWORD *)a5;
          v31 = *(unsigned int *)(a5 + 8);
        }
        else
        {
          v38 = &v32->m128i_i8[-v36];
          sub_C8D5F0(a5, v37, v33, 0x28u, v13, v12);
          v34 = *(_QWORD *)a5;
          v31 = *(unsigned int *)(a5 + 8);
          v32 = (const __m128i *)&v38[*(_QWORD *)a5];
        }
      }
      else
      {
        v34 = *(_QWORD *)a5;
      }
      LODWORD(v13) = 0;
      v35 = (__m128i *)(v34 + 40 * v31);
      *v35 = _mm_loadu_si128(v32);
      v35[1] = _mm_loadu_si128(v32 + 1);
      v35[2].m128i_i64[0] = v32[2].m128i_i64[0];
      ++*(_DWORD *)(a5 + 8);
    }
    return (unsigned int)v13;
  }
  v18 = v42;
  if ( *(_QWORD **)(a2 + 56) != v42 )
  {
    v19 = *v42 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v19 )
      BUG();
    v20 = *(_QWORD *)v19;
    if ( (*(_QWORD *)v19 & 4) == 0 && (*(_BYTE *)(v19 + 44) & 4) != 0 )
    {
      while ( 1 )
      {
        v21 = v20 & 0xFFFFFFFFFFFFFFF8LL;
        v19 = v21;
        if ( (*(_BYTE *)(v21 + 44) & 4) == 0 )
          break;
        v20 = *(_QWORD *)v21;
      }
    }
    v43 = v44;
    v45 = v18;
    v22 = sub_2FDF4D0(a1, v19);
    v18 = v45;
    v13 = v43;
    if ( v22 )
      return (unsigned int)v13;
  }
  v23 = *((unsigned __int16 *)v18 + 34);
  if ( v23 == 396 )
  {
    if ( *(_WORD *)(v6 + 68) != 1513 )
      return (unsigned int)v13;
    *a3 = *(_QWORD *)(v18[4] + 64LL);
    v24 = *(unsigned int *)(a5 + 8);
    v25 = (const __m128i *)v18[4];
    v26 = v24 + 1;
    if ( v24 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
    {
      v39 = *(_QWORD *)a5;
      v40 = (const void *)(a5 + 16);
      if ( *(_QWORD *)a5 <= (unsigned __int64)v25 && (unsigned __int64)v25 < v39 + 40 * v24 )
      {
        v41 = &v25->m128i_i8[-v39];
        sub_C8D5F0(a5, v40, v26, 0x28u, v13, v12);
        v27 = *(_QWORD *)a5;
        v24 = *(unsigned int *)(a5 + 8);
        v25 = (const __m128i *)&v41[*(_QWORD *)a5];
        goto LABEL_33;
      }
      sub_C8D5F0(a5, v40, v26, 0x28u, v13, v12);
      v24 = *(unsigned int *)(a5 + 8);
    }
    v27 = *(_QWORD *)a5;
LABEL_33:
    v28 = (__m128i *)(v27 + 40 * v24);
    *v28 = _mm_loadu_si128(v25);
    v28[1] = _mm_loadu_si128(v25 + 1);
    v28[2].m128i_i64[0] = v25[2].m128i_i64[0];
    ++*(_DWORD *)(a5 + 8);
    *a4 = *(_QWORD *)(*(_QWORD *)(v6 + 32) + 24LL);
    goto LABEL_34;
  }
  if ( v23 == 1513 && *(_WORD *)(v6 + 68) == 1513 )
  {
    *a3 = *(_QWORD *)(v18[4] + 24LL);
    if ( a6 )
    {
      sub_2E88E20(v6);
      LODWORD(v13) = 0;
      return (unsigned int)v13;
    }
LABEL_34:
    LODWORD(v13) = 0;
  }
  return (unsigned int)v13;
}
