// Function: sub_3939080
// Address: 0x3939080
//
__int64 __fastcall sub_3939080(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int *a5)
{
  __int64 v8; // rdi
  int *v9; // rsi
  unsigned __int64 v10; // rax
  __int64 v12; // rdx
  int **v13; // rbx
  unsigned __int64 v14; // r13
  char *v15; // rcx
  signed __int64 v16; // r13
  int *v17; // rbx
  unsigned __int64 v18; // rbx
  int *v19; // r15
  const __m128i **v20; // rbx
  const __m128i *i; // r13
  unsigned __int64 v22; // rax
  int *v23; // rbx
  const __m128i **v24; // r15
  const __m128i *k; // r13
  __m128i *v26; // rax
  __int64 v27; // rax
  const __m128i ***v28; // [rsp+0h] [rbp-80h]
  _QWORD *v29; // [rsp+8h] [rbp-78h]
  unsigned __int64 v30; // [rsp+10h] [rbp-70h]
  const __m128i **v31; // [rsp+18h] [rbp-68h]
  const __m128i **j; // [rsp+18h] [rbp-68h]
  int v33; // [rsp+2Ch] [rbp-54h] BYREF
  __int64 v34; // [rsp+30h] [rbp-50h] BYREF
  __int64 v35; // [rsp+38h] [rbp-48h] BYREF
  int **v36; // [rsp+40h] [rbp-40h] BYREF
  __int64 v37; // [rsp+48h] [rbp-38h]

  v8 = (__int64)&v34;
  v9 = *(int **)(a2 + 32);
  v36 = 0;
  v37 = 0;
  (*(void (__fastcall **)(__int64 *, int *, __int64, __int64, int ***))(*(_QWORD *)v9 + 24LL))(&v34, v9, a3, a4, &v36);
  v10 = v34 & 0xFFFFFFFFFFFFFFFELL;
  v30 = v34 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v34 & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *(_BYTE *)(a1 + 32) |= 3u;
    *(_QWORD *)a1 = v10;
    return a1;
  }
  v34 = 0;
  if ( !(_DWORD)v37 )
  {
LABEL_33:
    *(_DWORD *)(a2 + 8) = 11;
    v9 = &v33;
    v33 = 11;
    sub_3939040(&v35, &v33);
    v27 = v35;
    *(_BYTE *)(a1 + 32) |= 3u;
    *(_QWORD *)a1 = v27 & 0xFFFFFFFFFFFFFFFELL;
    goto LABEL_34;
  }
  v12 = (unsigned int)(v37 - 1);
  v13 = v36;
  while ( v13[6] != a5 )
  {
    v13 += 7;
    if ( &v36[7 * v12 + 7] == v13 )
      goto LABEL_33;
  }
  *(_BYTE *)(a1 + 32) = *(_BYTE *)(a1 + 32) & 0xFC | 2;
  v14 = (char *)v13[1] - (char *)*v13;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  if ( v14 )
  {
    if ( v14 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_41;
    v15 = (char *)sub_22077B0(v14);
  }
  else
  {
    v14 = 0;
    v15 = 0;
  }
  *(_QWORD *)a1 = v15;
  *(_QWORD *)(a1 + 16) = &v15[v14];
  *(_QWORD *)(a1 + 8) = v15;
  v9 = *v13;
  v16 = (char *)v13[1] - (char *)*v13;
  if ( v13[1] != *v13 )
    v15 = (char *)memmove(v15, v9, (char *)v13[1] - (char *)*v13);
  *(_QWORD *)(a1 + 8) = &v15[v16];
  v17 = v13[3];
  v28 = (const __m128i ***)v17;
  if ( !v17 )
  {
    *(_QWORD *)(a1 + 24) = 0;
    goto LABEL_34;
  }
  v8 = 48;
  v29 = (_QWORD *)sub_22077B0(0x30u);
  if ( !v29 )
    goto LABEL_32;
  v18 = *((_QWORD *)v17 + 1) - *(_QWORD *)v17;
  *v29 = 0;
  v29[1] = 0;
  v29[2] = 0;
  if ( v18 )
  {
    if ( v18 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_41;
    v8 = v18;
    v19 = (int *)sub_22077B0(v18);
  }
  else
  {
    v19 = 0;
  }
  *v29 = v19;
  v29[1] = v19;
  v29[2] = (char *)v19 + v18;
  v20 = *v28;
  v31 = v28[1];
  if ( v31 != *v28 )
  {
    do
    {
      if ( v19 )
      {
        *((_QWORD *)v19 + 1) = v19;
        *(_QWORD *)v19 = v19;
        *((_QWORD *)v19 + 2) = 0;
        for ( i = *v20; i != (const __m128i *)v20; i = (const __m128i *)i->m128i_i64[0] )
        {
          v9 = v19;
          v8 = sub_22077B0(0x20u);
          *(__m128i *)(v8 + 16) = _mm_loadu_si128(i + 1);
          sub_2208C80((_QWORD *)v8, (__int64)v19);
          ++*((_QWORD *)v19 + 2);
        }
      }
      v19 += 6;
      v20 += 3;
    }
    while ( v31 != v20 );
  }
  v12 = (__int64)v28;
  v29[1] = v19;
  v22 = (char *)v28[4] - (char *)v28[3];
  v29[3] = 0;
  v29[4] = 0;
  v29[5] = 0;
  if ( !v22 )
  {
    v23 = 0;
    goto LABEL_26;
  }
  v30 = v22;
  if ( v22 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_41:
    sub_4261EA(v8, v9, v12);
  v23 = (int *)sub_22077B0(v22);
LABEL_26:
  v12 = (__int64)v29;
  v29[3] = v23;
  v29[5] = (char *)v23 + v30;
  v29[4] = v23;
  v24 = v28[3];
  for ( j = v28[4]; j != v24; v24 += 3 )
  {
    if ( v23 )
    {
      *((_QWORD *)v23 + 1) = v23;
      *(_QWORD *)v23 = v23;
      *((_QWORD *)v23 + 2) = 0;
      for ( k = *v24; k != (const __m128i *)v24; k = (const __m128i *)k->m128i_i64[0] )
      {
        v26 = (__m128i *)sub_22077B0(0x20u);
        v9 = v23;
        v26[1] = _mm_loadu_si128(k + 1);
        sub_2208C80(v26, (__int64)v23);
        ++*((_QWORD *)v23 + 2);
      }
    }
    v23 += 6;
  }
  v29[4] = v23;
LABEL_32:
  *(_QWORD *)(a1 + 24) = v29;
LABEL_34:
  if ( (v34 & 1) != 0 || (v34 & 0xFFFFFFFFFFFFFFFELL) != 0 )
    sub_16BCAE0(&v34, (__int64)v9, v12);
  return a1;
}
