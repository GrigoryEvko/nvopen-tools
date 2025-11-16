// Function: sub_122F1C0
// Address: 0x122f1c0
//
__int64 __fastcall sub_122F1C0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 result; // rax
  int v6; // eax
  __int64 v7; // r9
  bool v8; // zf
  _BYTE *v9; // rsi
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdi
  __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // rcx
  const __m128i *v16; // rax
  __m128i *v17; // rax
  __int64 v18; // rdx
  int v19; // eax
  __int64 v20; // rdi
  __int64 v21; // rsi
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rsi
  __int64 v26; // rdx
  __m128i *v27; // rax
  __int64 v28; // rcx
  const __m128i *v29; // r14
  __int64 v30; // rax
  __int64 v31; // rdx
  const __m128i *v32; // rax
  unsigned __int64 v33; // rcx
  __m128i *v34; // rdx
  __int64 v35; // rcx
  const __m128i *v36; // rcx
  const __m128i *v37; // r15
  __int64 v38; // rdi
  int v39; // eax
  int v40; // [rsp+0h] [rbp-E0h]
  __m128i *v41; // [rsp+8h] [rbp-D8h]
  unsigned __int64 v42; // [rsp+10h] [rbp-D0h]
  __int64 v43; // [rsp+18h] [rbp-C8h]
  __int64 *v45; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v46; // [rsp+38h] [rbp-A8h] BYREF
  _BYTE *v47; // [rsp+40h] [rbp-A0h] BYREF
  _BYTE *v48; // [rsp+48h] [rbp-98h]
  _BYTE *v49; // [rsp+50h] [rbp-90h]
  __m128i *v50; // [rsp+60h] [rbp-80h] BYREF
  __int64 v51; // [rsp+68h] [rbp-78h]
  __m128i v52; // [rsp+70h] [rbp-70h] BYREF
  __m128i *v53; // [rsp+80h] [rbp-60h] BYREF
  __int64 v54; // [rsp+88h] [rbp-58h]
  __m128i v55; // [rsp+90h] [rbp-50h] BYREF
  char v56; // [rsp+A0h] [rbp-40h]
  char v57; // [rsp+A1h] [rbp-3Fh]

  result = 0;
  if ( *(_DWORD *)(a1 + 240) != 6 )
    return result;
  v42 = *(_QWORD *)(a1 + 232);
  v43 = a1 + 176;
  v6 = sub_1205200(a1 + 176);
  *(_DWORD *)(a1 + 240) = v6;
  if ( v6 == 7 )
  {
LABEL_40:
    if ( !*(_DWORD *)(a2 + 8) )
    {
      v57 = 1;
      v53 = (__m128i *)"operand bundle set must not be empty";
      v56 = 3;
      sub_11FD800(v43, v42, (__int64)&v53, 1);
      return 1;
    }
    *(_DWORD *)(a1 + 240) = sub_1205200(v43);
    return 0;
  }
  while ( 1 )
  {
    if ( *(_DWORD *)(a2 + 8) && (unsigned __int8)sub_120AFE0(a1, 4, "expected ',' in input list") )
      return 1;
    v52.m128i_i8[0] = 0;
    v50 = &v52;
    v51 = 0;
    if ( (unsigned __int8)sub_120B3D0(a1, (__int64)&v50)
      || (unsigned __int8)sub_120AFE0(a1, 12, "expected '(' in operand bundle") )
    {
      goto LABEL_19;
    }
    v8 = *(_DWORD *)(a1 + 240) == 13;
    v47 = 0;
    v48 = 0;
    v49 = 0;
    if ( !v8 )
      break;
LABEL_27:
    v10 = *(unsigned int *)(a2 + 8);
    if ( (unsigned int)v10 >= *(_DWORD *)(a2 + 12) )
    {
      v22 = sub_C8D7D0(a2, a2 + 16, 0, 0x38u, (unsigned __int64 *)&v46, v7);
      v53 = &v55;
      v41 = (__m128i *)v22;
      if ( v50 == &v52 )
      {
        v55 = _mm_load_si128(&v52);
      }
      else
      {
        v53 = v50;
        v55.m128i_i64[0] = v52.m128i_i64[0];
      }
      v23 = v51;
      v24 = (__int64)v47;
      v52.m128i_i8[0] = 0;
      v25 = (__int64)v48;
      v26 = (__int64)v49;
      v51 = 0;
      v54 = v23;
      v49 = 0;
      v50 = &v52;
      v48 = 0;
      v47 = 0;
      v27 = (__m128i *)((char *)v41 + 56 * *(unsigned int *)(a2 + 8));
      if ( v27 )
      {
        v27->m128i_i64[0] = (__int64)v27[1].m128i_i64;
        if ( v53 == &v55 )
        {
          v27[1] = _mm_load_si128(&v55);
        }
        else
        {
          v27->m128i_i64[0] = (__int64)v53;
          v27[1].m128i_i64[0] = v55.m128i_i64[0];
        }
        v28 = v54;
        v27[2].m128i_i64[0] = v24;
        v27[2].m128i_i64[1] = v25;
        v27->m128i_i64[1] = v28;
        v27[3].m128i_i64[0] = v26;
      }
      else
      {
        v25 = v26 - v24;
        if ( v24 )
          j_j___libc_free_0(v24, v25);
        if ( v53 != &v55 )
        {
          v25 = v55.m128i_i64[0] + 1;
          j_j___libc_free_0(v53, v55.m128i_i64[0] + 1);
        }
      }
      v29 = *(const __m128i **)a2;
      v30 = *(unsigned int *)(a2 + 8);
      v31 = *(_QWORD *)a2 + 56 * v30;
      if ( *(_QWORD *)a2 != v31 )
      {
        v32 = v29 + 1;
        v33 = 7
            * (((0xDB6DB6DB6DB6DB7LL * ((unsigned __int64)(v31 - (_QWORD)v29 - 56) >> 3)) & 0x1FFFFFFFFFFFFFFFLL) + 1);
        v34 = v41;
        v25 = (__int64)&v41->m128i_i64[v33];
        do
        {
          if ( v34 )
          {
            v34->m128i_i64[0] = (__int64)v34[1].m128i_i64;
            v36 = (const __m128i *)v32[-1].m128i_i64[0];
            if ( v36 == v32 )
            {
              v34[1] = _mm_loadu_si128(v32);
            }
            else
            {
              v34->m128i_i64[0] = (__int64)v36;
              v34[1].m128i_i64[0] = v32->m128i_i64[0];
            }
            v34->m128i_i64[1] = v32[-1].m128i_i64[1];
            v35 = v32[1].m128i_i64[0];
            v32[-1].m128i_i64[0] = (__int64)v32;
            v32[-1].m128i_i64[1] = 0;
            v32->m128i_i8[0] = 0;
            v34[2].m128i_i64[0] = v35;
            v34[2].m128i_i64[1] = v32[1].m128i_i64[1];
            v34[3].m128i_i64[0] = v32[2].m128i_i64[0];
            v32[2].m128i_i64[0] = 0;
            v32[1].m128i_i64[1] = 0;
            v32[1].m128i_i64[0] = 0;
          }
          v34 = (__m128i *)((char *)v34 + 56);
          v32 = (const __m128i *)((char *)v32 + 56);
        }
        while ( v34 != (__m128i *)v25 );
        v30 = *(unsigned int *)(a2 + 8);
        v29 = *(const __m128i **)a2;
      }
      v37 = (const __m128i *)((char *)v29 + 56 * v30);
      if ( v29 != v37 )
      {
        do
        {
          v38 = v37[-2].m128i_i64[1];
          v37 = (const __m128i *)((char *)v37 - 56);
          if ( v38 )
          {
            v25 = v37[3].m128i_i64[0] - v38;
            j_j___libc_free_0(v38, v25);
          }
          if ( (const __m128i *)v37->m128i_i64[0] != &v37[1] )
          {
            v25 = v37[1].m128i_i64[0] + 1;
            j_j___libc_free_0(v37->m128i_i64[0], v25);
          }
        }
        while ( v29 != v37 );
        v29 = *(const __m128i **)a2;
      }
      v39 = v46;
      if ( (const __m128i *)(a2 + 16) != v29 )
      {
        v40 = v46;
        _libc_free(v29, v25);
        v39 = v40;
      }
      ++*(_DWORD *)(a2 + 8);
      *(_QWORD *)a2 = v41;
      *(_DWORD *)(a2 + 12) = v39;
    }
    else
    {
      v53 = &v55;
      if ( v50 == &v52 )
      {
        v55 = _mm_load_si128(&v52);
      }
      else
      {
        v53 = v50;
        v55.m128i_i64[0] = v52.m128i_i64[0];
      }
      v11 = v51;
      v52.m128i_i8[0] = 0;
      v12 = (__int64)v47;
      v13 = (__int64)v49;
      v51 = 0;
      v54 = v11;
      v14 = 7 * v10;
      v50 = &v52;
      v15 = (__int64)v48;
      v49 = 0;
      v48 = 0;
      v16 = *(const __m128i **)a2;
      v47 = 0;
      v17 = (__m128i *)((char *)v16 + 8 * v14);
      if ( v17 )
      {
        v17->m128i_i64[0] = (__int64)v17[1].m128i_i64;
        if ( v53 == &v55 )
        {
          v17[1] = _mm_load_si128(&v55);
        }
        else
        {
          v17->m128i_i64[0] = (__int64)v53;
          v17[1].m128i_i64[0] = v55.m128i_i64[0];
        }
        v18 = v54;
        v17[2].m128i_i64[0] = v12;
        v17[2].m128i_i64[1] = v15;
        v17->m128i_i64[1] = v18;
        v17[3].m128i_i64[0] = v13;
      }
      else
      {
        v21 = v13 - v12;
        if ( v12 )
          j_j___libc_free_0(v12, v21);
        if ( v53 != &v55 )
          j_j___libc_free_0(v53, v55.m128i_i64[0] + 1);
      }
      ++*(_DWORD *)(a2 + 8);
    }
    v19 = sub_1205200(v43);
    v20 = (__int64)v47;
    *(_DWORD *)(a1 + 240) = v19;
    if ( v20 )
      j_j___libc_free_0(v20, &v49[-v20]);
    if ( v50 != &v52 )
      j_j___libc_free_0(v50, v52.m128i_i64[0] + 1);
    if ( *(_DWORD *)(a1 + 240) == 7 )
      goto LABEL_40;
  }
  do
  {
    v57 = 1;
    v45 = 0;
    v46 = 0;
    v53 = (__m128i *)"expected type";
    v56 = 3;
    if ( (unsigned __int8)sub_12190A0(a1, &v45, (int *)&v53, 0) )
      break;
    if ( *((_BYTE *)v45 + 8) == 9 )
    {
      if ( (unsigned __int8)sub_12255B0((__int64 **)a1, &v46, a3) )
        break;
      v9 = v48;
      if ( v48 != v49 )
      {
LABEL_11:
        if ( v9 )
        {
          *(_QWORD *)v9 = v46;
          v9 = v48;
        }
        v48 = v9 + 8;
        goto LABEL_14;
      }
    }
    else
    {
      if ( (unsigned __int8)sub_1224B80((__int64 **)a1, (__int64)v45, &v46, a3) )
        break;
      v9 = v48;
      if ( v48 != v49 )
        goto LABEL_11;
    }
    sub_9281F0((__int64)&v47, v9, &v46);
LABEL_14:
    if ( *(_DWORD *)(a1 + 240) == 13 )
      goto LABEL_27;
  }
  while ( v48 == v47 || !(unsigned __int8)sub_120AFE0(a1, 4, "expected ',' in input list") );
  if ( v47 )
    j_j___libc_free_0(v47, v49 - v47);
LABEL_19:
  if ( v50 != &v52 )
    j_j___libc_free_0(v50, v52.m128i_i64[0] + 1);
  return 1;
}
