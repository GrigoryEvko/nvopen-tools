// Function: sub_C0B8E0
// Address: 0xc0b8e0
//
__int64 __fastcall sub_C0B8E0(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 result; // rax
  __int64 v6; // rdx
  const __m128i *v7; // rsi
  const __m128i **v8; // r12
  char **v9; // r15
  __int64 v10; // rax
  char *v11; // r10
  size_t v12; // r11
  _QWORD *v13; // rax
  const __m128i *v14; // rax
  __int64 v15; // r11
  __int64 v16; // rdi
  __m128i *v17; // rdx
  __m128i *v18; // rax
  __int64 v19; // rdi
  _QWORD *v20; // rdi
  __int64 v21; // rsi
  __int64 v22; // rax
  _QWORD *v23; // rdi
  char *src; // [rsp+10h] [rbp-250h]
  size_t n; // [rsp+18h] [rbp-248h]
  __int64 v26; // [rsp+20h] [rbp-240h]
  size_t v27; // [rsp+48h] [rbp-218h] BYREF
  _QWORD v28[2]; // [rsp+50h] [rbp-210h] BYREF
  _QWORD v29[2]; // [rsp+60h] [rbp-200h] BYREF
  _QWORD v30[2]; // [rsp+70h] [rbp-1F0h] BYREF
  _BYTE v31[8]; // [rsp+80h] [rbp-1E0h] BYREF
  __int64 v32; // [rsp+88h] [rbp-1D8h]
  unsigned int v33; // [rsp+98h] [rbp-1C8h]
  const __m128i **v34; // [rsp+A0h] [rbp-1C0h]
  unsigned int v35; // [rsp+A8h] [rbp-1B8h]
  const __m128i *v36; // [rsp+B0h] [rbp-1B0h] BYREF
  __int64 v37; // [rsp+B8h] [rbp-1A8h]
  _BYTE v38[128]; // [rsp+C0h] [rbp-1A0h] BYREF
  __int64 v39; // [rsp+140h] [rbp-120h] BYREF
  char *v40; // [rsp+148h] [rbp-118h]
  char v41; // [rsp+158h] [rbp-108h] BYREF
  __int64 *v42; // [rsp+1D8h] [rbp-88h]
  __int64 v43; // [rsp+1E8h] [rbp-78h] BYREF
  const __m128i *v44; // [rsp+1F8h] [rbp-68h]
  unsigned __int64 v45; // [rsp+200h] [rbp-60h]
  __int64 v46; // [rsp+208h] [rbp-58h] BYREF
  char v47; // [rsp+220h] [rbp-40h]

  v39 = *(_QWORD *)(a1 + 72);
  v4 = sub_A747B0(&v39, -1, "vector-function-abi-variant", 0x1Bu);
  if ( !v4 )
    v4 = sub_B49600(a1, "vector-function-abi-variant", 0x1Bu);
  v39 = v4;
  result = sub_A72240(&v39);
  v28[1] = v6;
  v28[0] = result;
  if ( !v6 )
    return result;
  v36 = (const __m128i *)v38;
  v37 = 0x800000000LL;
  sub_C937F0(v28, &v36, ",", 1, 0xFFFFFFFFLL, 1);
  v7 = v36;
  sub_C0B4B0((__int64)v31, v36, &v36[(unsigned int)v37]);
  v8 = &v34[2 * v35];
  if ( v34 == v8 )
    goto LABEL_31;
  v9 = (char **)v34;
  do
  {
    while ( 1 )
    {
      v7 = (const __m128i *)*v9;
      sub_C0A940((__int64)&v39, *v9, v9[1], *(_QWORD *)(a1 + 80));
      if ( v47 )
        break;
LABEL_7:
      v9 += 2;
      if ( v8 == (const __m128i **)v9 )
        goto LABEL_30;
    }
    v10 = sub_B43CA0(a1);
    v7 = v44;
    if ( sub_BA8CB0(v10, (__int64)v44, v45) )
    {
      v11 = *v9;
      v12 = (size_t)v9[1];
      v29[0] = v30;
      if ( &v11[v12] && !v11 )
        sub_426248((__int64)"basic_string::_M_construct null not valid");
      v27 = v12;
      if ( v12 > 0xF )
      {
        src = v11;
        n = v12;
        v22 = sub_22409D0(v29, &v27, 0);
        v12 = n;
        v29[0] = v22;
        v23 = (_QWORD *)v22;
        v11 = src;
        v30[0] = v27;
      }
      else
      {
        if ( v12 == 1 )
        {
          LOBYTE(v30[0]) = *v11;
          v13 = v30;
LABEL_15:
          v29[1] = v12;
          *((_BYTE *)v13 + v12) = 0;
          v14 = (const __m128i *)*(unsigned int *)(a2 + 8);
          v15 = (__int64)v14->m128i_i64 + 1;
          v7 = v14;
          if ( (unsigned __int64)v14->m128i_u64 + 1 > *(unsigned int *)(a2 + 12) )
          {
            if ( *(_QWORD *)a2 > (unsigned __int64)v29 || (unsigned __int64)v29 >= *(_QWORD *)a2 + 32 * (__int64)v14 )
            {
              sub_95D880(a2, v15);
              v14 = (const __m128i *)*(unsigned int *)(a2 + 8);
              v16 = *(_QWORD *)a2;
              v7 = v14;
              v17 = (__m128i *)v29;
            }
            else
            {
              v26 = *(_QWORD *)a2;
              sub_95D880(a2, v15);
              v16 = *(_QWORD *)a2;
              v14 = (const __m128i *)*(unsigned int *)(a2 + 8);
              v7 = v14;
              v17 = (__m128i *)((char *)v29 + *(_QWORD *)a2 - v26);
            }
          }
          else
          {
            v16 = *(_QWORD *)a2;
            v17 = (__m128i *)v29;
          }
          v18 = (__m128i *)(v16 + 32LL * (_QWORD)v14);
          if ( v18 )
          {
            v18->m128i_i64[0] = (__int64)v18[1].m128i_i64;
            if ( (__m128i *)v17->m128i_i64[0] == &v17[1] )
            {
              v18[1] = _mm_loadu_si128(v17 + 1);
            }
            else
            {
              v18->m128i_i64[0] = v17->m128i_i64[0];
              v18[1].m128i_i64[0] = v17[1].m128i_i64[0];
            }
            v19 = v17->m128i_i64[1];
            v17->m128i_i64[0] = (__int64)v17[1].m128i_i64;
            v17->m128i_i64[1] = 0;
            v18->m128i_i64[1] = v19;
            v17[1].m128i_i8[0] = 0;
            v7 = (const __m128i *)*(unsigned int *)(a2 + 8);
          }
          v20 = (_QWORD *)v29[0];
          *(_DWORD *)(a2 + 8) = (_DWORD)v7 + 1;
          if ( v20 != v30 )
          {
            v7 = (const __m128i *)(v30[0] + 1LL);
            j_j___libc_free_0(v20, v30[0] + 1LL);
          }
          goto LABEL_23;
        }
        if ( !v12 )
        {
          v13 = v30;
          goto LABEL_15;
        }
        v23 = v30;
      }
      memcpy(v23, v11, v12);
      v12 = v27;
      v13 = (_QWORD *)v29[0];
      goto LABEL_15;
    }
LABEL_23:
    if ( !v47 )
      goto LABEL_7;
    v47 = 0;
    if ( v44 != (const __m128i *)&v46 )
    {
      v7 = (const __m128i *)(v46 + 1);
      j_j___libc_free_0(v44, v46 + 1);
    }
    if ( v42 != &v43 )
    {
      v7 = (const __m128i *)(v43 + 1);
      j_j___libc_free_0(v42, v43 + 1);
    }
    if ( v40 == &v41 )
      goto LABEL_7;
    _libc_free(v40, v7);
    v9 += 2;
  }
  while ( v8 != (const __m128i **)v9 );
LABEL_30:
  v8 = v34;
LABEL_31:
  if ( v8 != &v36 )
    _libc_free(v8, v7);
  v21 = 16LL * v33;
  result = sub_C7D6A0(v32, v21, 8);
  if ( v36 != (const __m128i *)v38 )
    return _libc_free(v36, v21);
  return result;
}
