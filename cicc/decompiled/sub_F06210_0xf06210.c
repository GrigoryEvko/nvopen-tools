// Function: sub_F06210
// Address: 0xf06210
//
void __fastcall sub_F06210(__int64 a1, char *a2, char *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __m128i *v7; // rsi
  __m128i *v8; // rax
  __int64 v9; // rdx
  __m128i *v10; // r15
  __m128i *v11; // rbx
  __int64 v12; // rax
  const __m128i *v13; // r12
  __m128i *v14; // r14
  const __m128i *v15; // rcx
  const __m128i *v16; // rdi
  _BYTE *v17; // rdi
  __int64 v18; // rdx
  _BYTE *v19; // r12
  _BYTE *v20; // r15
  _QWORD *v21; // rax
  _BYTE *v22; // r9
  size_t v23; // r8
  __int64 v24; // rax
  _QWORD *v25; // rdi
  char *src; // [rsp+10h] [rbp-C0h]
  _BYTE *srca; // [rsp+10h] [rbp-C0h]
  size_t n; // [rsp+18h] [rbp-B8h]
  size_t na; // [rsp+18h] [rbp-B8h]
  char *v30[3]; // [rsp+20h] [rbp-B0h] BYREF
  size_t v31; // [rsp+38h] [rbp-98h] BYREF
  __m128i v32; // [rsp+40h] [rbp-90h] BYREF
  _QWORD v33[2]; // [rsp+50h] [rbp-80h] BYREF
  _BYTE *v34; // [rsp+60h] [rbp-70h] BYREF
  __int64 v35; // [rsp+68h] [rbp-68h]
  _BYTE v36[96]; // [rsp+70h] [rbp-60h] BYREF

  v30[0] = a2;
  v7 = (__m128i *)&v34;
  v30[1] = a3;
  v34 = v36;
  v35 = 0x300000000LL;
  sub_C93960(v30, (__int64)&v34, 44, -1, 0, a6);
  v8 = *(__m128i **)a1;
  v9 = (unsigned int)v35;
  if ( (unsigned int)v35 > (unsigned __int64)((__int64)(*(_QWORD *)(a1 + 16) - *(_QWORD *)a1) >> 5) )
  {
    v10 = *(__m128i **)(a1 + 8);
    n = (char *)v10 - (char *)v8;
    src = (char *)(32LL * (unsigned int)v35);
    v11 = 0;
    if ( (_DWORD)v35 )
    {
      v12 = sub_22077B0(32LL * (unsigned int)v35);
      v10 = *(__m128i **)(a1 + 8);
      v11 = (__m128i *)v12;
      v8 = *(__m128i **)a1;
    }
    if ( v8 != v10 )
    {
      v13 = v8 + 1;
      v14 = v11;
      while ( 1 )
      {
        if ( v14 )
        {
          v14->m128i_i64[0] = (__int64)v14[1].m128i_i64;
          v15 = (const __m128i *)v13[-1].m128i_i64[0];
          if ( v15 == v13 )
          {
            v14[1] = _mm_loadu_si128(v13);
          }
          else
          {
            v14->m128i_i64[0] = (__int64)v15;
            v14[1].m128i_i64[0] = v13->m128i_i64[0];
          }
          v14->m128i_i64[1] = v13[-1].m128i_i64[1];
          v13[-1].m128i_i64[0] = (__int64)v13;
        }
        else
        {
          v16 = (const __m128i *)v13[-1].m128i_i64[0];
          if ( v16 != v13 )
          {
            v7 = (__m128i *)(v13->m128i_i64[0] + 1);
            j_j___libc_free_0(v16, v7);
          }
        }
        v14 += 2;
        if ( v10 == &v13[1] )
          break;
        v13 += 2;
      }
      v10 = *(__m128i **)a1;
    }
    if ( v10 )
    {
      v7 = (__m128i *)(*(_QWORD *)(a1 + 16) - (_QWORD)v10);
      j_j___libc_free_0(v10, v7);
    }
    v9 = (unsigned int)v35;
    *(_QWORD *)a1 = v11;
    *(_QWORD *)(a1 + 8) = (char *)v11 + n;
    *(_QWORD *)(a1 + 16) = (char *)v11 + (_QWORD)src;
  }
  v17 = v34;
  v18 = 16 * v9;
  v19 = &v34[v18];
  if ( &v34[v18] != v34 )
  {
    v20 = v34;
    while ( 1 )
    {
      v22 = *(_BYTE **)v20;
      v23 = *((_QWORD *)v20 + 1);
      v32.m128i_i64[0] = (__int64)v33;
      if ( &v22[v23] && !v22 )
        sub_426248((__int64)"basic_string::_M_construct null not valid");
      v31 = v23;
      if ( v23 > 0xF )
        break;
      if ( v23 == 1 )
      {
        LOBYTE(v33[0]) = *v22;
        v21 = v33;
      }
      else
      {
        if ( v23 )
        {
          v25 = v33;
          goto LABEL_31;
        }
        v21 = v33;
      }
LABEL_21:
      v32.m128i_i64[1] = v23;
      v7 = &v32;
      *((_BYTE *)v21 + v23) = 0;
      sub_F06060((__m128i **)a1, &v32);
      if ( (_QWORD *)v32.m128i_i64[0] != v33 )
      {
        v7 = (__m128i *)(v33[0] + 1LL);
        j_j___libc_free_0(v32.m128i_i64[0], v33[0] + 1LL);
      }
      v20 += 16;
      if ( v19 == v20 )
      {
        v17 = v34;
        goto LABEL_33;
      }
    }
    srca = v22;
    na = v23;
    v24 = sub_22409D0(&v32, &v31, 0);
    v23 = na;
    v22 = srca;
    v32.m128i_i64[0] = v24;
    v25 = (_QWORD *)v24;
    v33[0] = v31;
LABEL_31:
    memcpy(v25, v22, v23);
    v23 = v31;
    v21 = (_QWORD *)v32.m128i_i64[0];
    goto LABEL_21;
  }
LABEL_33:
  if ( v17 != v36 )
    _libc_free(v17, v7);
}
