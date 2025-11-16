// Function: sub_38B5500
// Address: 0x38b5500
//
__int64 __fastcall sub_38B5500(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // r14
  __int64 v7; // rbx
  __int64 v8; // rdi
  __int64 v9; // rcx
  unsigned __int64 v10; // rsi
  unsigned __int8 v11; // al
  __m128i *v12; // r8
  bool v13; // zf
  const __m128i *v14; // rdx
  __int64 v15; // rax
  void *v16; // rcx
  void *v17; // r10
  size_t v18; // rdx
  void *v19; // rax
  __int64 *v20; // r14
  const __m128i *v21; // r12
  __m128i *v22; // rax
  __m128i *v23; // rcx
  unsigned int *v24; // r12
  __int64 v25; // rbx
  __int64 v26; // r13
  unsigned __int64 *v27; // rax
  __int64 v28; // rdx
  unsigned __int8 v29; // [rsp+7h] [rbp-D9h]
  unsigned __int64 v31; // [rsp+18h] [rbp-C8h]
  __int64 i; // [rsp+20h] [rbp-C0h]
  unsigned __int64 v33; // [rsp+28h] [rbp-B8h]
  __m128i *v34; // [rsp+30h] [rbp-B0h]
  unsigned __int64 *v35; // [rsp+30h] [rbp-B0h]
  __m128i *v36; // [rsp+38h] [rbp-A8h]
  size_t v37; // [rsp+38h] [rbp-A8h]
  const __m128i *v38; // [rsp+40h] [rbp-A0h]
  void *v39; // [rsp+40h] [rbp-A0h]
  __int32 v40; // [rsp+40h] [rbp-A0h]
  unsigned int *v41; // [rsp+48h] [rbp-98h]
  __m128i v42; // [rsp+50h] [rbp-90h] BYREF
  void *src; // [rsp+60h] [rbp-80h]
  _BYTE *v44; // [rsp+68h] [rbp-78h]
  __int64 v45; // [rsp+70h] [rbp-70h]
  __int64 v46; // [rsp+80h] [rbp-60h] BYREF
  int v47; // [rsp+88h] [rbp-58h] BYREF
  _QWORD *v48; // [rsp+90h] [rbp-50h]
  int *v49; // [rsp+98h] [rbp-48h]
  int *v50; // [rsp+A0h] [rbp-40h]
  __int64 v51; // [rsp+A8h] [rbp-38h]

  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  if ( (unsigned __int8)sub_388AF10(a1, 16, "expected ':' here")
    || (unsigned __int8)sub_388AF10(a1, 12, "expected '(' here") )
  {
    return 1;
  }
  else
  {
    v47 = 0;
    v49 = &v47;
    v50 = &v47;
    v48 = 0;
    v51 = 0;
    v5 = a3;
    v6 = a1 + 8;
    v7 = v5;
    while ( 1 )
    {
      v8 = a1;
      v9 = (__int64)(*(_QWORD *)(v7 + 8) - *(_QWORD *)v7) >> 3;
      v10 = (unsigned __int64)&v42;
      src = 0;
      v44 = 0;
      v45 = 0;
      v11 = sub_38B54B0(a1, v42.m128i_i64, &v46, -858993459 * (int)v9);
      if ( v11 )
      {
        v29 = v11;
        if ( src )
          j_j___libc_free_0((unsigned __int64)src);
        goto LABEL_40;
      }
      v12 = *(__m128i **)(v7 + 8);
      if ( v12 == *(__m128i **)(v7 + 16) )
      {
        sub_142E3D0(v7, *(const __m128i **)(v7 + 8), &v42);
        v17 = src;
      }
      else
      {
        if ( v12 )
        {
          *v12 = _mm_loadu_si128(&v42);
          v14 = (const __m128i *)(v44 - (_BYTE *)src);
          v13 = v44 == src;
          v12[1].m128i_i64[0] = 0;
          v12[1].m128i_i64[1] = 0;
          v12[2].m128i_i64[0] = 0;
          if ( v13 )
          {
            v16 = 0;
          }
          else
          {
            if ( (unsigned __int64)v14 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_49:
              sub_4261EA(v8, v10, v14);
            v36 = v12;
            v38 = v14;
            v15 = sub_22077B0((unsigned __int64)v14);
            v14 = v38;
            v12 = v36;
            v16 = (void *)v15;
          }
          v12[1].m128i_i64[0] = (__int64)v16;
          v12[2].m128i_i64[0] = (__int64)v14->m128i_i64 + (_QWORD)v16;
          v12[1].m128i_i64[1] = (__int64)v16;
          v17 = src;
          v18 = v44 - (_BYTE *)src;
          if ( v44 != src )
          {
            v34 = v12;
            v37 = v44 - (_BYTE *)src;
            v39 = src;
            v19 = memmove(v16, src, v18);
            v12 = v34;
            v18 = v37;
            v17 = v39;
            v16 = v19;
          }
          v12[1].m128i_i64[1] = (__int64)v16 + v18;
          v12 = *(__m128i **)(v7 + 8);
        }
        else
        {
          v17 = src;
        }
        *(_QWORD *)(v7 + 8) = (char *)v12 + 40;
      }
      if ( v17 )
        j_j___libc_free_0((unsigned __int64)v17);
      if ( *(_DWORD *)(a1 + 64) != 4 )
        break;
      *(_DWORD *)(a1 + 64) = sub_3887100(v6);
    }
    v8 = a1;
    v10 = 13;
    v20 = (__int64 *)v7;
    v29 = sub_388AF10(a1, 13, "expected ')' here");
    if ( !v29 )
    {
      for ( i = (__int64)v49; (int *)i != &v47; i = sub_220EEE0(i) )
      {
        v21 = *(const __m128i **)(i + 48);
        v14 = *(const __m128i **)(i + 40);
        v40 = *(_DWORD *)(i + 32);
        v31 = (char *)v21 - (char *)v14;
        if ( v21 == v14 )
        {
          v33 = 0;
          if ( v21 == v14 )
            goto LABEL_39;
        }
        else
        {
          if ( v31 > 0x7FFFFFFFFFFFFFF0LL )
            goto LABEL_49;
          v33 = sub_22077B0(v31);
          v21 = *(const __m128i **)(i + 48);
          v14 = *(const __m128i **)(i + 40);
          if ( v14 == v21 )
            goto LABEL_37;
        }
        v22 = (__m128i *)v33;
        v23 = (__m128i *)(v33 + (char *)v21 - (char *)v14);
        v41 = (unsigned int *)v23;
        do
        {
          if ( v22 )
            *v22 = _mm_loadu_si128(v14);
          ++v22;
          ++v14;
        }
        while ( v22 != v23 );
        v24 = (unsigned int *)v33;
        if ( v23 == (__m128i *)v33 )
        {
LABEL_38:
          v10 = v31;
          j_j___libc_free_0(v33);
          goto LABEL_39;
        }
        do
        {
          while ( 1 )
          {
            v42.m128i_i64[1] = 0;
            src = 0;
            v25 = *v24;
            v42.m128i_i32[0] = v40;
            v26 = *((_QWORD *)v24 + 1);
            v44 = 0;
            v27 = (unsigned __int64 *)sub_38917E0((_QWORD *)(a1 + 1344), v42.m128i_i32);
            if ( v42.m128i_i64[1] )
            {
              v35 = v27;
              j_j___libc_free_0(v42.m128i_u64[1]);
              v27 = v35;
            }
            v28 = *v20;
            v42.m128i_i64[1] = v26;
            v42.m128i_i64[0] = v28 + 40 * v25;
            v10 = v27[6];
            if ( v10 != v27[7] )
              break;
            v24 += 4;
            sub_38952E0(v27 + 5, (const __m128i *)v10, &v42);
            if ( v41 == v24 )
              goto LABEL_37;
          }
          if ( v10 )
          {
            *(__m128i *)v10 = _mm_loadu_si128(&v42);
            v10 = v27[6];
          }
          v10 += 16LL;
          v24 += 4;
          v27[6] = v10;
        }
        while ( v41 != v24 );
LABEL_37:
        if ( v33 )
          goto LABEL_38;
LABEL_39:
        v8 = i;
      }
    }
LABEL_40:
    sub_3889030(v48);
  }
  return v29;
}
