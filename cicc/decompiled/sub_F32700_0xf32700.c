// Function: sub_F32700
// Address: 0xf32700
//
__int64 __fastcall sub_F32700(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r8
  __int64 v5; // r9
  int v6; // edx
  __int64 result; // rax
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rsi
  __int64 v11; // rcx
  __m128i v12; // xmm1
  unsigned __int8 v13; // dl
  bool v14; // dl
  __int64 *v15; // r13
  __int64 *v16; // r12
  __int64 *v17; // rdi
  __int64 *v18; // r14
  __int64 v19; // rbx
  __int64 v20; // rdi
  void *v21; // rax
  __m128i *v22; // rdx
  __int64 v23; // r13
  unsigned __int8 v24; // al
  __int64 v25; // rax
  __int64 v26; // rax
  void *v27; // rdx
  __int64 *v28; // r13
  __int64 *v29; // rdi
  __int64 *v30; // r14
  __int64 v31; // rbx
  __int64 v32; // rdi
  __int64 v33; // r14
  __int64 v34; // rdx
  __int64 v35; // rcx
  __int64 v36; // r8
  __int64 v37; // r9
  __int64 v38; // r13
  __m128i *v39; // rax
  __m128i v40; // xmm2
  __int64 v41; // rdi
  int v42; // r15d
  __int64 v43; // [rsp+8h] [rbp-A8h] BYREF
  unsigned __int8 *v44; // [rsp+10h] [rbp-A0h] BYREF
  size_t v45; // [rsp+18h] [rbp-98h]
  __int64 v46; // [rsp+20h] [rbp-90h] BYREF
  __m128i v47; // [rsp+30h] [rbp-80h] BYREF
  __int64 *v48; // [rsp+40h] [rbp-70h] BYREF
  unsigned int v49; // [rsp+48h] [rbp-68h]
  _BYTE v50[40]; // [rsp+50h] [rbp-60h] BYREF
  unsigned __int8 v51; // [rsp+78h] [rbp-38h]

  v45 = 0;
  sub_109B500(&v47, a2, a3, v44, 0);
  v6 = v51 & 1;
  result = (2 * v6) | v51 & 0xFDu;
  v51 = (2 * v6) | v51 & 0xFD;
  if ( (_BYTE)v6 )
  {
    v21 = sub_CB72A0();
    v22 = (__m128i *)*((_QWORD *)v21 + 4);
    v23 = (__int64)v21;
    if ( *((_QWORD *)v21 + 3) - (_QWORD)v22 <= 0x1Fu )
    {
      v23 = sub_CB6200((__int64)v21, "WARNING: when loading pattern: '", 0x20u);
    }
    else
    {
      *v22 = _mm_load_si128((const __m128i *)&xmmword_3F89B60);
      v22[1] = _mm_load_si128((const __m128i *)&xmmword_3F89B70);
      *((_QWORD *)v21 + 4) += 32LL;
    }
    v24 = v51;
    v51 &= ~2u;
    if ( (v24 & 1) != 0 )
    {
      v25 = v47.m128i_i64[0];
      v47.m128i_i64[0] = 0;
      v43 = v25 | 1;
    }
    else
    {
      v43 = 1;
    }
    sub_C64870((__int64)&v44, &v43);
    v10 = (__int64)v44;
    v26 = sub_CB6200(v23, v44, v45);
    v27 = *(void **)(v26 + 32);
    if ( *(_QWORD *)(v26 + 24) - (_QWORD)v27 <= 9u )
    {
      v10 = (__int64)"' ignoring";
      sub_CB6200(v26, "' ignoring", 0xAu);
    }
    else
    {
      qmemcpy(v27, "' ignoring", 10);
      *(_QWORD *)(v26 + 32) += 10LL;
    }
    if ( v44 != (unsigned __int8 *)&v46 )
    {
      v10 = v46 + 1;
      j_j___libc_free_0(v44, v46 + 1);
    }
    if ( (v43 & 1) != 0 || (v43 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      sub_C63C30(&v43, v10);
    result = v51;
    if ( (v51 & 2) == 0 )
    {
      if ( (v51 & 1) == 0 )
      {
        v28 = v48;
        v16 = &v48[5 * v49];
        if ( v48 == v16 )
        {
LABEL_21:
          result = (__int64)v50;
          if ( v16 != (__int64 *)v50 )
            return _libc_free(v16, v10);
          return result;
        }
        do
        {
          v16 -= 5;
          v29 = (__int64 *)v16[2];
          if ( v29 != v16 + 5 )
            _libc_free(v29, v10);
          v30 = (__int64 *)*v16;
          v31 = *v16 + 80LL * *((unsigned int *)v16 + 2);
          if ( *v16 != v31 )
          {
            do
            {
              v31 -= 80;
              v32 = *(_QWORD *)(v31 + 8);
              if ( v32 != v31 + 24 )
                _libc_free(v32, v10);
            }
            while ( v30 != (__int64 *)v31 );
            v30 = (__int64 *)*v16;
          }
          if ( v30 != v16 + 2 )
            _libc_free(v30, v10);
        }
        while ( v28 != v16 );
LABEL_20:
        v16 = v48;
        goto LABEL_21;
      }
      goto LABEL_25;
    }
    goto LABEL_54;
  }
  v8 = *(unsigned int *)(a1 + 8);
  v9 = v8;
  if ( *(_DWORD *)(a1 + 12) <= (unsigned int)v8 )
  {
    v33 = a1 + 16;
    v38 = sub_C8D7D0(a1, a1 + 16, 0, 0x48u, (unsigned __int64 *)&v44, v5);
    v39 = (__m128i *)(v38 + 72LL * *(unsigned int *)(a1 + 8));
    if ( v39 )
    {
      v40 = _mm_loadu_si128(&v47);
      v39[1].m128i_i64[0] = (__int64)v39[2].m128i_i64;
      v39[1].m128i_i64[1] = 0x100000000LL;
      *v39 = v40;
      v34 = v49;
      if ( v49 )
        sub_F31AD0((__int64)v39[1].m128i_i64, (__int64)&v48, v49, v35, v36, v37);
    }
    v10 = v38;
    sub_F32260(a1, v38, v34, v35, v36, v37);
    v41 = *(_QWORD *)a1;
    v42 = (int)v44;
    if ( v33 != *(_QWORD *)a1 )
      _libc_free(v41, v38);
    result = v51;
    ++*(_DWORD *)(a1 + 8);
    *(_QWORD *)a1 = v38;
    *(_DWORD *)(a1 + 12) = v42;
    v14 = (result & 2) != 0;
  }
  else
  {
    v10 = 9 * v8;
    v11 = *(_QWORD *)a1 + 72 * v8;
    if ( !v11 )
    {
      *(_DWORD *)(a1 + 8) = v9 + 1;
      if ( (result & 1) == 0 )
        goto LABEL_9;
      goto LABEL_25;
    }
    v12 = _mm_loadu_si128(&v47);
    *(_QWORD *)(v11 + 16) = v11 + 32;
    *(_QWORD *)(v11 + 24) = 0x100000000LL;
    *(__m128i *)v11 = v12;
    if ( v49 )
    {
      v10 = (__int64)&v48;
      sub_F31AD0(v11 + 16, (__int64)&v48, v9, v11, v4, v5);
    }
    result = v51;
    v13 = v51;
    ++*(_DWORD *)(a1 + 8);
    v14 = (v13 & 2) != 0;
  }
  if ( v14 )
LABEL_54:
    sub_F30080(&v47, v10);
  if ( (result & 1) == 0 )
  {
LABEL_9:
    v15 = v48;
    v16 = &v48[5 * v49];
    if ( v48 == v16 )
      goto LABEL_21;
    do
    {
      v16 -= 5;
      v17 = (__int64 *)v16[2];
      if ( v17 != v16 + 5 )
        _libc_free(v17, v10);
      v18 = (__int64 *)*v16;
      v19 = *v16 + 80LL * *((unsigned int *)v16 + 2);
      if ( *v16 != v19 )
      {
        do
        {
          v19 -= 80;
          v20 = *(_QWORD *)(v19 + 8);
          if ( v20 != v19 + 24 )
            _libc_free(v20, v10);
        }
        while ( v18 != (__int64 *)v19 );
        v18 = (__int64 *)*v16;
      }
      if ( v18 != v16 + 2 )
        _libc_free(v18, v10);
    }
    while ( v15 != v16 );
    goto LABEL_20;
  }
LABEL_25:
  if ( v47.m128i_i64[0] )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v47.m128i_i64[0] + 8LL))(v47.m128i_i64[0]);
  return result;
}
