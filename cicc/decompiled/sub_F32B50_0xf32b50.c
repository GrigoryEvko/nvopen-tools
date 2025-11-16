// Function: sub_F32B50
// Address: 0xf32b50
//
__int64 __fastcall sub_F32B50(__int64 a1, const char *m128i_i8)
{
  size_t v2; // rbx
  __int64 v3; // r12
  __int64 i; // rbx
  __int64 v5; // rdx
  const __m128i **v6; // rax
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r13
  __int64 v10; // rbx
  unsigned int v11; // r14d
  const __m128i *v12; // r14
  __int64 v13; // rax
  const __m128i *v14; // r13
  const __m128i *v15; // rbx
  __int64 *v16; // rdi
  __int64 *v17; // r15
  __int64 v18; // r12
  __int64 v19; // rdi
  unsigned __int8 *v21; // r14
  __int64 v22; // rsi
  __int64 v23; // rbx
  __int64 v24; // rax
  volatile signed __int32 *v25; // rdx
  signed __int32 v26; // eax
  volatile signed __int32 *v27; // r12
  signed __int32 v28; // eax
  const __m128i *v29; // rax
  __int64 v30; // rdx
  const __m128i *v31; // rcx
  const __m128i *v32; // r12
  __m128i v33; // xmm1
  int v34; // eax
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  int v39; // r12d
  __int64 v40; // rcx
  __int64 v41; // rax
  const __m128i *v42; // rcx
  const __m128i *v43; // r14
  const __m128i *v44; // rbx
  __int64 *v45; // rdi
  __int64 *v46; // r12
  __int64 v47; // r15
  __int64 v48; // rdi
  _QWORD *v49; // r12
  __m128i *v50; // rax
  __m128i si128; // xmm0
  void *v52; // rdi
  __m128i *v53; // rbx
  signed __int32 v54; // eax
  __int64 v55; // rax
  signed __int32 v56; // eax
  __int64 v57; // rax
  volatile signed __int32 *v58; // r13
  signed __int32 v59; // eax
  signed __int32 v60; // eax
  const __m128i *v62; // [rsp+10h] [rbp-110h]
  const __m128i *v63; // [rsp+18h] [rbp-108h]
  const __m128i *v64; // [rsp+18h] [rbp-108h]
  __int64 v65; // [rsp+20h] [rbp-100h] BYREF
  char v66; // [rsp+30h] [rbp-F0h]
  __int64 v67[4]; // [rsp+40h] [rbp-E0h] BYREF
  __int16 v68; // [rsp+60h] [rbp-C0h]
  const char *v69; // [rsp+70h] [rbp-B0h]
  __int64 v70; // [rsp+78h] [rbp-A8h]
  const __m128i *v71; // [rsp+80h] [rbp-A0h] BYREF
  __int64 v72; // [rsp+88h] [rbp-98h]
  _BYTE v73[72]; // [rsp+90h] [rbp-90h] BYREF
  __int64 v74; // [rsp+D8h] [rbp-48h] BYREF
  volatile signed __int32 *v75; // [rsp+E0h] [rbp-40h]

  v2 = qword_4F8BD50;
  v71 = (const __m128i *)v73;
  *(_BYTE *)a1 = 0;
  v72 = 0x100000000LL;
  v74 = 0;
  v75 = 0;
  if ( v2 )
  {
    v21 = (unsigned __int8 *)qword_4F8BD48;
    v68 = 261;
    v67[1] = v2;
    v67[0] = (__int64)qword_4F8BD48;
    sub_C7EA90((__int64)&v65, v67, 0, 1u, 0, 0);
    if ( (v66 & 1) != 0 )
    {
      v49 = sub_CB72A0();
      v50 = (__m128i *)v49[4];
      if ( v49[3] - (_QWORD)v50 <= 0x28u )
      {
        v57 = sub_CB6200((__int64)v49, "WARNING: Internalize couldn't load file '", 0x29u);
        v52 = *(void **)(v57 + 32);
        v49 = (_QWORD *)v57;
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_3F89B80);
        v50[2].m128i_i8[8] = 39;
        v50[2].m128i_i64[0] = 0x20656C6966206461LL;
        *v50 = si128;
        v50[1] = _mm_load_si128((const __m128i *)&xmmword_3F89B90);
        v52 = (void *)(v49[4] + 41LL);
        v49[4] = v52;
      }
      m128i_i8 = (const char *)v21;
      if ( v2 > v49[3] - (_QWORD)v52 )
      {
        v55 = sub_CB6200((__int64)v49, v21, v2);
        v53 = *(__m128i **)(v55 + 32);
        v49 = (_QWORD *)v55;
      }
      else
      {
        memcpy(v52, v21, v2);
        v53 = (__m128i *)(v49[4] + v2);
        v49[4] = v53;
      }
      if ( v49[3] - (_QWORD)v53 <= 0x1Fu )
      {
        m128i_i8 = "'! Continuing as if it's empty.\n";
        sub_CB6200((__int64)v49, "'! Continuing as if it's empty.\n", 0x20u);
      }
      else
      {
        *v53 = _mm_load_si128((const __m128i *)&xmmword_3F89BA0);
        v53[1] = _mm_load_si128((const __m128i *)&xmmword_3F89BB0);
        v49[4] += 32LL;
      }
    }
    else
    {
      v22 = v65;
      if ( v65 && (v22 = v65, v23 = sub_22077B0(24), v24 = v65, v65 = 0, v23) )
      {
        *(_QWORD *)(v23 + 16) = v24;
        *(_QWORD *)(v23 + 8) = 0x100000001LL;
        *(_QWORD *)v23 = &unk_49E5010;
        v25 = (volatile signed __int32 *)(v23 + 8);
        if ( &_pthread_key_create )
        {
          _InterlockedAdd(v25, 1u);
          v26 = _InterlockedExchangeAdd(v25, 0xFFFFFFFF);
        }
        else
        {
          v26 = ++*(_DWORD *)(v23 + 8);
          *(_DWORD *)(v23 + 8) = v26 - 1;
        }
        if ( v26 == 1 )
        {
          (*(void (__fastcall **)(__int64))(*(_QWORD *)v23 + 16LL))(v23);
          if ( &_pthread_key_create )
          {
            v54 = _InterlockedExchangeAdd((volatile signed __int32 *)(v23 + 12), 0xFFFFFFFF);
          }
          else
          {
            v54 = *(_DWORD *)(v23 + 12);
            *(_DWORD *)(v23 + 12) = v54 - 1;
          }
          if ( v54 == 1 )
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v23 + 24LL))(v23);
        }
      }
      else
      {
        v23 = 0;
      }
      v27 = v75;
      v74 = v22;
      v75 = (volatile signed __int32 *)v23;
      if ( v27 )
      {
        if ( &_pthread_key_create )
        {
          v28 = _InterlockedExchangeAdd(v27 + 2, 0xFFFFFFFF);
        }
        else
        {
          v28 = *((_DWORD *)v27 + 2);
          *((_DWORD *)v27 + 2) = v28 - 1;
        }
        if ( v28 == 1 )
        {
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v27 + 16LL))(v27);
          if ( &_pthread_key_create )
          {
            v56 = _InterlockedExchangeAdd(v27 + 3, 0xFFFFFFFF);
          }
          else
          {
            v56 = *((_DWORD *)v27 + 3);
            *((_DWORD *)v27 + 3) = v56 - 1;
          }
          if ( v56 == 1 )
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v27 + 24LL))(v27);
        }
        v22 = v74;
      }
      sub_C7C840((__int64)v67, v22, 1, 0);
      while ( 1 )
      {
        m128i_i8 = v69;
        if ( !(_BYTE)v68 && !v69 )
          break;
        sub_F32700((__int64)&v71, (__int64)v69, v70);
        sub_C7C5C0((__int64)v67);
      }
    }
    if ( (v66 & 1) == 0 && v65 )
      (*(void (__fastcall **)(__int64, const char *))(*(_QWORD *)v65 + 8LL))(v65, m128i_i8);
  }
  v3 = qword_4F8BC50;
  for ( i = qword_4F8BC48; v3 != i; i += 32 )
  {
    m128i_i8 = *(const char **)i;
    v5 = *(_QWORD *)(i + 8);
    sub_F32700((__int64)&v71, (__int64)m128i_i8, v5);
  }
  *(_QWORD *)(a1 + 24) = 0;
  v6 = (const __m128i **)sub_22077B0(104);
  v9 = (__int64)v6;
  if ( v6 )
  {
    v10 = (__int64)(v6 + 2);
    v11 = v72;
    *v6 = (const __m128i *)(v6 + 2);
    v6[1] = (const __m128i *)0x100000000LL;
    v63 = v71;
    if ( !v11 )
    {
LABEL_6:
      *(_QWORD *)(v9 + 88) = v74;
      *(_QWORD *)(v9 + 96) = v75;
      *(_QWORD *)(a1 + 32) = sub_F2FBF0;
      *(_QWORD *)(a1 + 8) = v9;
      *(_QWORD *)(a1 + 24) = sub_F323D0;
      goto LABEL_7;
    }
    if ( v71 != (const __m128i *)v73 )
    {
      *v6 = v71;
      v34 = HIDWORD(v72);
      *(_DWORD *)(v9 + 8) = v11;
      *(_DWORD *)(v9 + 12) = v34;
      v71 = (const __m128i *)v73;
      v72 = 0;
      v63 = (const __m128i *)v73;
      goto LABEL_6;
    }
    v29 = (const __m128i *)v73;
    v30 = v11;
    v31 = (const __m128i *)&v74;
    if ( v11 != 1 )
    {
      m128i_i8 = (const char *)sub_C8D7D0(v9, v10, v11, 0x48u, (unsigned __int64 *)v67, v8);
      sub_F32260(v9, (__int64)m128i_i8, v35, v36, v37, v38);
      v39 = v67[0];
      v30 = (__int64)m128i_i8;
      if ( v10 != *(_QWORD *)v9 )
      {
        _libc_free(*(_QWORD *)v9, m128i_i8);
        v30 = (__int64)m128i_i8;
      }
      v40 = (unsigned int)v72;
      v29 = v71;
      *(_QWORD *)v9 = v30;
      *(_DWORD *)(v9 + 12) = v39;
      v63 = v29;
      v31 = (const __m128i *)((char *)v29 + 72 * v40);
      if ( v29 == v31 )
      {
        *(_DWORD *)(v9 + 8) = v11;
LABEL_75:
        LODWORD(v72) = 0;
        goto LABEL_6;
      }
      v10 = v30;
    }
    v32 = v29;
    do
    {
      if ( v10 )
      {
        v33 = _mm_loadu_si128(v32);
        *(_DWORD *)(v10 + 24) = 0;
        *(_QWORD *)(v10 + 16) = v10 + 32;
        *(_DWORD *)(v10 + 28) = 1;
        *(__m128i *)v10 = v33;
        if ( v32[1].m128i_i32[2] )
        {
          m128i_i8 = v32[1].m128i_i8;
          v64 = v31;
          sub_F31AD0(v10 + 16, (__int64)v32[1].m128i_i64, v30, (__int64)v31, v7, v8);
          v31 = v64;
        }
      }
      v32 = (const __m128i *)((char *)v32 + 72);
      v10 += 72;
    }
    while ( v31 != v32 );
    v41 = (unsigned int)v72;
    v42 = v71;
    *(_DWORD *)(v9 + 8) = v11;
    v63 = v42;
    v62 = (const __m128i *)((char *)v42 + 72 * v41);
    if ( v62 != v42 )
    {
      do
      {
        v62 = (const __m128i *)((char *)v62 - 72);
        v43 = (const __m128i *)v62[1].m128i_i64[0];
        v44 = (const __m128i *)((char *)v43 + 40 * v62[1].m128i_u32[2]);
        if ( v43 != v44 )
        {
          do
          {
            v44 = (const __m128i *)((char *)v44 - 40);
            v45 = (__int64 *)v44[1].m128i_i64[0];
            if ( v45 != (__int64 *)&v44[2].m128i_u64[1] )
              _libc_free(v45, m128i_i8);
            v46 = (__int64 *)v44->m128i_i64[0];
            v47 = v44->m128i_i64[0] + 80LL * v44->m128i_u32[2];
            if ( v44->m128i_i64[0] != v47 )
            {
              do
              {
                v47 -= 80;
                v48 = *(_QWORD *)(v47 + 8);
                if ( v48 != v47 + 24 )
                  _libc_free(v48, m128i_i8);
              }
              while ( v46 != (__int64 *)v47 );
              v46 = (__int64 *)v44->m128i_i64[0];
            }
            if ( v46 != (__int64 *)&v44[1] )
              _libc_free(v46, m128i_i8);
          }
          while ( v43 != v44 );
          v43 = (const __m128i *)v62[1].m128i_i64[0];
        }
        if ( v43 != &v62[2] )
          _libc_free(v43, m128i_i8);
      }
      while ( v62 != v63 );
      v63 = v71;
    }
    goto LABEL_75;
  }
  v58 = v75;
  *(_QWORD *)(a1 + 32) = sub_F2FBF0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 24) = sub_F323D0;
  if ( !v58 )
    goto LABEL_104;
  if ( &_pthread_key_create )
  {
    v59 = _InterlockedExchangeAdd(v58 + 2, 0xFFFFFFFF);
  }
  else
  {
    v59 = *((_DWORD *)v58 + 2);
    *((_DWORD *)v58 + 2) = v59 - 1;
  }
  v63 = v71;
  if ( v59 == 1 )
  {
    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v58 + 16LL))(v58);
    if ( &_pthread_key_create )
    {
      v60 = _InterlockedExchangeAdd(v58 + 3, 0xFFFFFFFF);
    }
    else
    {
      v60 = *((_DWORD *)v58 + 3);
      *((_DWORD *)v58 + 3) = v60 - 1;
    }
    v63 = v71;
    if ( v60 == 1 )
    {
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v58 + 24LL))(v58);
LABEL_104:
      v63 = v71;
    }
  }
LABEL_7:
  v12 = (const __m128i *)((char *)v63 + 72 * (unsigned int)v72);
  if ( v63 != v12 )
  {
    do
    {
      v13 = v12[-3].m128i_u32[0];
      v14 = (const __m128i *)v12[-4].m128i_i64[1];
      v12 = (const __m128i *)((char *)v12 - 72);
      v15 = (const __m128i *)((char *)v14 + 40 * v13);
      if ( v14 != v15 )
      {
        do
        {
          v15 = (const __m128i *)((char *)v15 - 40);
          v16 = (__int64 *)v15[1].m128i_i64[0];
          if ( v16 != (__int64 *)&v15[2].m128i_u64[1] )
            _libc_free(v16, m128i_i8);
          v17 = (__int64 *)v15->m128i_i64[0];
          v18 = v15->m128i_i64[0] + 80LL * v15->m128i_u32[2];
          if ( v15->m128i_i64[0] != v18 )
          {
            do
            {
              v18 -= 80;
              v19 = *(_QWORD *)(v18 + 8);
              if ( v19 != v18 + 24 )
                _libc_free(v19, m128i_i8);
            }
            while ( v17 != (__int64 *)v18 );
            v17 = (__int64 *)v15->m128i_i64[0];
          }
          if ( v17 != (__int64 *)&v15[1] )
            _libc_free(v17, m128i_i8);
        }
        while ( v14 != v15 );
        v14 = (const __m128i *)v12[1].m128i_i64[0];
      }
      if ( v14 != &v12[2] )
        _libc_free(v14, m128i_i8);
    }
    while ( v63 != v12 );
    v12 = v71;
  }
  if ( v12 != (const __m128i *)v73 )
    _libc_free(v12, m128i_i8);
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0x800000000LL;
  return a1;
}
