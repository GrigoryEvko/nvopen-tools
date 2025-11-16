// Function: sub_1C1C390
// Address: 0x1c1c390
//
__int64 __fastcall sub_1C1C390(__int64 a1, __int64 a2)
{
  void *v4; // rax
  char v5; // al
  _BOOL8 v6; // rcx
  char v7; // al
  _BOOL8 v8; // rcx
  char v9; // al
  _BOOL8 v10; // rcx
  char v11; // al
  _BOOL8 v12; // rcx
  char v13; // al
  _BOOL8 v14; // rcx
  char v15; // al
  _BOOL8 v16; // rcx
  char v17; // al
  _BOOL8 v18; // rcx
  __int64 result; // rax
  char v20; // al
  __int64 v21; // rsi
  __m128i *v22; // rax
  void *v23; // r8
  __int64 v24; // rcx
  __m128i *v25; // r13
  int v26; // eax
  __m128i *v27; // rsi
  unsigned int v28; // ecx
  void **p_src; // rax
  const __m128i *v30; // rdx
  _BYTE *v31; // rdi
  _BYTE *v32; // rax
  _BYTE *v33; // rdx
  size_t v34; // r13
  __int64 v35; // rax
  void *v36; // rax
  void *v37; // r12
  signed __int64 v38; // rsi
  unsigned int v39; // [rsp+0h] [rbp-70h]
  void **v40; // [rsp+8h] [rbp-68h]
  void *v41; // [rsp+8h] [rbp-68h]
  __int64 v42; // [rsp+8h] [rbp-68h]
  char v43; // [rsp+17h] [rbp-59h] BYREF
  __int64 v44; // [rsp+18h] [rbp-58h] BYREF
  void *src; // [rsp+20h] [rbp-50h] BYREF
  __m128i *v46; // [rsp+28h] [rbp-48h]
  __m128i *v47; // [rsp+30h] [rbp-40h]

  v4 = *(void **)a2;
  v46 = (__m128i *)a2;
  src = v4;
  v5 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v6 = 0;
  if ( v5 )
    v6 = src == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "pgoAppHash",
         0,
         v6,
         &v43,
         &v44) )
  {
    sub_1C141E0(a1, (__int64 *)&src);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v44);
  }
  else if ( v43 )
  {
    src = 0;
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    v46->m128i_i64[0] = (__int64)src;
  src = *(void **)(a2 + 8);
  v46 = (__m128i *)(a2 + 8);
  v7 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v8 = 0;
  if ( v7 )
    v8 = src == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "pgoProfileHash",
         0,
         v8,
         &v43,
         &v44) )
  {
    sub_1C141E0(a1, (__int64 *)&src);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v44);
  }
  else if ( v43 )
  {
    src = 0;
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    v46->m128i_i64[0] = (__int64)src;
  src = *(void **)(a2 + 16);
  v46 = (__m128i *)(a2 + 16);
  v9 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v10 = 0;
  if ( v9 )
    v10 = src == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "pgoOptionsHash",
         0,
         v10,
         &v43,
         &v44) )
  {
    sub_1C141E0(a1, (__int64 *)&src);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v44);
  }
  else if ( v43 )
  {
    src = 0;
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    v46->m128i_i64[0] = (__int64)src;
  src = *(void **)(a2 + 24);
  v46 = (__m128i *)(a2 + 24);
  v11 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v12 = 0;
  if ( v11 )
    v12 = src == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "oriIRHash",
         0,
         v12,
         &v43,
         &v44) )
  {
    sub_1C141E0(a1, (__int64 *)&src);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v44);
  }
  else if ( v43 )
  {
    src = 0;
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    v46->m128i_i64[0] = (__int64)src;
  LODWORD(src) = *(_DWORD *)(a2 + 32);
  v46 = (__m128i *)(a2 + 32);
  v13 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v14 = 0;
  if ( v13 )
    v14 = (_DWORD)src == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "N",
         0,
         v14,
         &v43,
         &v44) )
  {
    sub_1C14710(a1, (unsigned int *)&src);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v44);
  }
  else if ( v43 )
  {
    LODWORD(src) = 0;
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    v46->m128i_i32[0] = (int)src;
  LODWORD(src) = *(_DWORD *)(a2 + 36);
  v46 = (__m128i *)(a2 + 36);
  v15 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v16 = 0;
  if ( v15 )
    v16 = (_DWORD)src == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "M",
         0,
         v16,
         &v43,
         &v44) )
  {
    sub_1C14710(a1, (unsigned int *)&src);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v44);
  }
  else if ( v43 )
  {
    LODWORD(src) = 0;
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    v46->m128i_i32[0] = (int)src;
  LODWORD(src) = *(_DWORD *)(a2 + 40);
  v46 = (__m128i *)(a2 + 40);
  v17 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v18 = 0;
  if ( v17 )
    v18 = (_DWORD)src == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "numInvocations",
         0,
         v18,
         &v43,
         &v44) )
  {
    sub_1C14710(a1, (unsigned int *)&src);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v44);
  }
  else if ( v43 )
  {
    LODWORD(src) = 0;
  }
  result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  if ( !(_BYTE)result )
  {
    result = (__int64)v46;
    v46->m128i_i32[0] = (int)src;
  }
  if ( *(_DWORD *)(a2 + 36) )
  {
    v20 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    src = 0;
    v46 = 0;
    v47 = 0;
    if ( v20 )
    {
      v21 = *(unsigned int *)(a2 + 36);
      if ( !*(_DWORD *)(a2 + 36) )
      {
LABEL_63:
        result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 56LL))(a1);
        if ( !(_BYTE)result || (v31 = src, v46 != src) )
        {
          result = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
                     a1,
                     "ZeroPData",
                     0,
                     0,
                     &v43,
                     &v44);
          if ( (_BYTE)result )
          {
            sub_1C1C000(a1, (__int64 *)&src);
            result = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v44);
          }
          v31 = src;
        }
        if ( v31 )
          return j_j___libc_free_0(v31, (char *)v47 - v31);
        return result;
      }
      v22 = (__m128i *)sub_22077B0(16 * v21);
      v23 = src;
      v24 = 16 * v21;
      v25 = v22;
      if ( (char *)v46 - (_BYTE *)src > 0 )
      {
        v41 = src;
        memmove(v22, src, (char *)v46 - (_BYTE *)src);
        v23 = v41;
        v24 = 16 * v21;
        v38 = (char *)v47 - (_BYTE *)v41;
      }
      else
      {
        if ( !src )
          goto LABEL_55;
        v38 = (char *)v47 - (_BYTE *)src;
      }
      v42 = v24;
      j_j___libc_free_0(v23, v38);
      v24 = v42;
LABEL_55:
      v26 = *(_DWORD *)(a2 + 36);
      v27 = (__m128i *)((char *)v25 + v24);
      src = v25;
      v46 = v25;
      v47 = (__m128i *)((char *)v25 + v24);
      if ( v26 )
      {
        v28 = 0;
        p_src = &src;
        while ( 1 )
        {
          v30 = (const __m128i *)(*(_QWORD *)(a2 + 48) + 16LL * v28);
          if ( v27 == v25 )
          {
            v39 = v28;
            v40 = p_src;
            sub_CD12E0((__int64)p_src, v27, v30);
            p_src = v40;
            v28 = v39 + 1;
            if ( *(_DWORD *)(a2 + 36) <= v39 + 1 )
              goto LABEL_63;
          }
          else
          {
            if ( v25 )
            {
              *v25 = _mm_loadu_si128(v30);
              v25 = v46;
            }
            ++v28;
            v46 = v25 + 1;
            if ( *(_DWORD *)(a2 + 36) <= v28 )
              goto LABEL_63;
          }
          v25 = v46;
          v27 = v47;
        }
      }
      goto LABEL_63;
    }
    if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 56LL))(a1) && (v32 = src, v46 == src) )
    {
      v33 = src;
    }
    else
    {
      if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "ZeroPData",
             0,
             0,
             &v43,
             &v44) )
      {
        sub_1C1C000(a1, (__int64 *)&src);
        (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v44);
      }
      v32 = v46;
      v33 = src;
    }
    v34 = v32 - v33;
    v35 = sub_16E4080(a1);
    v36 = (void *)sub_145CBF0(*(__int64 **)(v35 + 8), v34, 4);
    v37 = src;
    *(_QWORD *)(a2 + 48) = v36;
    memcpy(v36, v37, v34);
    return j_j___libc_free_0(v37, (char *)v47 - (_BYTE *)v37);
  }
  return result;
}
