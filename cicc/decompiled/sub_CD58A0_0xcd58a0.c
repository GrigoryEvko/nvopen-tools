// Function: sub_CD58A0
// Address: 0xcd58a0
//
void *__fastcall sub_CD58A0(__int64 a1, __int64 a2)
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
  void *result; // rax
  char v20; // al
  __int64 v21; // rax
  __m128i *v22; // rax
  void *v23; // r8
  __int64 v24; // rcx
  __m128i *v25; // r13
  int v26; // edx
  __m128i *v27; // rax
  unsigned int v28; // ecx
  const __m128i *v29; // rdx
  _BYTE *v30; // rdi
  _BYTE *v31; // r13
  _BYTE *v32; // rax
  size_t v33; // r13
  __int64 *v34; // r8
  __int64 v35; // rax
  char *v36; // rdi
  signed __int64 v37; // rsi
  unsigned int v38; // [rsp+0h] [rbp-70h]
  __int64 v39; // [rsp+0h] [rbp-70h]
  __int64 v40; // [rsp+8h] [rbp-68h]
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
    sub_CCC650(a1, (__int64 *)&src);
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
    sub_CCC650(a1, (__int64 *)&src);
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
    sub_CCC650(a1, (__int64 *)&src);
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
    sub_CCC650(a1, (__int64 *)&src);
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
    sub_CCC2C0(a1, (unsigned int *)&src);
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
    sub_CCC2C0(a1, (unsigned int *)&src);
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
    sub_CCC2C0(a1, (unsigned int *)&src);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v44);
  }
  else if ( v43 )
  {
    LODWORD(src) = 0;
  }
  result = (void *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  if ( !(_BYTE)result )
  {
    result = v46;
    v46->m128i_i32[0] = (int)src;
  }
  if ( *(_DWORD *)(a2 + 36) )
  {
    v20 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    src = 0;
    v46 = 0;
    v47 = 0;
    if ( !v20 )
    {
      if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 56LL))(a1) && (v31 = src, v46 == src) )
      {
        v32 = src;
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
          sub_CD5510(a1, (__int64 *)&src);
          (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v44);
        }
        v31 = v46;
        v32 = src;
      }
      v33 = v31 - v32;
      v34 = *(__int64 **)(sub_CB0A70(a1) + 8);
      v35 = *v34;
      v34[10] += v33;
      v36 = (char *)((v35 + 3) & 0xFFFFFFFFFFFFFFFCLL);
      if ( v34[1] >= (unsigned __int64)&v36[v33] && v35 )
        *v34 = (__int64)&v36[v33];
      else
        v36 = (char *)sub_9D1E70((__int64)v34, v33, v33, 2);
      *(_QWORD *)(a2 + 48) = v36;
      result = memcpy(v36, src, v33);
      goto LABEL_67;
    }
    v21 = *(unsigned int *)(a2 + 36);
    if ( !*(_DWORD *)(a2 + 36) )
    {
LABEL_63:
      result = (void *)(*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 56LL))(a1);
      if ( (_BYTE)result )
      {
        v30 = src;
        if ( v46 == src )
          goto LABEL_68;
      }
      result = (void *)(*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
                         a1,
                         "ZeroPData",
                         0,
                         0,
                         &v43,
                         &v44);
      if ( (_BYTE)result )
      {
        sub_CD5510(a1, (__int64 *)&src);
        result = (void *)(*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v44);
      }
LABEL_67:
      v30 = src;
LABEL_68:
      if ( v30 )
        return (void *)j_j___libc_free_0(v30, (char *)v47 - v30);
      return result;
    }
    v40 = 16 * v21;
    v22 = (__m128i *)sub_22077B0(16 * v21);
    v23 = src;
    v24 = v40;
    v25 = v22;
    if ( (char *)v46 - (_BYTE *)src > 0 )
    {
      v39 = v40;
      v41 = src;
      memmove(v22, src, (char *)v46 - (_BYTE *)src);
      v23 = v41;
      v24 = v39;
      v37 = (char *)v47 - (_BYTE *)v41;
    }
    else
    {
      if ( !src )
        goto LABEL_55;
      v37 = (char *)v47 - (_BYTE *)src;
    }
    v42 = v24;
    j_j___libc_free_0(v23, v37);
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
      while ( 1 )
      {
        v29 = (const __m128i *)(*(_QWORD *)(a2 + 48) + 16LL * v28);
        if ( v25 == v27 )
        {
          v38 = v28;
          sub_CD12E0((__int64)&src, v25, v29);
          v28 = v38 + 1;
          if ( *(_DWORD *)(a2 + 36) <= v38 + 1 )
            goto LABEL_63;
        }
        else
        {
          if ( v25 )
          {
            *v25 = _mm_loadu_si128(v29);
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
  return result;
}
