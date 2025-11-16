// Function: sub_CAF680
// Address: 0xcaf680
//
_QWORD *__fastcall sub_CAF680(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  void *v5; // rax
  void *v6; // rdx
  char *v7; // rsi
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rcx
  char *v10; // rsi
  void *v11; // rax
  const void *v12; // r12
  void *v13; // r15
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rdx
  __int64 v16; // r14
  _QWORD *v17; // r13
  void *v18; // rbx
  size_t v19; // rdx
  int v20; // eax
  void *v21; // rcx
  size_t v22; // rdx
  int v23; // eax
  _QWORD *v24; // rdi
  _QWORD *result; // rax
  _QWORD *v26; // rax
  __int64 v27; // rax
  _QWORD *v28; // rdx
  _QWORD *v29; // r9
  _BOOL8 v30; // rdi
  size_t v31; // rcx
  size_t v32; // rdx
  unsigned int v33; // eax
  size_t v34; // [rsp+0h] [rbp-C0h]
  void *v35; // [rsp+8h] [rbp-B8h]
  _QWORD *v36; // [rsp+8h] [rbp-B8h]
  _QWORD *v37; // [rsp+8h] [rbp-B8h]
  char *v38; // [rsp+10h] [rbp-B0h]
  char *v39; // [rsp+18h] [rbp-A8h]
  _QWORD *v40; // [rsp+20h] [rbp-A0h]
  __int64 v42; // [rsp+28h] [rbp-98h]
  void *s2[2]; // [rsp+30h] [rbp-90h] BYREF
  char *v44; // [rsp+40h] [rbp-80h] BYREF
  char *v45; // [rsp+48h] [rbp-78h]
  char v46[8]; // [rsp+50h] [rbp-70h] BYREF
  __m128i v47; // [rsp+58h] [rbp-68h] BYREF
  _QWORD *v48; // [rsp+68h] [rbp-58h]
  _QWORD v49[9]; // [rsp+78h] [rbp-48h] BYREF

  sub_CAD680((__int64)v46, (unsigned __int64 **)a1, a3, a4, a5);
  *(__m128i *)s2 = _mm_loadu_si128(&v47);
  v5 = (void *)sub_C934D0(s2, (unsigned __int8 *)" \t", 2, 0);
  v6 = s2[1];
  v7 = 0;
  if ( v5 <= s2[1] )
  {
    v6 = v5;
    v7 = (char *)((char *)s2[1] - (char *)v5);
  }
  v45 = v7;
  v44 = (char *)s2[0] + (unsigned __int64)v6;
  v8 = sub_C935B0(&v44, (unsigned __int8 *)" \t", 2, 0);
  v9 = (unsigned __int64)v45;
  v10 = 0;
  if ( v8 < (unsigned __int64)v45 )
  {
    v10 = &v45[-v8];
    v9 = v8;
  }
  s2[1] = v10;
  s2[0] = &v44[v9];
  v11 = (void *)sub_C934D0(s2, (unsigned __int8 *)" \t", 2, 0);
  v12 = s2[0];
  if ( v11 > s2[1] )
    v11 = s2[1];
  v13 = v11;
  v44 = (char *)s2[0] + (unsigned __int64)v11;
  v45 = (char *)((char *)s2[1] - (char *)v11);
  v14 = sub_C935B0(&v44, (unsigned __int8 *)" \t", 2, 0);
  v15 = (unsigned __int64)v45;
  if ( v14 < (unsigned __int64)v45 )
  {
    v38 = &v45[-v14];
    v15 = v14;
  }
  else
  {
    v38 = 0;
  }
  v39 = &v44[v15];
  v16 = *(_QWORD *)(a1 + 128);
  v17 = (_QWORD *)(a1 + 120);
  v40 = (_QWORD *)(a1 + 120);
  if ( !v16 )
  {
    v17 = (_QWORD *)(a1 + 120);
    goto LABEL_32;
  }
  do
  {
    while ( 1 )
    {
      v18 = *(void **)(v16 + 40);
      v19 = (size_t)v13;
      if ( v18 <= v13 )
        v19 = *(_QWORD *)(v16 + 40);
      if ( v19 )
      {
        v20 = memcmp(*(const void **)(v16 + 32), v12, v19);
        if ( v20 )
          break;
      }
      if ( v18 != v13 && v18 < v13 )
      {
        v16 = *(_QWORD *)(v16 + 24);
        goto LABEL_19;
      }
LABEL_11:
      v17 = (_QWORD *)v16;
      v16 = *(_QWORD *)(v16 + 16);
      if ( !v16 )
        goto LABEL_20;
    }
    if ( v20 >= 0 )
      goto LABEL_11;
    v16 = *(_QWORD *)(v16 + 24);
LABEL_19:
    ;
  }
  while ( v16 );
LABEL_20:
  if ( v40 == v17 )
    goto LABEL_32;
  v21 = (void *)v17[5];
  v22 = (size_t)v13;
  if ( v21 <= v13 )
    v22 = v17[5];
  if ( v22 && (v35 = (void *)v17[5], v23 = memcmp(v12, (const void *)v17[4], v22), v21 = v35, v23) )
  {
    if ( v23 < 0 )
      goto LABEL_32;
  }
  else
  {
    if ( v21 == v13 || v21 <= v13 )
      goto LABEL_27;
LABEL_32:
    v36 = v17;
    v26 = (_QWORD *)sub_22077B0(64);
    v26[4] = v12;
    v17 = v26;
    v26[5] = v13;
    v26[6] = 0;
    v26[7] = 0;
    v27 = sub_CAF3B0((_QWORD *)(a1 + 112), v36, (__int64)(v26 + 4));
    v29 = v28;
    if ( v28 )
    {
      if ( v40 == v28 || v27 )
      {
        v30 = 1;
      }
      else
      {
        v31 = v28[5];
        v32 = (size_t)v13;
        if ( v31 <= (unsigned __int64)v13 )
          v32 = v31;
        if ( v32 && (v34 = v31, v37 = v29, v33 = memcmp(v12, (const void *)v29[4], v32), v29 = v37, v31 = v34, v33) )
        {
          v30 = v33 >> 31;
        }
        else
        {
          v30 = v31 > (unsigned __int64)v13;
          if ( (void *)v31 == v13 )
            v30 = 0;
        }
      }
      sub_220F040(v30, v17, v29, v40);
      ++*(_QWORD *)(a1 + 152);
    }
    else
    {
      v42 = v27;
      j_j___libc_free_0(v17, 64);
      v17 = (_QWORD *)v42;
    }
  }
LABEL_27:
  v24 = v48;
  v17[6] = v39;
  v17[7] = v38;
  result = v49;
  if ( v24 != v49 )
    return (_QWORD *)j_j___libc_free_0(v24, v49[0] + 1LL);
  return result;
}
