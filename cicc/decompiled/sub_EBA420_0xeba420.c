// Function: sub_EBA420
// Address: 0xeba420
//
__int64 __fastcall sub_EBA420(__int64 *a1)
{
  __int64 v2; // rdi
  __int64 v3; // rax
  __int64 v4; // r12
  unsigned int v5; // eax
  __int64 v6; // r8
  __int64 v7; // rdi
  unsigned int v8; // r15d
  __int64 v10; // rax
  bool v11; // zf
  __m128i *v12; // rbx
  __int64 v13; // r9
  __m128i *v14; // r12
  size_t v15; // r14
  __m128i si128; // xmm0
  __int64 v17; // rax
  __int64 v18; // r14
  _BOOL4 v19; // r11d
  __m128i *v20; // rax
  void *v21; // rcx
  __int64 v22; // rax
  _QWORD *v23; // rdx
  _QWORD *v24; // rbx
  size_t v25; // rcx
  size_t v26; // r8
  size_t v27; // rdx
  unsigned int v28; // eax
  _QWORD *v29; // rdi
  __int64 v30; // [rsp+0h] [rbp-A0h]
  size_t v31; // [rsp+0h] [rbp-A0h]
  unsigned __int64 v32; // [rsp+8h] [rbp-98h]
  _BOOL4 v33; // [rsp+8h] [rbp-98h]
  size_t v34; // [rsp+8h] [rbp-98h]
  void *s2[2]; // [rsp+10h] [rbp-90h] BYREF
  _QWORD *v36; // [rsp+20h] [rbp-80h]
  __int64 v37; // [rsp+28h] [rbp-78h]
  size_t n[2]; // [rsp+30h] [rbp-70h] BYREF
  const char *v39; // [rsp+40h] [rbp-60h] BYREF
  char v40; // [rsp+60h] [rbp-40h]
  char v41; // [rsp+61h] [rbp-3Fh]

  v2 = *a1;
  n[0] = 0;
  n[1] = 0;
  v3 = sub_ECD7B0(v2);
  v4 = sub_ECD6A0(v3);
  v5 = sub_EB61F0(*a1, (__int64 *)n);
  if ( (_BYTE)v5 )
  {
    v7 = *a1;
    v41 = 1;
    v39 = "expected identifier";
    v40 = 3;
    return (unsigned int)sub_ECDA70(v7, v4, &v39, 0, 0);
  }
  v8 = v5;
  v10 = *a1;
  v11 = *(_QWORD *)(*a1 + 856) == 0;
  v37 = *a1;
  if ( !v11 )
  {
    sub_EAA500(v10 + 816, (__int64)n);
    return v8;
  }
  v30 = *(_QWORD *)(v37 + 768);
  v12 = (__m128i *)v30;
  v13 = 16LL * *(unsigned int *)(v37 + 776);
  v32 = *(unsigned int *)(v37 + 776);
  v14 = (__m128i *)(v30 + v13);
  if ( v30 != v30 + v13 )
  {
    v15 = n[1];
    s2[0] = (void *)n[0];
    while ( v12->m128i_i64[1] != v15 || v15 && memcmp((const void *)v12->m128i_i64[0], s2[0], v15) )
    {
      if ( v14 == ++v12 )
      {
        if ( v32 <= 1 )
          goto LABEL_13;
        v36 = (_QWORD *)(v37 + 816);
        goto LABEL_19;
      }
    }
    if ( v14 != v12 )
      return v8;
    v36 = (_QWORD *)(v37 + 816);
    if ( v32 > 1 )
    {
LABEL_19:
      v18 = v30;
      s2[0] = (void *)(v37 + 824);
      do
      {
        v22 = sub_EAAF30(v36, (_QWORD *)s2[0], v18);
        v24 = v23;
        if ( v23 )
        {
          if ( v22 || v23 == s2[0] )
          {
            v19 = 1;
          }
          else
          {
            v25 = v23[5];
            v26 = *(_QWORD *)(v18 + 8);
            v27 = v26;
            if ( v25 <= v26 )
              v27 = v25;
            if ( v27
              && (v31 = *(_QWORD *)(v18 + 8),
                  v34 = v25,
                  v28 = memcmp(*(const void **)v18, (const void *)v24[4], v27),
                  v25 = v34,
                  v26 = v31,
                  v28) )
            {
              v19 = v28 >> 31;
            }
            else
            {
              v19 = v25 > v26;
              if ( v25 == v26 )
                v19 = 0;
            }
          }
          v33 = v19;
          v20 = (__m128i *)sub_22077B0(48);
          v21 = s2[0];
          v20[2] = _mm_loadu_si128((const __m128i *)v18);
          sub_220F040(v33, v20, v24, v21);
          ++*(_QWORD *)(v37 + 856);
        }
        v18 += 16;
      }
      while ( v14 != (__m128i *)v18 );
      goto LABEL_35;
    }
    goto LABEL_13;
  }
  if ( v32 <= 1 )
  {
LABEL_13:
    si128 = _mm_load_si128((const __m128i *)n);
    if ( v32 + 1 > *(unsigned int *)(v37 + 780) )
    {
      *(__m128i *)s2 = si128;
      sub_C8D5F0(v37 + 768, (const void *)(v37 + 784), v32 + 1, 0x10u, v6, v13);
      si128 = _mm_load_si128((const __m128i *)s2);
      v14 = (__m128i *)(*(_QWORD *)(v37 + 768) + 16LL * *(unsigned int *)(v37 + 776));
    }
    v17 = v37;
    *v14 = si128;
    ++*(_DWORD *)(v17 + 776);
    return v8;
  }
  v36 = (_QWORD *)(v37 + 816);
LABEL_35:
  v29 = v36;
  *(_DWORD *)(v37 + 776) = 0;
  sub_EAA500((__int64)v29, (__int64)n);
  return v8;
}
