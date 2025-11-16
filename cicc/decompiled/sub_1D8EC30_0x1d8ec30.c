// Function: sub_1D8EC30
// Address: 0x1d8ec30
//
__int64 __fastcall sub_1D8EC30(__int64 a1, void *a2, size_t a3)
{
  __int64 v3; // r12
  int v5; // eax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 result; // rax
  __int64 v9; // rdx
  _QWORD *v10; // r13
  void *v11; // rcx
  __int64 v12; // r14
  char v13; // al
  _QWORD *v14; // rdx
  int v15; // eax
  __int64 *v16; // r13
  unsigned __int64 *v17; // rdi
  __int64 v18; // rdx
  void *v19; // r14
  size_t v20; // r15
  void *v21; // rsi
  __int64 v22; // r8
  _QWORD *v23; // rcx
  __int64 v24; // rax
  unsigned int v25; // eax
  __int64 v26; // rcx
  __m128i **v27; // rdx
  int v28; // eax
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 *v31; // r12
  __int64 v32; // rax
  __int64 (__fastcall *v33)(_QWORD *); // rdx
  __int64 *v34; // rdi
  __int64 v35; // rax
  unsigned int v36; // r8d
  _QWORD *v37; // rcx
  _QWORD *v38; // r13
  _BYTE *v39; // rdi
  __int64 *v40; // rdx
  size_t v41; // rdx
  __int64 v42; // rax
  __int64 v43; // rax
  _BYTE *v44; // rax
  __m128i *v45; // rax
  _QWORD *v46; // [rsp+0h] [rbp-C0h]
  unsigned int v47; // [rsp+8h] [rbp-B8h]
  _QWORD *v48; // [rsp+8h] [rbp-B8h]
  _QWORD *v49; // [rsp+10h] [rbp-B0h]
  unsigned int v50; // [rsp+10h] [rbp-B0h]
  void *v51; // [rsp+18h] [rbp-A8h]
  __int64 v52; // [rsp+18h] [rbp-A8h]
  __int64 v53; // [rsp+18h] [rbp-A8h]
  unsigned int v54; // [rsp+18h] [rbp-A8h]
  __int64 v55; // [rsp+18h] [rbp-A8h]
  void *s1; // [rsp+20h] [rbp-A0h] BYREF
  size_t n; // [rsp+28h] [rbp-98h]
  _QWORD v58[2]; // [rsp+30h] [rbp-90h] BYREF
  __int16 v59; // [rsp+40h] [rbp-80h]
  __m128i *v60; // [rsp+50h] [rbp-70h] BYREF
  _QWORD *v61; // [rsp+58h] [rbp-68h]
  __m128i v62; // [rsp+60h] [rbp-60h] BYREF
  unsigned __int64 *v63; // [rsp+70h] [rbp-50h] BYREF
  size_t v64; // [rsp+78h] [rbp-48h]
  _QWORD src[8]; // [rsp+80h] [rbp-40h] BYREF

  v3 = a1 + 184;
  s1 = a2;
  n = a3;
  v5 = sub_16D1B30((__int64 *)(a1 + 184), (unsigned __int8 *)a2, a3);
  if ( v5 != -1 )
  {
    v6 = *(_QWORD *)(a1 + 184);
    v7 = v6 + 8LL * v5;
    if ( v7 != v6 + 8LL * *(unsigned int *)(a1 + 192) )
      return *(_QWORD *)(*(_QWORD *)v7 + 8LL);
  }
  v10 = (_QWORD *)sub_1D91820();
  if ( !v10 )
  {
LABEL_9:
    if ( sub_1D91820() )
    {
      v59 = 261;
      v58[0] = &s1;
      v63 = src;
      sub_1D8E100((__int64 *)&v63, "unsupported GC: ", (__int64)"");
      v13 = v59;
      if ( (_BYTE)v59 )
      {
        if ( (_BYTE)v59 == 1 )
        {
          v60 = (__m128i *)&v63;
          v62.m128i_i16[0] = 260;
        }
        else
        {
          v14 = (_QWORD *)v58[0];
          if ( HIBYTE(v59) != 1 )
          {
            v14 = v58;
            v13 = 2;
          }
          v60 = (__m128i *)&v63;
          v61 = v14;
          v62.m128i_i8[0] = 4;
          v62.m128i_i8[1] = v13;
        }
      }
      else
      {
        v62.m128i_i16[0] = 256;
      }
      sub_16BCFB0((__int64)&v60, 1u);
    }
    v58[0] = "unsupported GC: ";
    v58[1] = &s1;
    v59 = 1283;
    sub_16E2FC0((__int64 *)&v63, (__int64)v58);
    if ( 0x3FFFFFFFFFFFFFFFLL - v64 > 0x3E )
    {
      v45 = (__m128i *)sub_2241490(&v63, " (did you remember to link and initialize the CodeGen library?)", 63);
      v60 = &v62;
      if ( (__m128i *)v45->m128i_i64[0] == &v45[1] )
      {
        v62 = _mm_loadu_si128(v45 + 1);
      }
      else
      {
        v60 = (__m128i *)v45->m128i_i64[0];
        v62.m128i_i64[0] = v45[1].m128i_i64[0];
      }
      v61 = (_QWORD *)v45->m128i_i64[1];
      v45->m128i_i64[0] = (__int64)v45[1].m128i_i64;
      v45->m128i_i64[1] = 0;
      v45[1].m128i_i8[0] = 0;
      if ( v63 != src )
        j_j___libc_free_0(v63, src[0] + 1LL);
      sub_16BD160((__int64)&v60, 1u);
    }
    sub_4262D8((__int64)"basic_string::append");
  }
  v11 = s1;
  while ( 1 )
  {
    v12 = v10[1];
    if ( *(_QWORD *)(v12 + 8) == n )
    {
      if ( !n )
        break;
      a2 = *(void **)v12;
      v51 = v11;
      v15 = memcmp(v11, *(const void **)v12, n);
      v11 = v51;
      if ( !v15 )
        break;
    }
    v10 = (_QWORD *)*v10;
    if ( !v10 )
      goto LABEL_9;
  }
  (*(void (__fastcall **)(__m128i **, void *, __int64, void *))(v12 + 32))(&v60, a2, v9, v11);
  if ( !s1 )
  {
    LOBYTE(src[0]) = 0;
    v16 = (__int64 *)v60;
    v41 = 0;
    v63 = src;
    v64 = 0;
LABEL_44:
    v42 = v16[1];
    v16[2] = v41;
    *(_BYTE *)(v42 + v41) = 0;
    v17 = v63;
    goto LABEL_23;
  }
  v63 = src;
  sub_1D8E100((__int64 *)&v63, s1, (__int64)s1 + n);
  v16 = (__int64 *)v60;
  v17 = (unsigned __int64 *)v60->m128i_i64[1];
  if ( v63 == src )
  {
    v41 = v64;
    if ( v64 )
    {
      if ( v64 == 1 )
        *(_BYTE *)v17 = src[0];
      else
        memcpy(v17, src, v64);
      v41 = v64;
    }
    goto LABEL_44;
  }
  if ( v17 == &v60[1].m128i_u64[1] )
  {
    v60->m128i_i64[1] = (__int64)v63;
    v16[2] = v64;
    v16[3] = src[0];
  }
  else
  {
    v60->m128i_i64[1] = (__int64)v63;
    v18 = v16[3];
    v16[2] = v64;
    v16[3] = src[0];
    if ( v17 )
    {
      v63 = v17;
      src[0] = v18;
      goto LABEL_23;
    }
  }
  v63 = src;
  v17 = src;
LABEL_23:
  v64 = 0;
  *(_BYTE *)v17 = 0;
  if ( v63 != src )
    j_j___libc_free_0(v63, src[0] + 1LL);
  v19 = s1;
  v20 = n;
  v21 = s1;
  v22 = (unsigned int)sub_16D19C0(v3, (unsigned __int8 *)s1, n);
  v23 = (_QWORD *)(*(_QWORD *)(a1 + 184) + 8 * v22);
  v24 = *v23;
  if ( !*v23 )
  {
LABEL_37:
    v46 = v23;
    v47 = v22;
    v35 = malloc(v20 + 17);
    v36 = v47;
    v37 = v46;
    v38 = (_QWORD *)v35;
    if ( !v35 )
    {
      if ( v20 == -17 )
      {
        v43 = malloc(1u);
        v36 = v47;
        v37 = v46;
        if ( v43 )
        {
          v39 = (_BYTE *)(v43 + 16);
          v38 = (_QWORD *)v43;
          goto LABEL_47;
        }
      }
      v48 = v37;
      v50 = v36;
      sub_16BD1C0("Allocation failed", 1u);
      v36 = v50;
      v37 = v48;
    }
    v39 = v38 + 2;
    if ( v20 + 1 <= 1 )
    {
LABEL_39:
      v39[v20] = 0;
      v21 = (void *)v36;
      *v38 = v20;
      v38[1] = 0;
      *v37 = v38;
      ++*(_DWORD *)(a1 + 196);
      v40 = (__int64 *)(*(_QWORD *)(a1 + 184) + 8LL * (unsigned int)sub_16D1CD0(v3, v36));
      v24 = *v40;
      if ( *v40 )
        goto LABEL_41;
      do
      {
        do
        {
          v24 = v40[1];
          ++v40;
        }
        while ( !v24 );
LABEL_41:
        ;
      }
      while ( v24 == -8 );
      goto LABEL_27;
    }
LABEL_47:
    v49 = v37;
    v54 = v36;
    v44 = memcpy(v39, v19, v20);
    v37 = v49;
    v36 = v54;
    v39 = v44;
    goto LABEL_39;
  }
  if ( v24 == -8 )
  {
    --*(_DWORD *)(a1 + 200);
    goto LABEL_37;
  }
LABEL_27:
  *(_QWORD *)(v24 + 8) = v60;
  v25 = *(_DWORD *)(a1 + 168);
  if ( v25 >= *(_DWORD *)(a1 + 172) )
  {
    v21 = 0;
    sub_1D8EA70(a1 + 160, 0);
    v25 = *(_DWORD *)(a1 + 168);
  }
  v26 = *(_QWORD *)(a1 + 160);
  v27 = (__m128i **)(v26 + 8LL * v25);
  if ( v27 )
  {
    *v27 = v60;
    v28 = *(_DWORD *)(a1 + 168);
    v29 = *(_QWORD *)(a1 + 160);
    v60 = 0;
    v30 = (unsigned int)(v28 + 1);
    *(_DWORD *)(a1 + 168) = v30;
    return *(_QWORD *)(v29 + 8 * v30 - 8);
  }
  else
  {
    v31 = (__int64 *)v60;
    v32 = v25 + 1;
    *(_DWORD *)(a1 + 168) = v32;
    result = *(_QWORD *)(v26 + 8 * v32 - 8);
    if ( v31 )
    {
      v33 = *(__int64 (__fastcall **)(_QWORD *))(*v31 + 8);
      if ( v33 == sub_1D59FF0 )
      {
        v34 = (__int64 *)v31[1];
        *v31 = (__int64)&unk_49F9CF0;
        if ( v34 != v31 + 3 )
        {
          v52 = result;
          j_j___libc_free_0(v34, v31[3] + 1);
          result = v52;
        }
        v53 = result;
        j_j___libc_free_0(v31, 56);
        return v53;
      }
      else
      {
        v55 = result;
        ((void (__fastcall *)(__int64 *, void *, __int64 (__fastcall *)(_QWORD *), __int64, __int64))v33)(
          v31,
          v21,
          v33,
          v26,
          v22);
        return v55;
      }
    }
  }
  return result;
}
