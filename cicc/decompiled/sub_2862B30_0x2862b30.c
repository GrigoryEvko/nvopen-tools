// Function: sub_2862B30
// Address: 0x2862b30
//
__int64 __fastcall sub_2862B30(__int64 a1, __int64 a2, unsigned int a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rax
  char *v11; // rdx
  void *v12; // rdi
  size_t v13; // rsi
  unsigned __int64 v14; // rax
  __int64 v15; // r8
  __int64 v17; // r8
  unsigned __int64 v18; // r9
  unsigned int v19; // esi
  int v20; // eax
  __int64 v21; // rdi
  int v22; // eax
  __int64 v23; // r9
  __int64 v24; // rdx
  __int64 v25; // rcx
  bool v26; // zf
  __int64 v27; // r8
  __int64 v28; // rdx
  __int64 v29; // r9
  int v30; // esi
  __int64 v31; // rcx
  unsigned __int64 v32; // rbx
  __int64 v33; // rcx
  __m128i *v34; // r15
  __m128i v35; // xmm0
  char v36; // al
  __m128i v37; // xmm1
  __int64 *v38; // r14
  __int64 *v39; // rbx
  char v40; // r10
  __int64 v41; // rsi
  _QWORD *v42; // rax
  __int64 v43; // rdi
  __int64 v44; // rsi
  _QWORD *v45; // rax
  __int64 v46; // rsi
  __int64 *v47; // rax
  __int64 *v48; // r14
  __int64 *i; // r13
  __int64 v50; // rsi
  unsigned __int64 v51; // rcx
  __int64 v52; // rdi
  unsigned __int64 v53; // rbx
  __int64 v54; // [rsp+0h] [rbp-C0h]
  __int64 v57; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v58; // [rsp+28h] [rbp-98h] BYREF
  void *base; // [rsp+30h] [rbp-90h] BYREF
  __int64 v60; // [rsp+38h] [rbp-88h]
  _BYTE v61[48]; // [rsp+40h] [rbp-80h] BYREF
  __int64 v62; // [rsp+70h] [rbp-50h]

  v8 = *(unsigned int *)(a2 + 768);
  if ( (_DWORD)v8 && *(_BYTE *)(a2 + 745) )
    return 0;
  v9 = *(unsigned int *)(a4 + 48);
  base = v61;
  v60 = 0x400000000LL;
  if ( !(_DWORD)v9 )
  {
    v10 = *(_QWORD *)(a4 + 88);
    v11 = v61;
    if ( !v10 )
      goto LABEL_8;
    goto LABEL_5;
  }
  sub_2850210((__int64)&base, a4 + 40, v9, v8, a5, a6);
  v10 = *(_QWORD *)(a4 + 88);
  if ( v10 )
  {
    v18 = (unsigned int)v60 + 1LL;
    if ( v18 > HIDWORD(v60) )
    {
      v54 = *(_QWORD *)(a4 + 88);
      sub_C8D5F0((__int64)&base, v61, (unsigned int)v60 + 1LL, 8u, v17, v18);
      v10 = v54;
    }
    v11 = (char *)base + 8 * (unsigned int)v60;
LABEL_5:
    *(_QWORD *)v11 = v10;
    v12 = base;
    LODWORD(v60) = v60 + 1;
    v13 = (unsigned int)v60;
    v14 = 8LL * (unsigned int)v60;
    goto LABEL_6;
  }
  v13 = (unsigned int)v60;
  v12 = base;
  v14 = 8LL * (unsigned int)v60;
LABEL_6:
  if ( v14 > 8 )
    qsort(v12, v13, 8u, (__compar_fn_t)sub_284F380);
LABEL_8:
  if ( (unsigned __int8)sub_28626B0(a2, (__int64)&base, &v57) )
  {
    if ( base != v61 )
      _libc_free((unsigned __int64)base);
    return 0;
  }
  v19 = *(_DWORD *)(a2 + 24);
  v20 = *(_DWORD *)(a2 + 16);
  v21 = v57;
  ++*(_QWORD *)a2;
  v22 = v20 + 1;
  v23 = 2 * v19;
  v58 = v21;
  if ( 4 * v22 >= 3 * v19 )
  {
    v19 *= 2;
    goto LABEL_56;
  }
  v24 = v19 - *(_DWORD *)(a2 + 20) - v22;
  v25 = v19 >> 3;
  if ( (unsigned int)v24 <= (unsigned int)v25 )
  {
LABEL_56:
    sub_2862850(a2, v19);
    sub_28626B0(a2, (__int64)&base, &v58);
    v21 = v58;
    v22 = *(_DWORD *)(a2 + 16) + 1;
  }
  *(_DWORD *)(a2 + 16) = v22;
  v26 = *(_DWORD *)(v21 + 8) == 1;
  v62 = -1;
  if ( !v26 || (v24 = -1, **(_QWORD **)v21 != -1) )
    --*(_DWORD *)(a2 + 20);
  sub_2850210(v21, (__int64)&base, v24, v25, v15, v23);
  v28 = *(unsigned int *)(a2 + 768);
  v29 = v28 + 1;
  v30 = *(_DWORD *)(a2 + 768);
  if ( v28 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 772) )
  {
    v51 = *(_QWORD *)(a2 + 760);
    v52 = a2 + 760;
    if ( v51 > a4 || a4 >= v51 + 112 * v28 )
    {
      v32 = a4;
      sub_2850FC0(v52, v28 + 1, v28, v51, v27, v29);
      v28 = *(unsigned int *)(a2 + 768);
      v31 = *(_QWORD *)(a2 + 760);
      v30 = *(_DWORD *)(a2 + 768);
    }
    else
    {
      v53 = a4 - v51;
      sub_2850FC0(v52, v28 + 1, v28, v51, v27, v29);
      v31 = *(_QWORD *)(a2 + 760);
      v28 = *(unsigned int *)(a2 + 768);
      v32 = v31 + v53;
      v30 = *(_DWORD *)(a2 + 768);
    }
  }
  else
  {
    v31 = *(_QWORD *)(a2 + 760);
    v32 = a4;
  }
  v26 = 112 * v28 + v31 == 0;
  v33 = 112 * v28 + v31;
  v34 = (__m128i *)v33;
  if ( !v26 )
  {
    v35 = _mm_loadu_si128((const __m128i *)(v32 + 8));
    *(_QWORD *)v33 = *(_QWORD *)v32;
    v36 = *(_BYTE *)(v32 + 24);
    *(__m128i *)(v33 + 8) = v35;
    *(_BYTE *)(v33 + 24) = v36;
    *(_QWORD *)(v33 + 32) = *(_QWORD *)(v32 + 32);
    *(_QWORD *)(v33 + 40) = v33 + 56;
    *(_QWORD *)(v33 + 48) = 0x400000000LL;
    if ( *(_DWORD *)(v32 + 48) )
      sub_2850210(v33 + 40, v32 + 40, v28, v33, v27, v29);
    v37 = _mm_loadu_si128((const __m128i *)(v32 + 96));
    v34[5].m128i_i64[1] = *(_QWORD *)(v32 + 88);
    v34[6] = v37;
    v30 = *(_DWORD *)(a2 + 768);
  }
  *(_DWORD *)(a2 + 768) = v30 + 1;
  v38 = *(__int64 **)(a4 + 40);
  v39 = &v38[*(unsigned int *)(a4 + 48)];
  if ( v38 != v39 )
  {
    v40 = *(_BYTE *)(a2 + 2148);
    do
    {
      v41 = *v38;
      if ( !v40 )
        goto LABEL_49;
      v42 = *(_QWORD **)(a2 + 2128);
      v43 = *(unsigned int *)(a2 + 2140);
      v28 = (__int64)&v42[v43];
      if ( v42 != (_QWORD *)v28 )
      {
        while ( v41 != *v42 )
        {
          if ( (_QWORD *)v28 == ++v42 )
            goto LABEL_50;
        }
        goto LABEL_34;
      }
LABEL_50:
      if ( (unsigned int)v43 < *(_DWORD *)(a2 + 2136) )
      {
        *(_DWORD *)(a2 + 2140) = v43 + 1;
        *(_QWORD *)v28 = v41;
        v40 = *(_BYTE *)(a2 + 2148);
        ++*(_QWORD *)(a2 + 2120);
      }
      else
      {
LABEL_49:
        sub_C8CC70(a2 + 2120, v41, v28, v33, v27, v29);
        v40 = *(_BYTE *)(a2 + 2148);
      }
LABEL_34:
      ++v38;
    }
    while ( v39 != v38 );
  }
  v44 = *(_QWORD *)(a4 + 88);
  if ( v44 )
  {
    if ( !*(_BYTE *)(a2 + 2148) )
      goto LABEL_53;
    v45 = *(_QWORD **)(a2 + 2128);
    v33 = *(unsigned int *)(a2 + 2140);
    v28 = (__int64)&v45[v33];
    if ( v45 != (_QWORD *)v28 )
    {
      while ( v44 != *v45 )
      {
        if ( (_QWORD *)v28 == ++v45 )
          goto LABEL_52;
      }
      goto LABEL_41;
    }
LABEL_52:
    if ( (unsigned int)v33 < *(_DWORD *)(a2 + 2136) )
    {
      *(_DWORD *)(a2 + 2140) = v33 + 1;
      *(_QWORD *)v28 = v44;
      ++*(_QWORD *)(a2 + 2120);
    }
    else
    {
LABEL_53:
      sub_C8CC70(a2 + 2120, v44, v28, v33, v27, v29);
    }
  }
LABEL_41:
  if ( base != v61 )
    _libc_free((unsigned __int64)base);
  v46 = *(_QWORD *)(a4 + 88);
  if ( v46 )
    sub_285AF50(a1 + 36280, v46, a3);
  v47 = *(__int64 **)(a4 + 40);
  v48 = &v47[*(unsigned int *)(a4 + 48)];
  for ( i = v47; v48 != i; ++i )
  {
    v50 = *i;
    sub_285AF50(a1 + 36280, v50, a3);
  }
  return 1;
}
