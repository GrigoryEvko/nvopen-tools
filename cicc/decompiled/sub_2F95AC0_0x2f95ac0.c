// Function: sub_2F95AC0
// Address: 0x2f95ac0
//
void __fastcall sub_2F95AC0(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  __int64 v5; // rsi
  __int64 v6; // rsi
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // rcx
  __int64 v10; // r13
  void *v11; // rax
  __int64 v12; // r13
  unsigned __int64 v13; // rbx
  __int64 v14; // rdi
  _QWORD *v15; // rax
  _QWORD *i; // rsi
  __int64 v17; // rdx
  int v18; // eax
  unsigned __int64 v19; // rax
  _BOOL4 v20; // eax
  unsigned __int64 v21; // rax
  const __m128i *v22; // rsi
  const __m128i *v23; // rdi
  __int64 v24; // rsi
  _QWORD *v25; // rdx
  unsigned int *v26; // rax
  const __m128i *v27; // rsi
  unsigned __int64 v28; // rax
  __m128i *v29; // rdi
  __int64 v30; // r14
  __int64 v31; // r10
  unsigned __int64 v32; // rdx
  _DWORD *v33; // rsi
  _BYTE *v34; // rax
  int v35; // edx
  unsigned int v36; // r10d
  __int64 v37; // rsi
  int v38; // eax
  unsigned __int64 v39; // rax
  _BOOL4 v40; // eax
  unsigned __int64 v41; // rax
  unsigned __int64 v42; // [rsp-148h] [rbp-148h] BYREF
  unsigned __int64 v43; // [rsp-140h] [rbp-140h] BYREF
  __m128i *v44; // [rsp-138h] [rbp-138h] BYREF
  __m128i *v45; // [rsp-130h] [rbp-130h]
  __m128i *v46; // [rsp-128h] [rbp-128h]
  __int64 v47; // [rsp-118h] [rbp-118h] BYREF
  unsigned __int64 v48[2]; // [rsp-110h] [rbp-110h] BYREF
  _DWORD v49[10]; // [rsp-100h] [rbp-100h] BYREF
  unsigned __int64 v50; // [rsp-D8h] [rbp-D8h] BYREF
  const __m128i *v51; // [rsp-D0h] [rbp-D0h]
  const __m128i *v52; // [rsp-C8h] [rbp-C8h]
  _BYTE *v53; // [rsp-C0h] [rbp-C0h]
  __int64 v54; // [rsp-B8h] [rbp-B8h]
  _BYTE v55[96]; // [rsp-B0h] [rbp-B0h] BYREF
  unsigned __int64 v56; // [rsp-50h] [rbp-50h]
  int v57; // [rsp-48h] [rbp-48h]

  if ( !*(_BYTE *)a1 )
    BUG();
  v5 = *(_QWORD *)(a1 + 16);
  v48[0] = (unsigned __int64)v49;
  v6 = v5 - *(_QWORD *)(a1 + 8);
  v47 = a1;
  v48[1] = 0x800000000LL;
  v49[8] = 0;
  sub_3157150(v48, v6 >> 3);
  v9 = a1;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = v55;
  v54 = 0x800000000LL;
  v56 = 0;
  v57 = 0;
  v10 = (__int64)(*(_QWORD *)(v9 + 16) - *(_QWORD *)(v9 + 8)) >> 3;
  if ( (_DWORD)v10 )
  {
    v11 = _libc_calloc((unsigned int)v10, 1u);
    v9 = a1;
    if ( !v11 )
      sub_C64F00("Allocation failed", 1u);
    v56 = (unsigned __int64)v11;
    v57 = v10;
  }
  v12 = 0x800000000000C09LL;
  v13 = a2 + (a3 << 8);
  if ( v13 == a2 )
    goto LABEL_53;
  while ( 2 )
  {
    v14 = *(unsigned int *)(a2 + 200);
    if ( *(_DWORD *)(*(_QWORD *)(v9 + 8) + 8 * v14 + 4) != -1 )
      goto LABEL_40;
    v15 = *(_QWORD **)(a2 + 120);
    for ( i = &v15[2 * *(unsigned int *)(a2 + 128)]; i != v15; v15 += 2 )
    {
      if ( (*v15 & 6) == 0 && *(_DWORD *)((*v15 & 0xFFFFFFFFFFFFFFF8LL) + 200) != -1 )
        goto LABEL_40;
    }
    v44 = 0;
    v17 = *(_QWORD *)a2;
    v45 = 0;
    v46 = 0;
    v18 = *(unsigned __int16 *)(v17 + 68);
    v20 = (_WORD)v18
       && ((v19 = (unsigned int)(v18 - 9), (unsigned __int16)v19 > 0x3Bu) || !_bittest64(&v12, v19))
       && (*(_QWORD *)(*(_QWORD *)(v17 + 16) + 24LL) & 0x10LL) == 0;
    *(_DWORD *)(*(_QWORD *)(v9 + 8) + 8 * v14) = v20;
    v21 = *(_QWORD *)(a2 + 40);
    v42 = a2;
    v22 = v45;
    v43 = v21;
    if ( v45 != v46 )
    {
      if ( v45 )
      {
        v45->m128i_i64[0] = a2;
        v22->m128i_i64[1] = v43;
        v22 = v45;
      }
      goto LABEL_18;
    }
LABEL_51:
    sub_2F95930((unsigned __int64 *)&v44, v22, &v42, &v43);
    v23 = v45;
    do
    {
      while ( 1 )
      {
        v24 = v23[-1].m128i_i64[0];
        v25 = (_QWORD *)v23[-1].m128i_i64[1];
        if ( v25 == (_QWORD *)(*(_QWORD *)(v24 + 40) + 16LL * *(unsigned int *)(v24 + 48)) )
          break;
        v23[-1].m128i_i64[1] = (__int64)(v25 + 2);
        if ( (*v25 & 6) != 0 || (v26 = (unsigned int *)(*v25 & 0xFFFFFFFFFFFFFFF8LL), v9 = v26[50], (_DWORD)v9 == -1) )
        {
          v23 = v45;
        }
        else
        {
          v9 = *(_QWORD *)(v47 + 8) + 8 * v9;
          if ( *(_DWORD *)(v9 + 4) == -1 )
          {
            v37 = *(_QWORD *)v26;
            v38 = *(unsigned __int16 *)(*(_QWORD *)v26 + 68LL);
            v40 = (_WORD)v38
               && ((v39 = (unsigned int)(v38 - 9), (unsigned __int16)v39 > 0x3Bu) || !_bittest64(&v12, v39))
               && (*(_QWORD *)(*(_QWORD *)(v37 + 16) + 24LL) & 0x10LL) == 0;
            *(_DWORD *)v9 = v40;
            v22 = v45;
            v41 = *v25 & 0xFFFFFFFFFFFFFFF8LL;
            v42 = v41;
            v43 = *(_QWORD *)(v41 + 40);
            if ( v45 != v46 )
            {
              if ( v45 )
              {
                v45->m128i_i64[0] = v41;
                v22->m128i_i64[1] = v43;
                v22 = v45;
              }
LABEL_18:
              v23 = v22 + 1;
              v45 = (__m128i *)&v22[1];
              continue;
            }
            goto LABEL_51;
          }
          v23 = v45;
          v27 = v51;
          v42 = v45[-1].m128i_u64[0];
          v28 = *v25 & 0xFFFFFFFFFFFFFFF8LL;
          v43 = v28;
          if ( v51 == v52 )
          {
            sub_2F95070(&v50, v51, &v43, &v42);
            v23 = v45;
          }
          else
          {
            if ( v51 )
            {
              v51->m128i_i64[0] = v28;
              v27->m128i_i64[1] = v42;
              v27 = v51;
              v23 = v45;
            }
            v51 = v27 + 1;
          }
        }
      }
      v29 = (__m128i *)&v23[-1];
      v45 = v29;
      if ( v29 == v44 )
      {
        sub_2F92C50((__int64)&v47, (_DWORD *)v24, (__int64)v25, v9, v7, v8);
      }
      else
      {
        v30 = v29[-1].m128i_i64[1];
        sub_2F92C50((__int64)&v47, (_DWORD *)v24, (__int64)v25, v9, v7, v8);
        v31 = v45[-1].m128i_i64[0];
        *(_DWORD *)(*(_QWORD *)(v47 + 8) + 8LL * *(unsigned int *)(v31 + 200)) += *(_DWORD *)(*(_QWORD *)(v47 + 8)
                                                                                            + 8LL
                                                                                            * *(unsigned int *)((*(_QWORD *)(v30 - 16) & 0xFFFFFFFFFFFFFFF8LL) + 200));
        v32 = *(_QWORD *)(v30 - 16) & 0xFFFFFFFFFFFFFFF8LL;
        v9 = *(unsigned int *)(v32 + 200);
        v33 = (_DWORD *)(*(_QWORD *)(v47 + 8) + 8 * v9);
        v8 = v9;
        if ( (_DWORD)v9 == v33[1] )
        {
          v34 = *(_BYTE **)(v32 + 120);
          v9 = (__int64)&v34[16 * *(unsigned int *)(v32 + 128)];
          if ( v34 == (_BYTE *)v9 )
          {
LABEL_35:
            if ( *v33 <= *(_DWORD *)(v47 + 4) )
            {
              v36 = *(_DWORD *)(v31 + 200);
              v33[1] = v36;
              sub_31571F0(v48, v36, (unsigned int)v8);
            }
          }
          else
          {
            v35 = 0;
            while ( (*v34 & 6) != 0 || (unsigned int)++v35 <= 3 )
            {
              v34 += 16;
              if ( (_BYTE *)v9 == v34 )
                goto LABEL_35;
            }
          }
        }
      }
      v23 = v45;
    }
    while ( v45 != v44 );
    if ( v45 )
      j_j___libc_free_0((unsigned __int64)v45);
LABEL_40:
    a2 += 256LL;
    if ( v13 != a2 )
    {
      v9 = v47;
      continue;
    }
    break;
  }
LABEL_53:
  sub_2F954D0(&v47);
  if ( v56 )
    _libc_free(v56);
  if ( v53 != v55 )
    _libc_free((unsigned __int64)v53);
  if ( v50 )
    j_j___libc_free_0(v50);
  if ( (_DWORD *)v48[0] != v49 )
    _libc_free(v48[0]);
}
