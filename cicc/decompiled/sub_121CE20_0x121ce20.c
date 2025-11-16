// Function: sub_121CE20
// Address: 0x121ce20
//
__int64 __fastcall sub_121CE20(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v4; // rbx
  const void *v5; // r13
  size_t v6; // r15
  __int64 v7; // r14
  int v8; // eax
  int v9; // eax
  __int64 v10; // rdx
  _QWORD *v11; // rax
  __int64 v12; // r13
  __int64 v14; // rax
  __int64 v15; // r14
  size_t v16; // r13
  __int64 v17; // r15
  const void *v18; // r12
  size_t v19; // rbx
  size_t v20; // rdx
  int v21; // eax
  __int64 v22; // rbx
  const void *v23; // r11
  size_t v24; // r15
  size_t v25; // rcx
  size_t v26; // rdx
  int v27; // eax
  __int64 v28; // rax
  _BYTE *v29; // rsi
  __int64 v30; // rdx
  __int64 v31; // r15
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rbx
  __int64 v35; // r15
  __int64 v36; // rdi
  __int64 v37; // rdi
  __int64 v38; // rdi
  size_t v39; // rbx
  size_t v40; // rcx
  size_t v41; // rdx
  int v42; // eax
  unsigned int v43; // edi
  __int64 v44; // rbx
  _QWORD *v45; // [rsp+10h] [rbp-50h]
  __int64 v46; // [rsp+18h] [rbp-48h]
  size_t v47; // [rsp+18h] [rbp-48h]
  __int64 v48; // [rsp+18h] [rbp-48h]
  size_t v49; // [rsp+18h] [rbp-48h]
  __int64 v50; // [rsp+20h] [rbp-40h]

  v3 = a1;
  v4 = a2;
  v5 = *(const void **)a2;
  v6 = *(_QWORD *)(a2 + 8);
  v7 = *(_QWORD *)(a1 + 344);
  v8 = sub_C92610();
  v9 = sub_C92860((__int64 *)(v7 + 128), v5, v6, v8);
  if ( v9 != -1 )
  {
    v10 = *(_QWORD *)(v7 + 128);
    v11 = (_QWORD *)(v10 + 8LL * v9);
    if ( v11 != (_QWORD *)(v10 + 8LL * *(unsigned int *)(v7 + 136)) )
      return *v11 + 8LL;
  }
  v14 = sub_BAA410(*(_QWORD *)(a1 + 344), *(void **)a2, *(_QWORD *)(a2 + 8));
  v12 = v14;
  v50 = a1 + 1240;
  if ( !*(_QWORD *)(a1 + 1248) )
  {
    v15 = a1 + 1240;
    goto LABEL_26;
  }
  v46 = v14;
  v15 = a1 + 1240;
  v16 = *(_QWORD *)(a2 + 8);
  v17 = *(_QWORD *)(a1 + 1248);
  v18 = *(const void **)a2;
  do
  {
    while ( 1 )
    {
      v19 = *(_QWORD *)(v17 + 40);
      v20 = v16;
      if ( v19 <= v16 )
        v20 = *(_QWORD *)(v17 + 40);
      if ( v20 )
      {
        v21 = memcmp(*(const void **)(v17 + 32), v18, v20);
        if ( v21 )
          break;
      }
      v22 = v19 - v16;
      if ( v22 >= 0x80000000LL )
        goto LABEL_16;
      if ( v22 > (__int64)0xFFFFFFFF7FFFFFFFLL )
      {
        v21 = v22;
        break;
      }
LABEL_7:
      v17 = *(_QWORD *)(v17 + 24);
      if ( !v17 )
        goto LABEL_17;
    }
    if ( v21 < 0 )
      goto LABEL_7;
LABEL_16:
    v15 = v17;
    v17 = *(_QWORD *)(v17 + 16);
  }
  while ( v17 );
LABEL_17:
  v23 = v18;
  v24 = v16;
  v3 = a1;
  v12 = v46;
  v4 = a2;
  if ( v50 == v15 )
    goto LABEL_26;
  v25 = *(_QWORD *)(v15 + 40);
  v26 = v24;
  if ( v25 <= v24 )
    v26 = *(_QWORD *)(v15 + 40);
  if ( v26 && (v47 = *(_QWORD *)(v15 + 40), v27 = memcmp(v23, *(const void **)(v15 + 32), v26), v25 = v47, v27) )
  {
LABEL_25:
    if ( v27 < 0 )
      goto LABEL_26;
  }
  else if ( (__int64)(v24 - v25) <= 0x7FFFFFFF )
  {
    if ( (__int64)(v24 - v25) >= (__int64)0xFFFFFFFF80000000LL )
    {
      v27 = v24 - v25;
      goto LABEL_25;
    }
LABEL_26:
    v45 = (_QWORD *)v15;
    v28 = sub_22077B0(72);
    v29 = *(_BYTE **)v4;
    v30 = *(_QWORD *)(v4 + 8);
    v31 = v28 + 32;
    v15 = v28;
    *(_QWORD *)(v28 + 32) = v28 + 48;
    v48 = v28 + 48;
    sub_12060D0((__int64 *)(v28 + 32), v29, (__int64)&v29[v30]);
    *(_QWORD *)(v15 + 64) = 0;
    v32 = sub_121CB90((_QWORD *)(v3 + 1232), v45, v31);
    v34 = v32;
    v35 = v33;
    if ( v33 )
    {
      if ( v50 == v33 || v32 )
      {
LABEL_29:
        v36 = 1;
        goto LABEL_30;
      }
      v39 = *(_QWORD *)(v15 + 40);
      v41 = *(_QWORD *)(v33 + 40);
      v40 = v41;
      if ( v39 <= v41 )
        v41 = *(_QWORD *)(v15 + 40);
      if ( v41
        && (v49 = v40,
            v42 = memcmp(*(const void **)(v15 + 32), *(const void **)(v35 + 32), v41),
            v40 = v49,
            (v43 = v42) != 0) )
      {
LABEL_43:
        v36 = v43 >> 31;
      }
      else
      {
        v44 = v39 - v40;
        v36 = 0;
        if ( v44 <= 0x7FFFFFFF )
        {
          if ( v44 < (__int64)0xFFFFFFFF80000000LL )
            goto LABEL_29;
          v43 = v44;
          goto LABEL_43;
        }
      }
LABEL_30:
      sub_220F040(v36, v15, v35, v50);
      ++*(_QWORD *)(v3 + 1272);
    }
    else
    {
      v37 = *(_QWORD *)(v15 + 32);
      if ( v48 != v37 )
        j_j___libc_free_0(v37, *(_QWORD *)(v15 + 48) + 1LL);
      v38 = v15;
      v15 = v34;
      j_j___libc_free_0(v38, 72);
    }
  }
  *(_QWORD *)(v15 + 64) = a3;
  return v12;
}
