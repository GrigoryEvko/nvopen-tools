// Function: sub_3899950
// Address: 0x3899950
//
__int64 __fastcall sub_3899950(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  __int64 *v4; // r12
  __int64 v5; // rbx
  int v6; // eax
  __int64 v7; // rdx
  _QWORD *v8; // rax
  __int64 v9; // r13
  __int64 v11; // rax
  __int64 v12; // r14
  size_t v13; // r13
  unsigned __int8 *v14; // r12
  __int64 v15; // r15
  size_t v16; // rbx
  size_t v17; // rdx
  int v18; // eax
  __int64 v19; // rbx
  unsigned __int8 *v20; // r11
  size_t v21; // r9
  size_t v22; // rbx
  size_t v23; // rdx
  int v24; // eax
  __int64 v25; // r9
  _QWORD *v26; // rbx
  __int64 v27; // rax
  _BYTE *v28; // rsi
  __int64 v29; // rdx
  __int64 v30; // rax
  _QWORD *v31; // rdx
  __int64 v32; // rbx
  _QWORD *v33; // r15
  unsigned int v34; // edi
  unsigned __int64 v35; // rdi
  unsigned __int64 v36; // rdi
  size_t v37; // rbx
  size_t v38; // rcx
  size_t v39; // rdx
  int v40; // eax
  unsigned int v41; // edi
  __int64 v42; // rbx
  __int64 v43; // [rsp+10h] [rbp-50h]
  __int64 v44; // [rsp+18h] [rbp-48h]
  size_t v45; // [rsp+18h] [rbp-48h]
  __int64 v46; // [rsp+18h] [rbp-48h]
  size_t v47; // [rsp+18h] [rbp-48h]
  _QWORD *v48; // [rsp+20h] [rbp-40h]

  v3 = a2;
  v4 = a1;
  v5 = a1[22];
  v6 = sub_16D1B30((__int64 *)(v5 + 128), *(unsigned __int8 **)a2, *(_QWORD *)(a2 + 8));
  if ( v6 != -1 )
  {
    v7 = *(_QWORD *)(v5 + 128);
    v8 = (_QWORD *)(v7 + 8LL * v6);
    if ( v8 != (_QWORD *)(v7 + 8LL * *(unsigned int *)(v5 + 136)) )
      return *v8 + 8LL;
  }
  v11 = sub_1633B90(a1[22], *(void **)a2, *(_QWORD *)(a2 + 8));
  v9 = v11;
  v48 = a1 + 129;
  if ( !a1[130] )
  {
    v12 = (__int64)(a1 + 129);
    goto LABEL_26;
  }
  v44 = v11;
  v12 = (__int64)(a1 + 129);
  v13 = *(_QWORD *)(a2 + 8);
  v14 = *(unsigned __int8 **)a2;
  v15 = a1[130];
  do
  {
    while ( 1 )
    {
      v16 = *(_QWORD *)(v15 + 40);
      v17 = v13;
      if ( v16 <= v13 )
        v17 = *(_QWORD *)(v15 + 40);
      if ( v17 )
      {
        v18 = memcmp(*(const void **)(v15 + 32), v14, v17);
        if ( v18 )
          break;
      }
      v19 = v16 - v13;
      if ( v19 >= 0x80000000LL )
        goto LABEL_16;
      if ( v19 > (__int64)0xFFFFFFFF7FFFFFFFLL )
      {
        v18 = v19;
        break;
      }
LABEL_7:
      v15 = *(_QWORD *)(v15 + 24);
      if ( !v15 )
        goto LABEL_17;
    }
    if ( v18 < 0 )
      goto LABEL_7;
LABEL_16:
    v12 = v15;
    v15 = *(_QWORD *)(v15 + 16);
  }
  while ( v15 );
LABEL_17:
  v20 = v14;
  v21 = v13;
  v4 = a1;
  v9 = v44;
  v3 = a2;
  if ( v48 == (_QWORD *)v12 )
    goto LABEL_26;
  v22 = *(_QWORD *)(v12 + 40);
  v23 = v21;
  if ( v22 <= v21 )
    v23 = *(_QWORD *)(v12 + 40);
  if ( v23 && (v45 = v21, v24 = memcmp(v20, *(const void **)(v12 + 32), v23), v21 = v45, v24) )
  {
LABEL_25:
    if ( v24 < 0 )
      goto LABEL_26;
  }
  else
  {
    v25 = v21 - v22;
    if ( v25 <= 0x7FFFFFFF )
    {
      if ( v25 >= (__int64)0xFFFFFFFF80000000LL )
      {
        v24 = v25;
        goto LABEL_25;
      }
LABEL_26:
      v26 = (_QWORD *)v12;
      v27 = sub_22077B0(0x48u);
      v28 = *(_BYTE **)v3;
      v29 = *(_QWORD *)(v3 + 8);
      v12 = v27;
      *(_QWORD *)(v27 + 32) = v27 + 48;
      v43 = v27 + 48;
      v46 = v27 + 32;
      sub_3887850((__int64 *)(v27 + 32), v28, (__int64)&v28[v29]);
      *(_QWORD *)(v12 + 64) = 0;
      v30 = sub_38996C0(v4 + 128, v26, v46);
      v32 = v30;
      v33 = v31;
      if ( v31 )
      {
        if ( v48 == v31 || v30 )
        {
LABEL_29:
          LOBYTE(v34) = 1;
          goto LABEL_30;
        }
        v37 = *(_QWORD *)(v12 + 40);
        v39 = v31[5];
        v38 = v39;
        if ( v37 <= v39 )
          v39 = *(_QWORD *)(v12 + 40);
        if ( v39
          && (v47 = v38, v40 = memcmp(*(const void **)(v12 + 32), (const void *)v33[4], v39),
                         v38 = v47,
                         (v41 = v40) != 0) )
        {
LABEL_43:
          v34 = v41 >> 31;
        }
        else
        {
          v42 = v37 - v38;
          LOBYTE(v34) = 0;
          if ( v42 <= 0x7FFFFFFF )
          {
            if ( v42 < (__int64)0xFFFFFFFF80000000LL )
              goto LABEL_29;
            v41 = v42;
            goto LABEL_43;
          }
        }
LABEL_30:
        sub_220F040(v34, v12, v33, v48);
        ++v4[133];
      }
      else
      {
        v35 = *(_QWORD *)(v12 + 32);
        if ( v43 != v35 )
          j_j___libc_free_0(v35);
        v36 = v12;
        v12 = v32;
        j_j___libc_free_0(v36);
      }
    }
  }
  *(_QWORD *)(v12 + 64) = a3;
  return v9;
}
