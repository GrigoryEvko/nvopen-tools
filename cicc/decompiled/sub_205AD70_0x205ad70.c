// Function: sub_205AD70
// Address: 0x205ad70
//
__int64 __fastcall sub_205AD70(__int64 *a1, __int64 a2, __int64 a3, const __m128i *a4)
{
  __int64 v7; // rbx
  __int64 v8; // r12
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rdx
  __int64 v11; // r14
  bool v12; // cf
  unsigned __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r15
  __int64 v16; // rdx
  int v17; // edi
  int v18; // edi
  __m128i v19; // xmm2
  __int64 v20; // rdi
  __int64 v21; // rdi
  char v22; // si
  __int64 v23; // r15
  __int64 v24; // rdx
  unsigned int v25; // esi
  unsigned int v26; // esi
  unsigned int v27; // esi
  unsigned int v28; // edx
  __int64 v29; // rdx
  __m128i v30; // xmm0
  __int64 v31; // rdx
  unsigned int v32; // edx
  unsigned int v33; // edx
  __int64 i; // r14
  __int64 v35; // rdi
  __int64 v37; // r15
  __int64 v38; // rax
  const __m128i *v39; // [rsp+0h] [rbp-60h]
  __int64 v40; // [rsp+10h] [rbp-50h]
  __int64 v41; // [rsp+18h] [rbp-48h]
  __int64 v42; // [rsp+18h] [rbp-48h]
  __int64 v43; // [rsp+18h] [rbp-48h]
  __int64 v44; // [rsp+20h] [rbp-40h]
  __int64 v45; // [rsp+20h] [rbp-40h]
  __int64 v46; // [rsp+20h] [rbp-40h]
  __int64 v47; // [rsp+28h] [rbp-38h]

  v7 = a1[1];
  v8 = *a1;
  v9 = 0xCCCCCCCCCCCCCCCDLL * ((v7 - *a1) >> 4);
  if ( v9 == 0x199999999999999LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v10 = 1;
  v11 = a2;
  if ( v9 )
    v10 = 0xCCCCCCCCCCCCCCCDLL * ((v7 - v8) >> 4);
  v12 = __CFADD__(v10, v9);
  v13 = v10 - 0x3333333333333333LL * ((v7 - v8) >> 4);
  v14 = a2 - v8;
  if ( v12 )
  {
    v37 = 0x7FFFFFFFFFFFFFD0LL;
  }
  else
  {
    if ( !v13 )
    {
      v40 = 0;
      v15 = 80;
      v47 = 0;
      goto LABEL_7;
    }
    if ( v13 > 0x199999999999999LL )
      v13 = 0x199999999999999LL;
    v37 = 80 * v13;
  }
  v39 = a4;
  v43 = a2;
  v46 = a2 - v8;
  v38 = sub_22077B0(v37);
  v14 = v46;
  a2 = v43;
  a4 = v39;
  v47 = v38;
  v40 = v38 + v37;
  v15 = v38 + 80;
LABEL_7:
  v16 = v47 + v14;
  if ( v16 )
  {
    v17 = *(_DWORD *)(a3 + 8);
    *(_DWORD *)(a3 + 8) = 0;
    *(_DWORD *)(v16 + 8) = v17;
    *(_QWORD *)v16 = *(_QWORD *)a3;
    v18 = *(_DWORD *)(a3 + 24);
    *(_DWORD *)(a3 + 24) = 0;
    v19 = _mm_loadu_si128(a4);
    *(_DWORD *)(v16 + 24) = v18;
    v20 = *(_QWORD *)(a3 + 16);
    *(__m128i *)(v16 + 56) = v19;
    *(_QWORD *)(v16 + 16) = v20;
    *(_QWORD *)(v16 + 32) = *(_QWORD *)(a3 + 32);
    v21 = *(_QWORD *)(a3 + 40);
    v22 = *(_BYTE *)(a3 + 48);
    *(_QWORD *)(v16 + 40) = v21;
    *(_BYTE *)(v16 + 48) = v22;
    *(_QWORD *)(v16 + 72) = a4[1].m128i_i64[0];
  }
  if ( a2 != v8 )
  {
    v23 = v47;
    v24 = v8;
    while ( !v23 )
    {
LABEL_14:
      v24 += 80;
      if ( a2 == v24 )
      {
        v15 = v23 + 160;
        goto LABEL_21;
      }
      v23 += 80;
    }
    v26 = *(_DWORD *)(v24 + 8);
    *(_DWORD *)(v23 + 8) = v26;
    if ( v26 <= 0x40 )
    {
      *(_QWORD *)v23 = *(_QWORD *)v24;
      v25 = *(_DWORD *)(v24 + 24);
      *(_DWORD *)(v23 + 24) = v25;
      if ( v25 > 0x40 )
      {
LABEL_19:
        v42 = a2;
        v45 = v24;
        sub_16A4FD0(v23 + 16, (const void **)(v24 + 16));
        a2 = v42;
        v24 = v45;
        goto LABEL_13;
      }
    }
    else
    {
      v41 = a2;
      v44 = v24;
      sub_16A4FD0(v23, (const void **)v24);
      v24 = v44;
      a2 = v41;
      v27 = *(_DWORD *)(v44 + 24);
      *(_DWORD *)(v23 + 24) = v27;
      if ( v27 > 0x40 )
        goto LABEL_19;
    }
    *(_QWORD *)(v23 + 16) = *(_QWORD *)(v24 + 16);
LABEL_13:
    *(_QWORD *)(v23 + 32) = *(_QWORD *)(v24 + 32);
    *(_QWORD *)(v23 + 40) = *(_QWORD *)(v24 + 40);
    *(_BYTE *)(v23 + 48) = *(_BYTE *)(v24 + 48);
    *(__m128i *)(v23 + 56) = _mm_loadu_si128((const __m128i *)(v24 + 56));
    *(_QWORD *)(v23 + 72) = *(_QWORD *)(v24 + 72);
    goto LABEL_14;
  }
LABEL_21:
  if ( a2 != v7 )
  {
    do
    {
      v32 = *(_DWORD *)(v11 + 8);
      *(_DWORD *)(v15 + 8) = v32;
      if ( v32 <= 0x40 )
      {
        *(_QWORD *)v15 = *(_QWORD *)v11;
        v28 = *(_DWORD *)(v11 + 24);
        *(_DWORD *)(v15 + 24) = v28;
        if ( v28 > 0x40 )
          goto LABEL_28;
      }
      else
      {
        sub_16A4FD0(v15, (const void **)v11);
        v33 = *(_DWORD *)(v11 + 24);
        *(_DWORD *)(v15 + 24) = v33;
        if ( v33 > 0x40 )
        {
LABEL_28:
          sub_16A4FD0(v15 + 16, (const void **)(v11 + 16));
          goto LABEL_25;
        }
      }
      *(_QWORD *)(v15 + 16) = *(_QWORD *)(v11 + 16);
LABEL_25:
      v29 = *(_QWORD *)(v11 + 32);
      v30 = _mm_loadu_si128((const __m128i *)(v11 + 56));
      v11 += 80;
      v15 += 80;
      *(_QWORD *)(v15 - 48) = v29;
      v31 = *(_QWORD *)(v11 - 40);
      *(__m128i *)(v15 - 24) = v30;
      *(_QWORD *)(v15 - 40) = v31;
      *(_BYTE *)(v15 - 32) = *(_BYTE *)(v11 - 32);
      *(_QWORD *)(v15 - 8) = *(_QWORD *)(v11 - 8);
    }
    while ( v7 != v11 );
  }
  for ( i = v8; v7 != i; i += 80 )
  {
    if ( *(_DWORD *)(i + 24) > 0x40u )
    {
      v35 = *(_QWORD *)(i + 16);
      if ( v35 )
        j_j___libc_free_0_0(v35);
    }
    if ( *(_DWORD *)(i + 8) > 0x40u && *(_QWORD *)i )
      j_j___libc_free_0_0(*(_QWORD *)i);
  }
  if ( v8 )
    j_j___libc_free_0(v8, a1[2] - v8);
  a1[1] = v15;
  *a1 = v47;
  a1[2] = v40;
  return v40;
}
