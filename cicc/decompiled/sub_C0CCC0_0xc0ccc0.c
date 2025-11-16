// Function: sub_C0CCC0
// Address: 0xc0ccc0
//
__int64 __fastcall sub_C0CCC0(__int64 a1, char a2)
{
  int v3; // edx
  __int64 result; // rax
  int v5; // eax
  bool v6; // zf
  __int64 v7; // rax
  int v8; // edx
  int v9; // eax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // r12
  char *v13; // rax
  char *v14; // r13
  int v15; // esi
  char *v16; // rax
  __int64 v17; // r12
  __int64 v18; // r14
  int v19; // edx
  char *v20; // rsi
  int v21; // eax
  void *v22; // rdi
  unsigned __int64 v23; // rsi
  unsigned __int64 v24; // r8
  char *v25; // r9
  const void ***v26; // r12
  const void ***i; // r10
  __int64 v28; // rax
  unsigned __int64 v29; // rax
  const void **v30; // r14
  __int64 v31; // r15
  char v32; // cl
  unsigned __int64 v33; // r13
  char *v34; // rsi
  int v35; // eax
  unsigned __int64 v36; // rax
  int v37; // ecx
  unsigned int v38; // esi
  int v39; // edx
  __m128i v40; // xmm0
  int v41; // ecx
  unsigned int v42; // esi
  int v43; // edx
  __m128i v44; // xmm1
  _BYTE *v45; // [rsp+0h] [rbp-90h]
  const void ***v46; // [rsp+8h] [rbp-88h]
  char v47; // [rsp+14h] [rbp-7Ch]
  unsigned __int64 v48; // [rsp+18h] [rbp-78h]
  char *v49; // [rsp+20h] [rbp-70h]
  __int64 v50; // [rsp+30h] [rbp-60h] BYREF
  __int64 v51; // [rsp+38h] [rbp-58h] BYREF
  void *src[2]; // [rsp+40h] [rbp-50h] BYREF
  char *v53; // [rsp+50h] [rbp-40h]

  *(_BYTE *)(a1 + 45) = 1;
  if ( !a2 )
    goto LABEL_2;
  v11 = *(unsigned int *)(a1 + 16);
  src[0] = 0;
  src[1] = 0;
  v53 = 0;
  if ( !v11 )
  {
    v22 = 0;
    goto LABEL_37;
  }
  v12 = 8 * v11;
  v13 = (char *)sub_22077B0(8 * v11);
  v14 = v13;
  if ( (char *)src[1] - (char *)src[0] > 0 )
  {
    memmove(v13, src[0], (char *)src[1] - (char *)src[0]);
    j_j___libc_free_0(src[0], v53 - (char *)src[0]);
  }
  v15 = *(_DWORD *)(a1 + 16);
  v16 = &v14[v12];
  src[0] = v14;
  src[1] = v14;
  v53 = &v14[v12];
  if ( !v15 )
    goto LABEL_65;
  v17 = *(_QWORD *)(a1 + 8);
  v18 = v17 + 24LL * *(unsigned int *)(a1 + 24);
  if ( v17 == v18 )
    goto LABEL_65;
  while ( 1 )
  {
    v19 = *(_DWORD *)(v17 + 12);
    if ( v19 )
      break;
    if ( *(_QWORD *)v17 != -1 )
      goto LABEL_22;
LABEL_64:
    v17 += 24;
    if ( v18 == v17 )
      goto LABEL_65;
  }
  if ( v19 == 1 && *(_QWORD *)v17 == -2 )
    goto LABEL_64;
LABEL_22:
  if ( v18 != v17 )
  {
    v20 = v14;
LABEL_27:
    v51 = v17;
    if ( v16 == v20 )
    {
      sub_C0C5F0((__int64)src, v20, &v51);
      v20 = (char *)src[1];
    }
    else
    {
      if ( v20 )
      {
        *(_QWORD *)v20 = v17;
        v20 = (char *)src[1];
      }
      v20 += 8;
      src[1] = v20;
    }
    v17 += 24;
    if ( v17 == v18 )
    {
LABEL_35:
      v22 = src[0];
      v23 = (v20 - (char *)src[0]) >> 3;
      goto LABEL_38;
    }
    while ( 1 )
    {
      v21 = *(_DWORD *)(v17 + 12);
      if ( v21 )
      {
        if ( v21 != 1 || *(_QWORD *)v17 != -2 )
        {
LABEL_25:
          if ( v18 == v17 )
            goto LABEL_35;
          v16 = v53;
          goto LABEL_27;
        }
      }
      else if ( *(_QWORD *)v17 != -1 )
      {
        goto LABEL_25;
      }
      v17 += 24;
      if ( v18 == v17 )
        goto LABEL_35;
    }
  }
LABEL_65:
  v22 = v14;
LABEL_37:
  v23 = 0;
LABEL_38:
  sub_C0BDD0((__int64)v22, v23, 0);
  sub_C0BF50(a1);
  v24 = 0;
  v25 = 0;
  v45 = src[0];
  v26 = (const void ***)src[0];
  for ( i = (const void ***)src[1]; i != v26; v25 = v34 )
  {
    while ( 1 )
    {
      v30 = *v26;
      v31 = *(_QWORD *)(a1 + 32);
      v32 = *(_BYTE *)(a1 + 44);
      v33 = *((unsigned int *)*v26 + 2);
      v34 = (char *)**v26;
      if ( v33 > v24 )
        break;
      if ( *((_DWORD *)*v26 + 2) )
      {
        v46 = i;
        v47 = *(_BYTE *)(a1 + 44);
        v48 = v24;
        v49 = v25;
        v35 = memcmp(&v25[v24 - v33], v34, *((unsigned int *)*v26 + 2));
        v25 = v49;
        v24 = v48;
        v32 = v47;
        i = v46;
        if ( v35 )
          break;
      }
      v36 = v31 - (*(_DWORD *)(a1 + 40) != 6) - v33;
      if ( (v36 & ~(-1LL << v32)) != 0 )
        break;
      ++v26;
      v30[2] = (const void *)v36;
      if ( i == v26 )
        goto LABEL_48;
    }
    v28 = -(1LL << v32) & ((1LL << v32) + v31 - 1);
    v30[2] = (const void *)v28;
    v29 = v33 + v28;
    v6 = *(_DWORD *)(a1 + 40) == 6;
    *(_QWORD *)(a1 + 32) = v29;
    if ( !v6 )
      *(_QWORD *)(a1 + 32) = v29 + 1;
    ++v26;
    v24 = v33;
  }
LABEL_48:
  if ( v45 )
    j_j___libc_free_0(v45, v53 - v45);
LABEL_2:
  v3 = *(_DWORD *)(a1 + 40);
  if ( ((v3 - 2) & 0xFFFFFFFD) == 0 || v3 == 9 )
    *(_QWORD *)(a1 + 32) = 4
                         * ((*(_QWORD *)(a1 + 32) != 0)
                          + ((*(_QWORD *)(a1 + 32) - (unsigned __int64)(*(_QWORD *)(a1 + 32) != 0)) >> 2));
  if ( ((v3 - 3) & 0xFFFFFFFD) != 0 )
  {
    result = (unsigned int)(v3 - 4);
    if ( (unsigned int)result > 1 )
      goto LABEL_7;
LABEL_10:
    v5 = sub_C94890(" ", 1);
    src[0] = " ";
    LODWORD(src[1]) = 1;
    HIDWORD(src[1]) = v5;
    v6 = (unsigned __int8)sub_C0C4A0(a1, (char **)src, &v50) == 0;
    v7 = v50;
    if ( !v6 )
    {
      *(_QWORD *)(v50 + 16) = 0;
      v8 = *(_DWORD *)(a1 + 40);
      result = v7 + 16;
      goto LABEL_12;
    }
    v41 = *(_DWORD *)(a1 + 16);
    v42 = *(_DWORD *)(a1 + 24);
    v51 = v50;
    ++*(_QWORD *)a1;
    v43 = v41 + 1;
    if ( 4 * (v41 + 1) >= 3 * v42 )
    {
      v42 *= 2;
    }
    else if ( v42 - *(_DWORD *)(a1 + 20) - v43 > v42 >> 3 )
    {
      goto LABEL_57;
    }
    sub_C0C780(a1, v42);
    sub_C0C4A0(a1, (char **)src, &v51);
    v43 = *(_DWORD *)(a1 + 16) + 1;
    v7 = v51;
LABEL_57:
    *(_DWORD *)(a1 + 16) = v43;
    if ( *(_DWORD *)(v7 + 12) || *(_QWORD *)v7 != -1 )
      --*(_DWORD *)(a1 + 20);
    v44 = _mm_loadu_si128((const __m128i *)src);
    result = v7 + 16;
    *(_QWORD *)result = 0;
    *(__m128i *)(result - 16) = v44;
    *(_QWORD *)result = 0;
    v8 = *(_DWORD *)(a1 + 40);
LABEL_12:
    if ( !v8 )
    {
LABEL_13:
      v9 = sub_C94890(byte_3F871B3, 0);
      src[0] = (void *)byte_3F871B3;
      LODWORD(src[1]) = 0;
      HIDWORD(src[1]) = v9;
      v6 = (unsigned __int8)sub_C0C4A0(a1, (char **)src, &v50) == 0;
      v10 = v50;
      if ( !v6 )
      {
        *(_QWORD *)(v50 + 16) = 0;
        return v10 + 16;
      }
      v37 = *(_DWORD *)(a1 + 16);
      v38 = *(_DWORD *)(a1 + 24);
      v51 = v50;
      ++*(_QWORD *)a1;
      v39 = v37 + 1;
      if ( 4 * (v37 + 1) >= 3 * v38 )
      {
        v38 *= 2;
      }
      else if ( v38 - *(_DWORD *)(a1 + 20) - v39 > v38 >> 3 )
      {
LABEL_52:
        *(_DWORD *)(a1 + 16) = v39;
        if ( *(_DWORD *)(v10 + 12) || *(_QWORD *)v10 != -1 )
          --*(_DWORD *)(a1 + 20);
        v40 = _mm_loadu_si128((const __m128i *)src);
        result = v10 + 16;
        *(_QWORD *)result = 0;
        *(__m128i *)(result - 16) = v40;
        *(_QWORD *)result = 0;
        return result;
      }
      sub_C0C780(a1, v38);
      sub_C0C4A0(a1, (char **)src, &v51);
      v39 = *(_DWORD *)(a1 + 16) + 1;
      v10 = v51;
      goto LABEL_52;
    }
  }
  else
  {
    *(_QWORD *)(a1 + 32) = 8
                         * ((*(_QWORD *)(a1 + 32) != 0)
                          + ((*(_QWORD *)(a1 + 32) - (unsigned __int64)(*(_QWORD *)(a1 + 32) != 0)) >> 3));
    result = (unsigned int)(v3 - 4);
    if ( (unsigned int)result <= 1 )
      goto LABEL_10;
LABEL_7:
    if ( !v3 )
      goto LABEL_13;
  }
  return result;
}
