// Function: sub_B962A0
// Address: 0xb962a0
//
__int64 *__fastcall sub_B962A0(__int64 *a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // r14
  unsigned int v5; // edx
  const __m128i *v6; // rbx
  const __m128i *v7; // r12
  void **p_src; // r11
  __m128i v9; // xmm0
  char *v10; // r12
  __int64 v11; // rbx
  char *v12; // r14
  unsigned __int64 v13; // rax
  char *v14; // rbx
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // rax
  char *v19; // rbx
  __int64 *v20; // rax
  __int64 *v21; // rdx
  char *v22; // rbx
  __int64 v23; // r12
  __int64 v24; // rax
  unsigned __int64 v25; // r12
  char v27; // r10
  __int64 v28; // rdi
  int v29; // edx
  unsigned int v30; // ecx
  __int64 *v31; // rax
  __int64 v32; // r8
  __int64 v33; // rdx
  _QWORD *v34; // rax
  unsigned int v35; // eax
  int v36; // edx
  unsigned int v37; // ecx
  __int64 v38; // rdx
  __int64 *v39; // r9
  _QWORD *v40; // [rsp+8h] [rbp-B8h]
  void **v41; // [rsp+10h] [rbp-B0h]
  int v42; // [rsp+10h] [rbp-B0h]
  void **v43; // [rsp+10h] [rbp-B0h]
  __int64 *v44; // [rsp+28h] [rbp-98h] BYREF
  __m128i v45; // [rsp+30h] [rbp-90h] BYREF
  __int64 v46; // [rsp+40h] [rbp-80h]
  void *src; // [rsp+50h] [rbp-70h] BYREF
  __int64 v48; // [rsp+58h] [rbp-68h]
  _BYTE v49[96]; // [rsp+60h] [rbp-60h] BYREF

  v2 = a2 + 16;
  v3 = a2;
  src = v49;
  v5 = *(_DWORD *)(a2 + 24);
  v48 = 0x600000000LL;
  if ( !(v5 >> 1) )
    goto LABEL_42;
  if ( (*(_BYTE *)(a2 + 24) & 1) != 0 )
  {
    v6 = (const __m128i *)(a2 + 32);
    v7 = (const __m128i *)(a2 + 128);
  }
  else
  {
    v6 = *(const __m128i **)(a2 + 32);
    v7 = (const __m128i *)((char *)v6 + 24 * *(unsigned int *)(a2 + 40));
    if ( v6 == v7 )
      goto LABEL_7;
  }
  do
  {
    if ( v6->m128i_i64[0] != -4096 && v6->m128i_i64[0] != -8192 )
      break;
    v6 = (const __m128i *)((char *)v6 + 24);
  }
  while ( v6 != v7 );
LABEL_7:
  p_src = &src;
  if ( v7 == v6 )
  {
LABEL_42:
    v12 = v49;
    goto LABEL_43;
  }
  do
  {
    v9 = _mm_loadu_si128(v6);
    v45 = v9;
    v46 = v6[1].m128i_i64[0];
    if ( (v9.m128i_i64[1] & 0xFFFFFFFFFFFFFFFCLL) != 0
      && (v9.m128i_i8[8] & 3) == 1
      && *(_BYTE *)(v9.m128i_i64[1] & 0xFFFFFFFFFFFFFFFCLL) == 4 )
    {
      v27 = *(_BYTE *)(v3 + 24) & 1;
      if ( v27 )
      {
        v28 = v3 + 32;
        v29 = 3;
      }
      else
      {
        a2 = *(unsigned int *)(v3 + 40);
        v28 = *(_QWORD *)(v3 + 32);
        if ( !(_DWORD)a2 )
        {
          v35 = *(_DWORD *)(v3 + 24);
          ++*(_QWORD *)(v3 + 16);
          v44 = 0;
          v36 = (v35 >> 1) + 1;
          goto LABEL_48;
        }
        v29 = a2 - 1;
      }
      a2 = v45.m128i_i64[0];
      v30 = v29 & (((unsigned __int32)v45.m128i_i32[0] >> 9) ^ ((unsigned __int32)v45.m128i_i32[0] >> 4));
      v31 = (__int64 *)(v28 + 24LL * v30);
      v32 = *v31;
      if ( v45.m128i_i64[0] == *v31 )
      {
LABEL_37:
        v33 = (unsigned int)v48;
        v34 = v31 + 1;
        if ( (unsigned __int64)(unsigned int)v48 + 1 > HIDWORD(v48) )
        {
          a2 = (__int64)v49;
          v40 = v34;
          v41 = p_src;
          sub_C8D5F0(p_src, v49, (unsigned int)v48 + 1LL, 8);
          v33 = (unsigned int)v48;
          v34 = v40;
          p_src = v41;
        }
        *((_QWORD *)src + v33) = v34;
        LODWORD(v48) = v48 + 1;
        goto LABEL_11;
      }
      v42 = 1;
      v39 = 0;
      while ( v32 != -4096 )
      {
        if ( v32 == -8192 && !v39 )
          v39 = v31;
        v30 = v29 & (v42 + v30);
        v31 = (__int64 *)(v28 + 24LL * v30);
        v32 = *v31;
        if ( v45.m128i_i64[0] == *v31 )
          goto LABEL_37;
        ++v42;
      }
      v37 = 12;
      a2 = 4;
      if ( !v39 )
        v39 = v31;
      v35 = *(_DWORD *)(v3 + 24);
      ++*(_QWORD *)(v3 + 16);
      v44 = v39;
      v36 = (v35 >> 1) + 1;
      if ( v27 )
      {
LABEL_49:
        if ( 4 * v36 >= v37 )
        {
          v43 = p_src;
          LODWORD(a2) = 2 * a2;
        }
        else
        {
          if ( (int)a2 - *(_DWORD *)(v3 + 28) - v36 > (unsigned int)a2 >> 3 )
          {
LABEL_51:
            *(_DWORD *)(v3 + 24) = (2 * (v35 >> 1) + 2) | v35 & 1;
            v31 = v44;
            if ( *v44 != -4096 )
              --*(_DWORD *)(v3 + 28);
            v38 = v45.m128i_i64[0];
            v31[1] = 0;
            v31[2] = 0;
            *v31 = v38;
            goto LABEL_37;
          }
          v43 = p_src;
        }
        sub_B95E60(v2, a2);
        a2 = (__int64)&v45;
        sub_B926F0(v2, v45.m128i_i64, &v44);
        v35 = *(_DWORD *)(v3 + 24);
        p_src = v43;
        goto LABEL_51;
      }
      a2 = *(unsigned int *)(v3 + 40);
LABEL_48:
      v37 = 3 * a2;
      goto LABEL_49;
    }
LABEL_11:
    v6 = (const __m128i *)((char *)v6 + 24);
    if ( v6 == v7 )
      break;
    while ( v6->m128i_i64[0] == -8192 || v6->m128i_i64[0] == -4096 )
    {
      v6 = (const __m128i *)((char *)v6 + 24);
      if ( v7 == v6 )
        goto LABEL_15;
    }
  }
  while ( v6 != v7 );
LABEL_15:
  v10 = (char *)src;
  v11 = 8LL * (unsigned int)v48;
  v12 = (char *)src + v11;
  if ( src == (char *)src + v11 )
  {
LABEL_43:
    *a1 = (__int64)(a1 + 2);
    a1[1] = 0x600000000LL;
    goto LABEL_28;
  }
  _BitScanReverse64(&v13, v11 >> 3);
  sub_B8ECE0((char *)src, (__int64 *)((char *)src + v11), 2LL * (int)(63 - (v13 ^ 0x3F)));
  if ( (unsigned __int64)v11 <= 0x80 )
  {
    a2 = (__int64)v12;
    sub_B8E7A0(v10, v12);
  }
  else
  {
    v14 = v10 + 128;
    a2 = (__int64)(v10 + 128);
    sub_B8E7A0(v10, v10 + 128);
    if ( v10 + 128 != v12 )
    {
      do
      {
        while ( 1 )
        {
          v15 = *((_QWORD *)v14 - 1);
          v16 = *(_QWORD *)v14;
          v17 = (__int64)(v14 - 8);
          if ( *(_QWORD *)(*(_QWORD *)v14 + 8LL) < *(_QWORD *)(v15 + 8) )
            break;
          a2 = (__int64)v14;
          v14 += 8;
          *(_QWORD *)a2 = v16;
          if ( v14 == v12 )
            goto LABEL_21;
        }
        do
        {
          *(_QWORD *)(v17 + 8) = v15;
          a2 = v17;
          v15 = *(_QWORD *)(v17 - 8);
          v17 -= 8;
        }
        while ( *(_QWORD *)(v16 + 8) < *(_QWORD *)(v15 + 8) );
        v14 += 8;
        *(_QWORD *)a2 = v16;
      }
      while ( v14 != v12 );
    }
  }
LABEL_21:
  v18 = (unsigned int)v48;
  v19 = (char *)src;
  *a1 = (__int64)(a1 + 2);
  v12 = &v19[8 * v18];
  a1[1] = 0x600000000LL;
  if ( v19 != v12 )
  {
    v20 = *(__int64 **)v19;
    v21 = a1 + 2;
    v22 = v19 + 8;
    v23 = *v20;
    v24 = 0;
    v25 = v23 & 0xFFFFFFFFFFFFFFFCLL;
    while ( 1 )
    {
      v21[v24] = v25;
      v24 = (unsigned int)(*((_DWORD *)a1 + 2) + 1);
      *((_DWORD *)a1 + 2) = v24;
      if ( v22 == v12 )
        break;
      v25 = **(_QWORD **)v22 & 0xFFFFFFFFFFFFFFFCLL;
      if ( v24 + 1 > (unsigned __int64)*((unsigned int *)a1 + 3) )
      {
        a2 = (__int64)(a1 + 2);
        sub_C8D5F0(a1, a1 + 2, v24 + 1, 8);
        v24 = *((unsigned int *)a1 + 2);
      }
      v21 = (__int64 *)*a1;
      v22 += 8;
    }
    v12 = (char *)src;
  }
LABEL_28:
  if ( v12 != v49 )
    _libc_free(v12, a2);
  return a1;
}
