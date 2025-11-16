// Function: sub_15C9340
// Address: 0x15c9340
//
void __fastcall sub_15C9340(__int64 a1, _BYTE *a2, size_t a3, __int64 a4)
{
  _BYTE *v6; // rdi
  size_t v8; // rax
  int v9; // edi
  __int64 v10; // rsi
  __m128i v11; // xmm0
  _BYTE *v12; // rax
  size_t v13; // rdx
  _BYTE *v14; // r14
  size_t v15; // r12
  _QWORD *v16; // rax
  _BYTE *v17; // rdi
  _BYTE *v18; // rax
  __int64 v19; // rcx
  size_t v20; // rsi
  __int64 v21; // rdi
  __m128i v22; // xmm1
  const char *v23; // r12
  size_t v24; // rax
  __int64 v25; // rax
  _QWORD *v26; // rdi
  size_t v27; // [rsp+8h] [rbp-68h] BYREF
  size_t n[2]; // [rsp+10h] [rbp-60h] BYREF
  _QWORD src[2]; // [rsp+20h] [rbp-50h] BYREF
  int v30; // [rsp+30h] [rbp-40h]
  __int64 v31; // [rsp+38h] [rbp-38h]

  v6 = (_BYTE *)(a1 + 16);
  if ( a2 )
  {
    *(_QWORD *)a1 = v6;
    v8 = a3;
    n[0] = a3;
    if ( a3 > 0xF )
    {
      v25 = sub_22409D0(a1, n, 0);
      *(_QWORD *)a1 = v25;
      v6 = (_BYTE *)v25;
      *(_QWORD *)(a1 + 16) = n[0];
    }
    else
    {
      if ( a3 == 1 )
      {
        *(_BYTE *)(a1 + 16) = *a2;
LABEL_5:
        *(_QWORD *)(a1 + 8) = v8;
        v6[v8] = 0;
        goto LABEL_7;
      }
      if ( !a3 )
        goto LABEL_5;
    }
    memcpy(v6, a2, a3);
    v8 = n[0];
    v6 = *(_BYTE **)a1;
    goto LABEL_5;
  }
  *(_QWORD *)a1 = v6;
  *(_QWORD *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
LABEL_7:
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 32) = a1 + 48;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  v9 = *(unsigned __int8 *)(a4 + 16);
  *(_BYTE *)(a1 + 48) = 0;
  if ( (_BYTE)v9 )
  {
    if ( (unsigned __int8)v9 <= 0x17u )
      goto LABEL_10;
    sub_15C9090((__int64)n, (_QWORD *)(a4 + 48));
    v22 = _mm_loadu_si128((const __m128i *)n);
    *(_QWORD *)(a1 + 80) = src[0];
    *(__m128i *)(a1 + 64) = v22;
LABEL_27:
    v9 = *(unsigned __int8 *)(a4 + 16);
    goto LABEL_10;
  }
  v10 = sub_1626D20(a4);
  if ( !v10 )
    goto LABEL_27;
  sub_15C9150((const char **)n, v10);
  v11 = _mm_loadu_si128((const __m128i *)n);
  v9 = *(unsigned __int8 *)(a4 + 16);
  *(_QWORD *)(a1 + 80) = src[0];
  *(__m128i *)(a1 + 64) = v11;
LABEL_10:
  if ( (_BYTE)v9 == 17 || (unsigned __int8)v9 <= 3u )
  {
    v12 = (_BYTE *)sub_1649960(a4);
    v14 = v12;
    v15 = v13;
    if ( v13 )
    {
      if ( *v12 == 1 )
      {
        v15 = v13 - 1;
        v14 = v12 + 1;
      }
      v27 = v15;
      n[0] = (size_t)src;
      if ( v15 > 0xF )
      {
        n[0] = sub_22409D0(n, &v27, 0);
        v26 = (_QWORD *)n[0];
        src[0] = v27;
      }
      else
      {
        if ( v15 == 1 )
        {
          LOBYTE(src[0]) = *v14;
          v16 = src;
          goto LABEL_18;
        }
        if ( !v15 )
        {
          v16 = src;
          goto LABEL_18;
        }
        v26 = src;
      }
      memcpy(v26, v14, v15);
      v15 = v27;
      v16 = (_QWORD *)n[0];
      goto LABEL_18;
    }
    if ( !v12 )
    {
      LOBYTE(src[0]) = 0;
      v17 = *(_BYTE **)(a1 + 32);
      n[0] = (size_t)src;
      goto LABEL_33;
    }
    n[0] = (size_t)src;
    v16 = src;
LABEL_18:
    n[1] = v15;
    *((_BYTE *)v16 + v15) = 0;
    v17 = *(_BYTE **)(a1 + 32);
    v18 = v17;
    if ( (_QWORD *)n[0] != src )
    {
      v19 = src[0];
      v20 = n[1];
      if ( (_BYTE *)(a1 + 48) == v17 )
      {
        *(_QWORD *)(a1 + 32) = n[0];
        *(_QWORD *)(a1 + 40) = v20;
        *(_QWORD *)(a1 + 48) = v19;
      }
      else
      {
        v21 = *(_QWORD *)(a1 + 48);
        *(_QWORD *)(a1 + 32) = n[0];
        *(_QWORD *)(a1 + 40) = v20;
        *(_QWORD *)(a1 + 48) = v19;
        if ( v18 )
        {
          n[0] = (size_t)v18;
          src[0] = v21;
          goto LABEL_22;
        }
      }
      n[0] = (size_t)src;
      v18 = src;
LABEL_22:
      n[1] = 0;
      *v18 = 0;
      if ( (_QWORD *)n[0] != src )
        j_j___libc_free_0(n[0], src[0] + 1LL);
      return;
    }
    v15 = n[1];
    if ( n[1] )
    {
      if ( n[1] == 1 )
        *v17 = src[0];
      else
        memcpy(v17, src, n[1]);
      v15 = n[1];
      v17 = *(_BYTE **)(a1 + 32);
    }
LABEL_33:
    *(_QWORD *)(a1 + 40) = v15;
    v17[v15] = 0;
    v18 = (_BYTE *)n[0];
    goto LABEL_22;
  }
  if ( (unsigned __int8)v9 <= 0x10u )
  {
    v31 = a1 + 32;
    v30 = 1;
    src[1] = 0;
    n[0] = (size_t)&unk_49EFBE0;
    src[0] = 0;
    n[1] = 0;
    sub_15537D0(a4, (__int64)n, 0, 0);
    sub_16E7BC0(n);
  }
  else if ( (unsigned __int8)v9 > 0x17u )
  {
    v23 = (const char *)sub_15F29F0((unsigned int)(v9 - 24));
    v24 = strlen(v23);
    sub_2241130(a1 + 32, 0, *(_QWORD *)(a1 + 40), v23, v24);
  }
}
