// Function: sub_1C31130
// Address: 0x1c31130
//
void __fastcall sub_1C31130(char **a1, int a2)
{
  __int64 v2; // rdx
  char *v3; // r12
  _DWORD *v4; // r14
  size_t v5; // r13
  _DWORD *v6; // rax
  char v7; // si
  size_t v8; // rbx
  const void *v9; // r15
  size_t v10; // r12
  signed __int64 v11; // rax
  size_t v12; // rdx
  const void *v13; // r12
  __m128i *v14; // r15
  int v15; // eax
  unsigned int v16; // r12d
  __m128i *v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rax
  size_t v20; // r12
  size_t v21; // rcx
  size_t v22; // rdx
  __int64 v23; // rax
  size_t v24; // rax
  __int64 v25; // [rsp-80h] [rbp-80h]
  __m128i *v26; // [rsp-78h] [rbp-78h]
  char **v27; // [rsp-68h] [rbp-68h]
  __m128i *v28; // [rsp-60h] [rbp-60h]
  size_t v29; // [rsp-60h] [rbp-60h]
  __m128i *v30; // [rsp-58h] [rbp-58h] BYREF
  size_t v31; // [rsp-50h] [rbp-50h]
  __m128i v32[4]; // [rsp-48h] [rbp-48h] BYREF

  if ( !a2 )
    return;
  v25 = (__int64)&a1[(unsigned int)(a2 - 1) + 1];
  v27 = a1;
  do
  {
    v2 = -1;
    v3 = *v27;
    v30 = v32;
    if ( v3 )
      v2 = (__int64)&v3[strlen(v3)];
    sub_1C30A70((__int64 *)&v30, v3, v2);
    v4 = *(_DWORD **)&dword_4FBA468[2];
    if ( !*(_QWORD *)&dword_4FBA468[2] )
    {
      v4 = dword_4FBA468;
      goto LABEL_34;
    }
    v5 = v31;
    v26 = v30;
    v28 = v30;
    while ( 1 )
    {
      v8 = *((_QWORD *)v4 + 5);
      v9 = (const void *)*((_QWORD *)v4 + 4);
      v10 = v8;
      if ( v5 <= v8 )
        v10 = v5;
      if ( v10 )
      {
        LODWORD(v11) = memcmp(v28, *((const void **)v4 + 4), v10);
        if ( (_DWORD)v11 )
          goto LABEL_15;
      }
      v11 = v5 - v8;
      if ( (__int64)(v5 - v8) >= 0x80000000LL )
        break;
      if ( v11 > (__int64)0xFFFFFFFF7FFFFFFFLL )
      {
LABEL_15:
        if ( (int)v11 >= 0 )
          break;
      }
      v6 = (_DWORD *)*((_QWORD *)v4 + 2);
      v7 = 1;
      if ( !v6 )
        goto LABEL_17;
LABEL_8:
      v4 = v6;
    }
    v6 = (_DWORD *)*((_QWORD *)v4 + 3);
    v7 = 0;
    if ( v6 )
      goto LABEL_8;
LABEL_17:
    v12 = v10;
    v13 = v9;
    v14 = v28;
    if ( !v7 )
      goto LABEL_18;
LABEL_34:
    if ( v4 == *(_DWORD **)&dword_4FBA468[4] )
      goto LABEL_25;
    v19 = sub_220EF80(v4);
    v5 = v31;
    v14 = v30;
    v8 = *(_QWORD *)(v19 + 40);
    v13 = *(const void **)(v19 + 32);
    v26 = v30;
    v12 = v8;
    if ( v31 <= v8 )
      v12 = v31;
LABEL_18:
    if ( v12 && (v15 = memcmp(v13, v14, v12)) != 0 )
    {
LABEL_23:
      if ( v15 < 0 )
        goto LABEL_24;
    }
    else if ( (__int64)(v8 - v5) < 0x80000000LL )
    {
      if ( (__int64)(v8 - v5) > (__int64)0xFFFFFFFF7FFFFFFFLL )
      {
        v15 = v8 - v5;
        goto LABEL_23;
      }
LABEL_24:
      if ( v4 )
      {
LABEL_25:
        v16 = 1;
        if ( v4 == dword_4FBA468 )
          goto LABEL_26;
        v20 = v31;
        v21 = *((_QWORD *)v4 + 5);
        v22 = v21;
        if ( v31 <= v21 )
          v22 = v31;
        if ( v22
          && (v29 = *((_QWORD *)v4 + 5),
              LODWORD(v23) = memcmp(v30, *((const void **)v4 + 4), v22),
              v21 = v29,
              (_DWORD)v23) )
        {
LABEL_45:
          v16 = (unsigned int)v23 >> 31;
        }
        else
        {
          v24 = v20;
          v16 = 0;
          v23 = v24 - v21;
          if ( v23 < 0x80000000LL )
          {
            if ( v23 > (__int64)0xFFFFFFFF7FFFFFFFLL )
              goto LABEL_45;
            v16 = 1;
          }
        }
LABEL_26:
        v17 = (__m128i *)sub_22077B0(64);
        v17[2].m128i_i64[0] = (__int64)v17[3].m128i_i64;
        if ( v30 == v32 )
        {
          v17[3] = _mm_load_si128(v32);
        }
        else
        {
          v17[2].m128i_i64[0] = (__int64)v30;
          v17[3].m128i_i64[0] = v32[0].m128i_i64[0];
        }
        v18 = v31;
        v32[0].m128i_i8[0] = 0;
        v31 = 0;
        v17[2].m128i_i64[1] = v18;
        v30 = v32;
        sub_220F040(v16, v17, v4, dword_4FBA468);
        ++*(_QWORD *)&dword_4FBA468[8];
        v26 = v30;
      }
      else
      {
        v26 = v14;
      }
    }
    if ( v26 != v32 )
      j_j___libc_free_0(v26, v32[0].m128i_i64[0] + 1);
    ++v27;
  }
  while ( v27 != (char **)v25 );
}
