// Function: sub_F9F240
// Address: 0xf9f240
//
__int64 __fastcall sub_F9F240(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  int v4; // ecx
  __int64 v5; // r15
  __int64 v6; // rax
  __int64 v7; // r12
  int v8; // r15d
  __int64 v9; // r13
  __int64 v10; // rbx
  unsigned __int64 v11; // rax
  int v12; // r8d
  size_t v13; // r10
  char *v14; // r11
  unsigned int i; // ebx
  __int64 v16; // r15
  const void *v17; // rcx
  bool v18; // al
  int v19; // eax
  char *v20; // rdi
  const void *v21; // rsi
  bool v22; // al
  size_t v23; // rdx
  int v24; // eax
  unsigned int v26; // ebx
  int v27; // [rsp+0h] [rbp-A0h]
  size_t v28; // [rsp+0h] [rbp-A0h]
  size_t v29; // [rsp+8h] [rbp-98h]
  char *v30; // [rsp+8h] [rbp-98h]
  const void *v31; // [rsp+10h] [rbp-90h]
  int v32; // [rsp+10h] [rbp-90h]
  char *v33; // [rsp+18h] [rbp-88h]
  const void *v34; // [rsp+18h] [rbp-88h]
  int v35; // [rsp+24h] [rbp-7Ch]
  __int64 v36; // [rsp+28h] [rbp-78h]
  __m128i v37; // [rsp+30h] [rbp-70h] BYREF
  __m128i v38; // [rsp+40h] [rbp-60h]
  __m128i v39; // [rsp+50h] [rbp-50h] BYREF
  __m128i v40; // [rsp+60h] [rbp-40h] BYREF

  v3 = *(_QWORD *)(a1 + 8);
  v4 = *(_DWORD *)(a1 + 16);
  v5 = v3 + 32LL * *(unsigned int *)(a1 + 24);
  v6 = *(_QWORD *)a1;
  v39.m128i_i64[0] = a1;
  v39.m128i_i64[1] = v6;
  if ( v4 )
  {
    v40.m128i_i64[1] = v5;
    v40.m128i_i64[0] = v3;
    sub_B8D830((__int64)&v39);
    v5 = *(_QWORD *)(a1 + 8) + 32LL * *(unsigned int *)(a1 + 24);
  }
  else
  {
    v40.m128i_i64[0] = v5;
    v40.m128i_i64[1] = v5;
  }
  v7 = v40.m128i_i64[0];
  v37 = _mm_loadu_si128(&v39);
  v38 = _mm_loadu_si128(&v40);
  if ( v40.m128i_i64[0] == v5 )
    return 1;
  v36 = v5;
LABEL_5:
  v8 = *(_DWORD *)(a2 + 24);
  if ( v8 )
  {
    v9 = *(_QWORD *)(a2 + 8);
    v10 = (unsigned int)sub_C94890(*(_QWORD **)(v7 + 16), *(_QWORD *)(v7 + 24));
    v11 = sub_C94890(*(_QWORD **)v7, *(_QWORD *)(v7 + 8));
    v35 = 1;
    v12 = v8 - 1;
    v13 = *(_QWORD *)(v7 + 8);
    v14 = *(char **)v7;
    for ( i = (v8 - 1) & (((0xBF58476D1CE4E5B9LL * ((v11 << 32) | v10)) >> 31) ^ (484763065 * v10)); ; i = v12 & v26 )
    {
      v16 = v9 + 32LL * i;
      v17 = *(const void **)v16;
      v18 = v14 + 1 == 0;
      if ( *(_QWORD *)v16 == -1 )
        goto LABEL_12;
      v18 = v14 + 2 == 0;
      if ( v17 == (const void *)-2LL )
        goto LABEL_12;
      if ( v13 != *(_QWORD *)(v16 + 8) )
        goto LABEL_25;
      if ( v13 )
        break;
LABEL_13:
      v20 = *(char **)(v7 + 16);
      v21 = *(const void **)(v16 + 16);
      v22 = v20 + 1 == 0;
      if ( v21 == (const void *)-1LL )
        goto LABEL_18;
      v22 = v20 + 2 == 0;
      if ( v21 == (const void *)-2LL )
        goto LABEL_18;
      v23 = *(_QWORD *)(v7 + 24);
      if ( *(_QWORD *)(v16 + 24) == v23 )
      {
        v28 = v13;
        v30 = v14;
        v32 = v12;
        v34 = v17;
        if ( !v23 )
          goto LABEL_19;
        v24 = memcmp(v20, v21, v23);
        v17 = v34;
        v12 = v32;
        v14 = v30;
        v13 = v28;
        v22 = v24 == 0;
LABEL_18:
        if ( v22 )
        {
LABEL_19:
          v38.m128i_i64[0] += 32;
          sub_B8D830((__int64)&v37);
          v7 = v38.m128i_i64[0];
          if ( v38.m128i_i64[0] == v36 )
            return 1;
          goto LABEL_5;
        }
      }
LABEL_21:
      if ( v17 == (const void *)-1LL && *(_QWORD *)(v16 + 16) == -1 )
        return 0;
LABEL_25:
      v26 = v35 + i;
      ++v35;
    }
    v27 = v12;
    v29 = v13;
    v31 = *(const void **)v16;
    v33 = v14;
    v19 = memcmp(v14, *(const void **)v16, v13);
    v14 = v33;
    v17 = v31;
    v13 = v29;
    v12 = v27;
    v18 = v19 == 0;
LABEL_12:
    if ( !v18 )
      goto LABEL_21;
    goto LABEL_13;
  }
  return 0;
}
