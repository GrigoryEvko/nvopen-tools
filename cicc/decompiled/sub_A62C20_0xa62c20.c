// Function: sub_A62C20
// Address: 0xa62c20
//
const char **__fastcall sub_A62C20(__int64 a1, const char *a2)
{
  const char **result; // rax
  __int64 v5; // rcx
  const char **v6; // rdx
  char v7; // dl
  __int64 v8; // rdx
  int v9; // r11d
  unsigned int v10; // eax
  __int64 v11; // r14
  __int64 v12; // rdi
  _QWORD *v13; // r15
  _QWORD *v14; // r14
  _BYTE *v15; // rdi
  const void *v16; // rsi
  _BYTE *v17; // rax
  __int64 v18; // rdx
  size_t v19; // rdx
  const char *v20; // rsi
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // r14
  const __m128i *v25; // rax
  __int64 v26; // rdx
  const __m128i *v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rdi
  int v30; // eax
  __int64 v31; // rax
  __int64 v32; // [rsp+0h] [rbp-120h]
  __int64 v33; // [rsp+8h] [rbp-118h]
  int v34; // [rsp+10h] [rbp-110h]
  __int64 v35; // [rsp+10h] [rbp-110h]
  int v36; // [rsp+10h] [rbp-110h]
  _QWORD v37[2]; // [rsp+20h] [rbp-100h] BYREF
  _QWORD v38[2]; // [rsp+30h] [rbp-F0h] BYREF
  _QWORD v39[6]; // [rsp+40h] [rbp-E0h] BYREF
  _QWORD *v40; // [rsp+70h] [rbp-B0h]
  _DWORD v41[40]; // [rsp+80h] [rbp-A0h] BYREF

  if ( !*(_BYTE *)(a1 + 244) )
    goto LABEL_8;
  result = *(const char ***)(a1 + 224);
  v5 = *(unsigned int *)(a1 + 236);
  v6 = &result[v5];
  if ( result == v6 )
  {
LABEL_7:
    if ( (unsigned int)v5 < *(_DWORD *)(a1 + 232) )
    {
      *(_DWORD *)(a1 + 236) = v5 + 1;
      *v6 = a2;
      ++*(_QWORD *)(a1 + 216);
      goto LABEL_9;
    }
LABEL_8:
    result = (const char **)sub_C8CC70(a1 + 216, a2);
    if ( !v7 )
      return result;
LABEL_9:
    v37[1] = 0;
    v39[5] = 0x100000000LL;
    v37[0] = v38;
    v39[0] = &unk_49DD210;
    v40 = v37;
    LOBYTE(v38[0]) = 0;
    memset(&v39[1], 0, 32);
    sub_CB5980(v39, 0, 0, 0);
    v8 = *(unsigned int *)(a1 + 48);
    v9 = *(_DWORD *)(a1 + 32) + 1;
    v10 = v8;
    *(_DWORD *)(a1 + 32) = v9;
    if ( *(_DWORD *)(a1 + 52) <= (unsigned int)v8 )
    {
      v20 = (const char *)(a1 + 56);
      v34 = v9;
      v32 = a1 + 56;
      v33 = sub_C8D7D0(a1 + 40, a1 + 56, 0, 40, v41);
      v21 = 40LL * *(unsigned int *)(a1 + 48);
      v22 = v21 + v33;
      if ( v21 + v33 )
      {
        v20 = byte_3F871B3;
        *(_DWORD *)v22 = v34;
        sub_A4F810((__int64 *)(v22 + 8), byte_3F871B3);
        v21 = 40LL * *(unsigned int *)(a1 + 48);
      }
      v23 = *(_QWORD *)(a1 + 40);
      v24 = v23 + v21;
      if ( v23 != v23 + v21 )
      {
        v25 = (const __m128i *)(v23 + 24);
        v20 = (const char *)(v33 + 8 * ((unsigned __int64)(v24 - v23 - 40) >> 3) + 40);
        v26 = v33;
        do
        {
          if ( v26 )
          {
            *(_DWORD *)v26 = v25[-2].m128i_i32[2];
            *(_QWORD *)(v26 + 8) = v26 + 24;
            v27 = (const __m128i *)v25[-1].m128i_i64[0];
            if ( v25 == v27 )
            {
              *(__m128i *)(v26 + 24) = _mm_loadu_si128(v25);
            }
            else
            {
              *(_QWORD *)(v26 + 8) = v27;
              *(_QWORD *)(v26 + 24) = v25->m128i_i64[0];
            }
            *(_QWORD *)(v26 + 16) = v25[-1].m128i_i64[1];
            v25[-1].m128i_i64[0] = (__int64)v25;
            v25[-1].m128i_i64[1] = 0;
            v25->m128i_i8[0] = 0;
          }
          v26 += 40;
          v25 = (const __m128i *)((char *)v25 + 40);
        }
        while ( v20 != (const char *)v26 );
        v28 = *(_QWORD *)(a1 + 40);
        v24 = v28 + 40LL * *(unsigned int *)(a1 + 48);
        if ( v28 != v24 )
        {
          do
          {
            v24 -= 40;
            v29 = *(_QWORD *)(v24 + 8);
            if ( v29 != v24 + 24 )
            {
              v35 = v28;
              v20 = (const char *)(*(_QWORD *)(v24 + 24) + 1LL);
              j_j___libc_free_0(v29, v20);
              v28 = v35;
            }
          }
          while ( v28 != v24 );
          v24 = *(_QWORD *)(a1 + 40);
        }
      }
      v30 = v41[0];
      if ( v32 != v24 )
      {
        v36 = v41[0];
        _libc_free(v24, v20);
        v30 = v36;
      }
      *(_DWORD *)(a1 + 52) = v30;
      v31 = *(unsigned int *)(a1 + 48);
      *(_QWORD *)(a1 + 40) = v33;
      v11 = 40 * v31;
      *(_DWORD *)(a1 + 48) = v31 + 1;
    }
    else
    {
      v11 = 40 * v8;
      v12 = *(_QWORD *)(a1 + 40) + 40 * v8;
      if ( v12 )
      {
        *(_DWORD *)v12 = v9;
        sub_A4F810((__int64 *)(v12 + 8), byte_3F871B3);
        v10 = *(_DWORD *)(a1 + 48);
        v11 = 40LL * v10;
      }
      *(_DWORD *)(a1 + 48) = v10 + 1;
    }
    sub_A54BD0((__int64)v41, (__int64)v39);
    sub_A5C090((__int64)v41, (__int64)a2, (__int64 *)a1);
    if ( (unsigned __int8)(*a2 - 5) > 0x1Fu || *a2 == 7 )
    {
      sub_A54D10((__int64)v41, (__int64)a2);
    }
    else
    {
      sub_904010((__int64)v41, " = ");
      sub_A5F6B0((__int64)v41, a2, (__int64 *)a1);
      sub_A54D10((__int64)v41, (__int64)a2);
    }
    v13 = v40;
    v14 = (_QWORD *)(*(_QWORD *)(a1 + 40) + v11);
    v15 = (_BYTE *)v14[1];
    v16 = (const void *)*v40;
    v17 = v40 + 2;
    if ( (_QWORD *)*v40 == v40 + 2 )
    {
      v19 = v40[1];
      if ( v19 )
      {
        if ( v19 == 1 )
          *v15 = *((_BYTE *)v40 + 16);
        else
          memcpy(v15, v16, v19);
        v19 = v13[1];
        v15 = (_BYTE *)v14[1];
      }
      v14[2] = v19;
      v15[v19] = 0;
      v15 = (_BYTE *)*v13;
      goto LABEL_20;
    }
    if ( v15 == (_BYTE *)(v14 + 3) )
    {
      v14[1] = v16;
      v14[2] = v13[1];
      v14[3] = v13[2];
    }
    else
    {
      v14[1] = v16;
      v18 = v14[3];
      v14[2] = v13[1];
      v14[3] = v13[2];
      if ( v15 )
      {
        *v13 = v15;
        v13[2] = v18;
        goto LABEL_20;
      }
    }
    *v13 = v17;
    v15 = v17;
LABEL_20:
    v13[1] = 0;
    *v15 = 0;
    --*(_DWORD *)(a1 + 32);
    v39[0] = &unk_49DD210;
    result = (const char **)sub_CB5840(v39);
    if ( (_QWORD *)v37[0] != v38 )
      return (const char **)j_j___libc_free_0(v37[0], v38[0] + 1LL);
    return result;
  }
  while ( a2 != *result )
  {
    if ( v6 == ++result )
      goto LABEL_7;
  }
  return result;
}
