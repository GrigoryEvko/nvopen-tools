// Function: sub_398D8F0
// Address: 0x398d8f0
//
void *__fastcall sub_398D8F0(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // r12d
  __int64 v6; // rax
  char v7; // r15
  char v8; // si
  char *v9; // rax
  char v10; // al
  __int64 v12; // rbx
  __int64 v13; // rdx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rcx
  __m128i *v17; // rax
  __int64 v18; // rbx
  __int64 v19; // r12
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rdx
  __m128i *v24; // rax
  __m128i *v25; // rdi
  __m128i *v26; // rdi
  __int64 v27; // [rsp+10h] [rbp-90h]
  __m128i *v29; // [rsp+20h] [rbp-80h] BYREF
  __int64 v30; // [rsp+28h] [rbp-78h]
  __m128i v31; // [rsp+30h] [rbp-70h] BYREF
  _QWORD v32[2]; // [rsp+40h] [rbp-60h] BYREF
  unsigned __int64 v33; // [rsp+50h] [rbp-50h]
  char *v34; // [rsp+58h] [rbp-48h]
  int v35; // [rsp+60h] [rbp-40h]
  __int64 v36; // [rsp+68h] [rbp-38h]

  v4 = 0;
  v6 = *(_QWORD *)(a1 + 8);
  v35 = 1;
  v34 = 0;
  v32[0] = &unk_49EFC48;
  v33 = 0;
  v32[1] = 0;
  v36 = v6;
  sub_16E7A40((__int64)v32, 0, 0, 0);
  do
  {
    while ( 1 )
    {
      v10 = a2;
      ++v4;
      v8 = a2 & 0x7F;
      a2 >>= 7;
      if ( !a2 )
      {
        v7 = 0;
        if ( (v10 & 0x40) == 0 )
          goto LABEL_4;
        goto LABEL_3;
      }
      if ( a2 == -1 )
      {
        v7 = 0;
        if ( (v10 & 0x40) != 0 )
          break;
      }
LABEL_3:
      v8 |= 0x80u;
      v7 = 1;
LABEL_4:
      v9 = v34;
      if ( (unsigned __int64)v34 >= v33 )
        goto LABEL_10;
LABEL_5:
      v34 = v9 + 1;
      *v9 = v8;
      if ( !v7 )
        goto LABEL_11;
    }
    v9 = v34;
    if ( (unsigned __int64)v34 < v33 )
      goto LABEL_5;
LABEL_10:
    sub_16E7DE0((__int64)v32, v8);
  }
  while ( v7 );
LABEL_11:
  if ( *(_BYTE *)(a1 + 24) )
  {
    v12 = *(_QWORD *)(a1 + 16);
    sub_16E2FC0((__int64 *)&v29, a3);
    v16 = *(unsigned int *)(v12 + 8);
    if ( (unsigned int)v16 >= *(_DWORD *)(v12 + 12) )
    {
      sub_12BE710(v12, 0, v13, v16, v14, v15);
      LODWORD(v16) = *(_DWORD *)(v12 + 8);
    }
    v17 = (__m128i *)(*(_QWORD *)v12 + 32LL * (unsigned int)v16);
    if ( v17 )
    {
      v17->m128i_i64[0] = (__int64)v17[1].m128i_i64;
      if ( v29 == &v31 )
      {
        v17[1] = _mm_load_si128(&v31);
      }
      else
      {
        v17->m128i_i64[0] = (__int64)v29;
        v17[1].m128i_i64[0] = v31.m128i_i64[0];
      }
      v17->m128i_i64[1] = v30;
      v30 = 0;
      v31.m128i_i8[0] = 0;
      ++*(_DWORD *)(v12 + 8);
    }
    else
    {
      v26 = v29;
      *(_DWORD *)(v12 + 8) = v16 + 1;
      if ( v26 != &v31 )
        j_j___libc_free_0((unsigned __int64)v26);
    }
    v27 = v4;
    if ( v4 > 1 )
    {
      v18 = 1;
      do
      {
        v29 = &v31;
        v19 = *(_QWORD *)(a1 + 16);
        sub_3984920((__int64 *)&v29, byte_3F871B3, (__int64)byte_3F871B3);
        v23 = *(unsigned int *)(v19 + 8);
        if ( (unsigned int)v23 >= *(_DWORD *)(v19 + 12) )
        {
          sub_12BE710(v19, 0, v23, v20, v21, v22);
          LODWORD(v23) = *(_DWORD *)(v19 + 8);
        }
        v24 = (__m128i *)(*(_QWORD *)v19 + 32LL * (unsigned int)v23);
        if ( v24 )
        {
          v24->m128i_i64[0] = (__int64)v24[1].m128i_i64;
          if ( v29 == &v31 )
          {
            v24[1] = _mm_load_si128(&v31);
          }
          else
          {
            v24->m128i_i64[0] = (__int64)v29;
            v24[1].m128i_i64[0] = v31.m128i_i64[0];
          }
          v24->m128i_i64[1] = v30;
          v30 = 0;
          v31.m128i_i8[0] = 0;
          ++*(_DWORD *)(v19 + 8);
        }
        else
        {
          v25 = v29;
          *(_DWORD *)(v19 + 8) = v23 + 1;
          if ( v25 != &v31 )
            j_j___libc_free_0((unsigned __int64)v25);
        }
        ++v18;
      }
      while ( v18 != v27 );
    }
  }
  v32[0] = &unk_49EFD28;
  return sub_16E7960((__int64)v32);
}
