// Function: sub_398DD10
// Address: 0x398dd10
//
void *__fastcall sub_398DD10(__int64 a1, unsigned __int64 a2, __int64 a3)
{
  unsigned int v3; // r15d
  __int64 v7; // rax
  char v8; // si
  char v9; // al
  char *v10; // rax
  __int64 v12; // rbx
  __int64 v13; // rdx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // rcx
  __m128i *v17; // rax
  __int64 i; // rbx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // rcx
  __int64 v23; // rdx
  __m128i *v24; // rax
  __m128i *v25; // rdi
  __m128i *v26; // rdi
  __int64 v27; // [rsp+18h] [rbp-88h]
  __m128i *v28; // [rsp+20h] [rbp-80h] BYREF
  __int64 v29; // [rsp+28h] [rbp-78h]
  __m128i v30; // [rsp+30h] [rbp-70h] BYREF
  _QWORD v31[2]; // [rsp+40h] [rbp-60h] BYREF
  unsigned __int64 v32; // [rsp+50h] [rbp-50h]
  char *v33; // [rsp+58h] [rbp-48h]
  int v34; // [rsp+60h] [rbp-40h]
  __int64 v35; // [rsp+68h] [rbp-38h]

  v3 = 0;
  v7 = *(_QWORD *)(a1 + 8);
  v34 = 1;
  v33 = 0;
  v31[0] = &unk_49EFC48;
  v32 = 0;
  v31[1] = 0;
  v35 = v7;
  sub_16E7A40((__int64)v31, 0, 0, 0);
  do
  {
    while ( 1 )
    {
      ++v3;
      v8 = a2 & 0x7F;
      v9 = a2 & 0x7F | 0x80;
      a2 >>= 7;
      if ( a2 )
        v8 = v9;
      v10 = v33;
      if ( (unsigned __int64)v33 >= v32 )
        break;
      ++v33;
      *v10 = v8;
      if ( !a2 )
        goto LABEL_7;
    }
    sub_16E7DE0((__int64)v31, v8);
  }
  while ( a2 );
LABEL_7:
  if ( *(_BYTE *)(a1 + 24) )
  {
    v12 = *(_QWORD *)(a1 + 16);
    sub_16E2FC0((__int64 *)&v28, a3);
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
      if ( v28 == &v30 )
      {
        v17[1] = _mm_load_si128(&v30);
      }
      else
      {
        v17->m128i_i64[0] = (__int64)v28;
        v17[1].m128i_i64[0] = v30.m128i_i64[0];
      }
      v17->m128i_i64[1] = v29;
      v29 = 0;
      v30.m128i_i8[0] = 0;
      ++*(_DWORD *)(v12 + 8);
    }
    else
    {
      v26 = v28;
      *(_DWORD *)(v12 + 8) = v16 + 1;
      if ( v26 != &v30 )
        j_j___libc_free_0((unsigned __int64)v26);
    }
    if ( v3 > 1 )
    {
      for ( i = 1; i != v3; ++i )
      {
        v19 = *(_QWORD *)(a1 + 16);
        v28 = &v30;
        v27 = v19;
        sub_3984920((__int64 *)&v28, byte_3F871B3, (__int64)byte_3F871B3);
        v22 = v27;
        v23 = *(unsigned int *)(v27 + 8);
        if ( (unsigned int)v23 >= *(_DWORD *)(v27 + 12) )
        {
          sub_12BE710(v27, 0, v23, v27, v20, v21);
          v22 = v27;
          LODWORD(v23) = *(_DWORD *)(v27 + 8);
        }
        v24 = (__m128i *)(*(_QWORD *)v22 + 32LL * (unsigned int)v23);
        if ( v24 )
        {
          v24->m128i_i64[0] = (__int64)v24[1].m128i_i64;
          if ( v28 == &v30 )
          {
            v24[1] = _mm_load_si128(&v30);
          }
          else
          {
            v24->m128i_i64[0] = (__int64)v28;
            v24[1].m128i_i64[0] = v30.m128i_i64[0];
          }
          v24->m128i_i64[1] = v29;
          v29 = 0;
          v30.m128i_i8[0] = 0;
          ++*(_DWORD *)(v22 + 8);
        }
        else
        {
          v25 = v28;
          *(_DWORD *)(v22 + 8) = v23 + 1;
          if ( v25 != &v30 )
            j_j___libc_free_0((unsigned __int64)v25);
        }
      }
    }
  }
  v31[0] = &unk_49EFD28;
  return sub_16E7960((__int64)v31);
}
