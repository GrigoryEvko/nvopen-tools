// Function: sub_372B930
// Address: 0x372b930
//
void __fastcall sub_372B930(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r15
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r14
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // r12
  __m128i *v14; // rbx
  __m128i v15; // xmm0
  __int64 v16; // rdx
  __int64 m128i_i64; // rdi
  unsigned __int64 v18; // rbx
  unsigned __int64 v19; // rdi
  int v20; // ebx
  unsigned __int64 v21; // [rsp+8h] [rbp-48h]
  unsigned __int64 v22[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a1 + 16;
  v11 = sub_C8D7D0(a1, a1 + 16, a2, 0x60u, v22, a6);
  v12 = *(_QWORD *)a1;
  v13 = *(_QWORD *)a1 + 96LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v13 )
  {
    v14 = (__m128i *)v11;
    do
    {
      while ( 1 )
      {
        if ( v14 )
        {
          v15 = _mm_loadu_si128((const __m128i *)v12);
          v14[1].m128i_i32[2] = 0;
          v14[1].m128i_i64[0] = (__int64)v14[2].m128i_i64;
          v14[1].m128i_i32[3] = 4;
          *v14 = v15;
          v16 = *(unsigned int *)(v12 + 24);
          if ( (_DWORD)v16 )
            break;
        }
        v12 += 96LL;
        v14 += 6;
        if ( v13 == v12 )
          goto LABEL_7;
      }
      m128i_i64 = (__int64)v14[1].m128i_i64;
      v21 = v12;
      v14 += 6;
      sub_372A010(m128i_i64, (char **)(v12 + 16), v16, v8, v9, v10);
      v12 = v21 + 96;
    }
    while ( v13 != v21 + 96 );
LABEL_7:
    v18 = *(_QWORD *)a1;
    v13 = *(_QWORD *)a1 + 96LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v13 )
    {
      do
      {
        v13 -= 96LL;
        v19 = *(_QWORD *)(v13 + 16);
        if ( v19 != v13 + 32 )
          _libc_free(v19);
      }
      while ( v13 != v18 );
      v13 = *(_QWORD *)a1;
    }
  }
  v20 = v22[0];
  if ( v6 != v13 )
    _libc_free(v13);
  *(_QWORD *)a1 = v11;
  *(_DWORD *)(a1 + 12) = v20;
}
