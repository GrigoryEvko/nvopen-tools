// Function: sub_398DBD0
// Address: 0x398dbd0
//
void __fastcall sub_398DBD0(__int64 a1, char a2, __int64 a3, __int64 a4, int a5, int a6)
{
  __int64 v7; // rbx
  __int64 v8; // rax
  __int64 v9; // rbx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rdx
  __m128i *v14; // rax
  __m128i *v15; // rdi
  __m128i *v16; // [rsp+0h] [rbp-40h] BYREF
  __int64 v17; // [rsp+8h] [rbp-38h]
  __m128i v18[3]; // [rsp+10h] [rbp-30h] BYREF

  v7 = *(_QWORD *)(a1 + 8);
  v8 = *(unsigned int *)(v7 + 8);
  if ( (unsigned int)v8 >= *(_DWORD *)(v7 + 12) )
  {
    sub_16CD150(*(_QWORD *)(a1 + 8), (const void *)(v7 + 16), 0, 1, a5, a6);
    v8 = *(unsigned int *)(v7 + 8);
  }
  *(_BYTE *)(*(_QWORD *)v7 + v8) = a2;
  ++*(_DWORD *)(v7 + 8);
  if ( *(_BYTE *)(a1 + 24) )
  {
    v9 = *(_QWORD *)(a1 + 16);
    sub_16E2FC0((__int64 *)&v16, a3);
    v13 = *(unsigned int *)(v9 + 8);
    if ( (unsigned int)v13 >= *(_DWORD *)(v9 + 12) )
    {
      sub_12BE710(v9, 0, v13, v10, v11, v12);
      LODWORD(v13) = *(_DWORD *)(v9 + 8);
    }
    v14 = (__m128i *)(*(_QWORD *)v9 + 32LL * (unsigned int)v13);
    if ( v14 )
    {
      v14->m128i_i64[0] = (__int64)v14[1].m128i_i64;
      if ( v16 == v18 )
      {
        v14[1] = _mm_load_si128(v18);
      }
      else
      {
        v14->m128i_i64[0] = (__int64)v16;
        v14[1].m128i_i64[0] = v18[0].m128i_i64[0];
      }
      v14->m128i_i64[1] = v17;
      v17 = 0;
      v18[0].m128i_i8[0] = 0;
      ++*(_DWORD *)(v9 + 8);
    }
    else
    {
      v15 = v16;
      *(_DWORD *)(v9 + 8) = v13 + 1;
      if ( v15 != v18 )
        j_j___libc_free_0((unsigned __int64)v15);
    }
  }
}
