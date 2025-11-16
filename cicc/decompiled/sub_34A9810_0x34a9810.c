// Function: sub_34A9810
// Address: 0x34a9810
//
void __fastcall sub_34A9810(__int64 a1, const __m128i *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __m128i v7; // xmm1
  unsigned int *v8; // r13
  __m128i v9; // xmm2
  __int64 v10; // rsi
  __int8 v11; // r15
  __int64 v12; // rbx
  __int64 v13; // r12
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rcx
  int v17; // r10d
  __int64 v18; // r8
  unsigned int i; // eax
  __int64 v20; // rdi
  __int64 v21; // r9
  unsigned int v22; // eax
  __int128 *v23; // rbx
  __int128 *v24; // r12
  char v25; // cl
  __int128 v26; // rax
  _QWORD v27[2]; // [rsp+18h] [rbp-E0h] BYREF
  __int128 v28; // [rsp+28h] [rbp-D0h]
  __int64 v29; // [rsp+38h] [rbp-C0h]
  __int128 v30; // [rsp+48h] [rbp-B0h]
  __int64 v31; // [rsp+58h] [rbp-A0h]
  __m128i v32; // [rsp+68h] [rbp-90h]
  __m128i v33; // [rsp+78h] [rbp-80h]
  __int64 v34; // [rsp+88h] [rbp-70h]
  unsigned int *v35; // [rsp+98h] [rbp-60h]
  __int128 v36; // [rsp+A0h] [rbp-58h]
  __int64 v37; // [rsp+B0h] [rbp-48h]
  __int64 v38; // [rsp+B8h] [rbp-40h]

  v7 = _mm_loadu_si128(a2);
  v8 = (unsigned int *)a2->m128i_i64[0];
  v27[0] = a2;
  v9 = _mm_loadu_si128(a2 + 1);
  v10 = a2[2].m128i_i64[0];
  v27[1] = a1;
  v11 = a2[1].m128i_i8[8];
  v12 = a2->m128i_i64[1];
  v32 = v7;
  v13 = a2[1].m128i_i64[0];
  v33 = v9;
  v34 = v10;
  sub_34A9250(
    v27,
    v10,
    a3,
    a4,
    a5,
    a6,
    (unsigned int *)v7.m128i_i64[0],
    v7.m128i_i64[1],
    v9.m128i_i64[0],
    v9.m128i_i8[8],
    v10);
  if ( !v11 )
  {
    v12 = qword_4F81350[0];
    v13 = qword_4F81350[1];
  }
  v14 = *(_QWORD *)(a1 + 1408);
  v15 = *(unsigned int *)(v14 + 24);
  v16 = *(_QWORD *)(v14 + 8);
  if ( (_DWORD)v15 )
  {
    v17 = 1;
    v18 = (unsigned int)(v15 - 1);
    for ( i = v18
            & (((0xBF58476D1CE4E5B9LL
               * ((unsigned int)(unsigned __int16)v13
                | ((_DWORD)v12 << 16)
                | ((unsigned __int64)(((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4)) << 32))) >> 31)
             ^ (484763065 * ((unsigned __int16)v13 | ((_DWORD)v12 << 16)))); ; i = v18 & v22 )
    {
      v20 = v16 + 56LL * i;
      v21 = *(_QWORD *)v20;
      if ( v8 == *(unsigned int **)v20 && v12 == *(_QWORD *)(v20 + 8) && v13 == *(_QWORD *)(v20 + 16) )
        break;
      if ( v21 == -4096 && *(_QWORD *)(v20 + 8) == -1 && *(_QWORD *)(v20 + 16) == -1 )
        return;
      v22 = v17 + i;
      ++v17;
    }
    if ( v20 != v16 + 56 * v15 )
    {
      v23 = *(__int128 **)(v20 + 24);
      v24 = &v23[*(unsigned int *)(v20 + 32)];
      while ( v24 != v23 )
      {
        v26 = *v23;
        v29 = 0;
        v28 = 0;
        if ( v26 == __PAIR128__(qword_4F81350[1], qword_4F81350[0]) )
        {
          v26 = 0u;
          v25 = 0;
        }
        else
        {
          v25 = 1;
        }
        v28 = v26;
        ++v23;
        LOBYTE(v29) = v25;
        v38 = v10;
        v37 = v29;
        v30 = v26;
        v31 = v29;
        v35 = v8;
        v36 = v26;
        sub_34A9250(v27, v10, *((__int64 *)&v26 + 1), v29, v18, v21, v8, v26, *((__int64 *)&v26 + 1), v25, v10);
      }
    }
  }
}
