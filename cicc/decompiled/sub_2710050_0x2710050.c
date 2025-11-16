// Function: sub_2710050
// Address: 0x2710050
//
void __fastcall sub_2710050(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v4; // edx
  __int64 v5; // rcx
  unsigned int v6; // eax
  __int64 *v7; // r13
  __int64 v8; // r8
  unsigned __int64 v9; // rdx
  __int64 *v10; // rbx
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // r9
  __int64 v14; // rbx
  __m128i *v15; // r13
  _QWORD *v16; // rax
  int v17; // r9d
  __int64 v18; // rax
  __m128i *v19; // r13
  __m128i *v20; // rdx
  _QWORD *v21; // rax
  int v22; // ebx
  __int64 *v23; // [rsp+8h] [rbp-68h]
  unsigned __int64 v24; // [rsp+18h] [rbp-58h] BYREF
  __m128i *v25; // [rsp+20h] [rbp-50h] BYREF
  __int64 v26; // [rsp+28h] [rbp-48h]
  __m128i v27[4]; // [rsp+30h] [rbp-40h] BYREF

  v4 = *(_DWORD *)(a1 + 240);
  v5 = *(_QWORD *)(a1 + 224);
  if ( v4 )
  {
    v6 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v5 + 16LL * v6);
    v8 = *v7;
    if ( a2 == *v7 )
      goto LABEL_3;
    v17 = 1;
    while ( v8 != -4096 )
    {
      v6 = (v4 - 1) & (v17 + v6);
      v7 = (__int64 *)(v5 + 16LL * v6);
      v8 = *v7;
      if ( a2 == *v7 )
        goto LABEL_3;
      ++v17;
    }
  }
  v7 = (__int64 *)(v5 + 16LL * v4);
LABEL_3:
  v9 = v7[1] & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v7[1] & 4) != 0 )
  {
    v10 = *(__int64 **)v9;
    v11 = *(_QWORD *)v9 + 8LL * *(unsigned int *)(v9 + 8);
  }
  else
  {
    v10 = v7 + 1;
    if ( !v9 )
      return;
    v11 = (__int64)(v7 + 2);
  }
  if ( v10 != (__int64 *)v11 )
  {
    v12 = sub_AA4FF0(*v10);
    if ( !v12 )
LABEL_20:
      BUG();
    while ( (unsigned int)*(unsigned __int8 *)(v12 - 24) - 80 > 1 )
    {
      if ( (__int64 *)v11 == ++v10 )
        return;
      v12 = sub_AA4FF0(*v10);
      if ( !v12 )
        goto LABEL_20;
    }
    v14 = v12 - 24;
    if ( *(_DWORD *)(a3 + 8) >= *(_DWORD *)(a3 + 12) )
    {
      v18 = sub_C8D7D0(a3, a3 + 16, 0, 0x38u, &v24, v13);
      v25 = v27;
      v19 = (__m128i *)v18;
      sub_270E4F0((__int64 *)&v25, "funclet", (__int64)"");
      v20 = (__m128i *)((char *)v19 + 56 * *(unsigned int *)(a3 + 8));
      if ( v20 )
      {
        v20->m128i_i64[0] = (__int64)v20[1].m128i_i64;
        if ( v25 == v27 )
        {
          v20[1] = _mm_load_si128(v27);
        }
        else
        {
          v20->m128i_i64[0] = (__int64)v25;
          v20[1].m128i_i64[0] = v27[0].m128i_i64[0];
        }
        v23 = (__int64 *)v20;
        v20->m128i_i64[1] = v26;
        v25 = v27;
        v26 = 0;
        v27[0].m128i_i8[0] = 0;
        v20[2].m128i_i64[0] = 0;
        v20[2].m128i_i64[1] = 0;
        v20[3].m128i_i64[0] = 0;
        v21 = (_QWORD *)sub_22077B0(8u);
        v23[4] = (__int64)v21;
        v23[6] = (__int64)(v21 + 1);
        *v21 = v14;
        v23[5] = (__int64)(v21 + 1);
      }
      if ( v25 != v27 )
        j_j___libc_free_0((unsigned __int64)v25);
      sub_B56820(a3, v19);
      v22 = v24;
      if ( a3 + 16 != *(_QWORD *)a3 )
        _libc_free(*(_QWORD *)a3);
      ++*(_DWORD *)(a3 + 8);
      *(_QWORD *)a3 = v19;
      *(_DWORD *)(a3 + 12) = v22;
    }
    else
    {
      v25 = v27;
      sub_270E4F0((__int64 *)&v25, "funclet", (__int64)"");
      v15 = (__m128i *)(*(_QWORD *)a3 + 56LL * *(unsigned int *)(a3 + 8));
      if ( v15 )
      {
        v15->m128i_i64[0] = (__int64)v15[1].m128i_i64;
        if ( v25 == v27 )
        {
          v15[1] = _mm_load_si128(v27);
        }
        else
        {
          v15->m128i_i64[0] = (__int64)v25;
          v15[1].m128i_i64[0] = v27[0].m128i_i64[0];
        }
        v15->m128i_i64[1] = v26;
        v25 = v27;
        v26 = 0;
        v27[0].m128i_i8[0] = 0;
        v15[2].m128i_i64[0] = 0;
        v15[2].m128i_i64[1] = 0;
        v15[3].m128i_i64[0] = 0;
        v16 = (_QWORD *)sub_22077B0(8u);
        v15[2].m128i_i64[0] = (__int64)v16;
        v15[3].m128i_i64[0] = (__int64)(v16 + 1);
        *v16 = v14;
        v15[2].m128i_i64[1] = (__int64)(v16 + 1);
      }
      if ( v25 != v27 )
        j_j___libc_free_0((unsigned __int64)v25);
      ++*(_DWORD *)(a3 + 8);
    }
  }
}
