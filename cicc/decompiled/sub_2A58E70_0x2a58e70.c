// Function: sub_2A58E70
// Address: 0x2a58e70
//
__int64 __fastcall sub_2A58E70(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  const __m128i *v4; // r14
  unsigned int v6; // r13d
  unsigned __int64 v7; // rcx
  __int64 v8; // rax
  __m128i *v9; // rbx
  unsigned int v10; // edx
  __m128i *v11; // rbx
  __int64 v12; // rdi
  __m128i *v13; // rdi
  __int64 v14; // r8
  __int64 v15; // rax
  _BYTE *v16; // rdi
  void *v18; // rax
  __int64 v19; // rdx
  size_t v20; // rdx
  __int64 v21; // r14
  unsigned __int64 v22; // rdi
  int v23; // eax
  unsigned __int64 v24; // rdi
  int v25; // [rsp+0h] [rbp-C0h]
  int v26; // [rsp+8h] [rbp-B8h]
  char *v27; // [rsp+8h] [rbp-B8h]
  int v28; // [rsp+8h] [rbp-B8h]
  __int32 v29; // [rsp+8h] [rbp-B8h]
  unsigned __int64 v30; // [rsp+18h] [rbp-A8h] BYREF
  __int64 v31; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v32; // [rsp+28h] [rbp-98h]
  __int64 v33; // [rsp+30h] [rbp-90h]
  __int64 v34; // [rsp+38h] [rbp-88h]
  _BYTE *v35; // [rsp+40h] [rbp-80h]
  __int64 v36; // [rsp+48h] [rbp-78h]
  _BYTE v37[32]; // [rsp+50h] [rbp-70h] BYREF
  __int64 v38; // [rsp+70h] [rbp-50h]
  __int64 v39; // [rsp+78h] [rbp-48h]
  __int64 v40; // [rsp+80h] [rbp-40h]

  v4 = (const __m128i *)&v31;
  v6 = *(_DWORD *)(a1 + 8);
  v36 = 0x400000000LL;
  v40 = a4;
  v7 = *(unsigned int *)(a1 + 12);
  v8 = v6;
  v39 = a3;
  v9 = *(__m128i **)a1;
  v10 = v6;
  v35 = v37;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v38 = a2;
  if ( (unsigned __int64)v6 + 1 > v7 )
  {
    v21 = a1 + 16;
    if ( v9 > (__m128i *)&v31 || &v31 >= &v9->m128i_i64[13 * v6] )
    {
      v9 = (__m128i *)sub_C8D7D0(a1, a1 + 16, v6 + 1LL, 0x68u, &v30, (__int64)&v31);
      sub_2A58C60(a1, v9);
      v24 = *(_QWORD *)a1;
      if ( *(_QWORD *)a1 == v21 )
      {
        *(_DWORD *)(a1 + 12) = v30;
        v8 = *(unsigned int *)(a1 + 8);
        v4 = (const __m128i *)&v31;
        *(_QWORD *)a1 = v9;
        v10 = v8;
      }
      else
      {
        v28 = v30;
        _libc_free(v24);
        *(_QWORD *)a1 = v9;
        *(_DWORD *)(a1 + 12) = v28;
        v8 = *(unsigned int *)(a1 + 8);
        v4 = (const __m128i *)&v31;
        v10 = *(_DWORD *)(a1 + 8);
      }
    }
    else
    {
      v27 = (char *)((char *)&v31 - (char *)v9);
      v9 = (__m128i *)sub_C8D7D0(a1, a1 + 16, v6 + 1LL, 0x68u, &v30, (char *)&v31 - (char *)v9);
      sub_2A58C60(a1, v9);
      v22 = *(_QWORD *)a1;
      v23 = v30;
      if ( *(_QWORD *)a1 == v21 )
      {
        *(_QWORD *)a1 = v9;
        *(_DWORD *)(a1 + 12) = v23;
      }
      else
      {
        v25 = v30;
        _libc_free(v22);
        *(_QWORD *)a1 = v9;
        *(_DWORD *)(a1 + 12) = v25;
      }
      v8 = *(unsigned int *)(a1 + 8);
      v4 = (const __m128i *)&v27[(_QWORD)v9];
      v10 = *(_DWORD *)(a1 + 8);
    }
  }
  v11 = (__m128i *)((char *)v9 + 104 * v8);
  if ( v11 )
  {
    v11->m128i_i64[0] = 0;
    v11->m128i_i64[1] = 0;
    v11[1].m128i_i64[0] = 0;
    v11[1].m128i_i32[2] = 0;
    sub_C7D6A0(0, 0, 8);
    v12 = v4[1].m128i_u32[2];
    v11[1].m128i_i32[2] = v12;
    if ( (_DWORD)v12 )
    {
      v18 = (void *)sub_C7D670(16 * v12, 8);
      v19 = v11[1].m128i_u32[2];
      v11->m128i_i64[1] = (__int64)v18;
      v11[1].m128i_i32[0] = v4[1].m128i_i32[0];
      v11[1].m128i_i32[1] = v4[1].m128i_i32[1];
      memcpy(v18, (const void *)v4->m128i_i64[1], 16 * v19);
    }
    else
    {
      v11->m128i_i64[1] = 0;
      v11[1].m128i_i64[0] = 0;
    }
    v13 = v11 + 3;
    v11[2].m128i_i64[0] = (__int64)v11[3].m128i_i64;
    v11[2].m128i_i64[1] = 0x400000000LL;
    v14 = v4[2].m128i_u32[2];
    if ( (_DWORD)v14 && &v11[2] != &v4[2] )
    {
      v20 = 8LL * (unsigned int)v14;
      if ( (unsigned int)v14 <= 4
        || (v29 = v4[2].m128i_i32[2],
            sub_C8D5F0((__int64)v11[2].m128i_i64, &v11[3], (unsigned int)v14, 8u, v14, (__int64)v11[2].m128i_i64),
            v13 = (__m128i *)v11[2].m128i_i64[0],
            LODWORD(v14) = v29,
            (v20 = 8LL * v4[2].m128i_u32[2]) != 0) )
      {
        v26 = v14;
        memcpy(v13, (const void *)v4[2].m128i_i64[0], v20);
        LODWORD(v14) = v26;
      }
      v11[2].m128i_i32[2] = v14;
    }
    v15 = v4[6].m128i_i64[0];
    v11[5] = _mm_loadu_si128(v4 + 5);
    v11[6].m128i_i64[0] = v15;
    v10 = *(_DWORD *)(a1 + 8);
  }
  v16 = v35;
  *(_DWORD *)(a1 + 8) = v10 + 1;
  if ( v16 != v37 )
    _libc_free((unsigned __int64)v16);
  sub_C7D6A0(v32, 16LL * (unsigned int)v34, 8);
  return v6;
}
