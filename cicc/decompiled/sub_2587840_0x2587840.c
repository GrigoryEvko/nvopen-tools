// Function: sub_2587840
// Address: 0x2587840
//
__int64 __fastcall sub_2587840(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rax
  int v8; // edi
  signed int v9; // r14d
  __int64 v10; // rdx
  unsigned int v11; // r13d
  __int64 v13; // rdi
  __int64 (*v14)(void); // rax
  __int64 v15; // rdx
  __int8 v16; // cl
  __m128i *v17; // rax
  __m128i v18; // xmm0
  __m128i v19; // xmm1
  __int8 v20; // si
  __int64 v21; // rdi
  __m128i v22; // xmm0
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // rcx
  __int64 v28; // rdx
  __int64 v29; // rax
  __int64 v30; // [rsp+0h] [rbp-D0h] BYREF
  __int64 v31; // [rsp+8h] [rbp-C8h]
  __m128i v32; // [rsp+10h] [rbp-C0h] BYREF
  __m128i v33; // [rsp+20h] [rbp-B0h]
  __m128i v34[3]; // [rsp+30h] [rbp-A0h] BYREF
  __int64 v35; // [rsp+60h] [rbp-70h]
  char *v36; // [rsp+68h] [rbp-68h] BYREF
  __int64 v37; // [rsp+70h] [rbp-60h]
  char v38; // [rsp+78h] [rbp-58h] BYREF
  __m128i v39; // [rsp+80h] [rbp-50h] BYREF
  __int64 v40; // [rsp+90h] [rbp-40h]
  _BYTE v41[56]; // [rsp+98h] [rbp-38h] BYREF

  v7 = *(_QWORD *)a2;
  v8 = *(_DWORD *)(a2 + 16);
  v36 = &v38;
  v37 = 0;
  v35 = v7;
  if ( v8 )
  {
    sub_2538240((__int64)&v36, (char **)(a2 + 8), a3, a4, a5, a6);
    v9 = **(_DWORD **)a1;
    v39.m128i_i64[1] = (__int64)v41;
    v40 = 0;
    v39.m128i_i64[0] = v35;
    if ( (_DWORD)v37 )
      sub_2538550((__int64)&v39.m128i_i64[1], (__int64)&v36, v24, (unsigned int)v37, v25, v26);
  }
  else
  {
    v9 = **(_DWORD **)a1;
    v39.m128i_i64[0] = v7;
    v39.m128i_i64[1] = (__int64)v41;
    v40 = 0;
  }
  v30 = sub_254CA10((__int64)&v39, v9);
  v31 = v10;
  if ( (_BYTE *)v39.m128i_i64[1] != v41 )
    _libc_free(v39.m128i_u64[1]);
  if ( (unsigned __int8)sub_2509800(&v30)
    && (v13 = sub_2587260(*(_QWORD *)(a1 + 8), v30, v31, *(_QWORD *)(a1 + 16), 0, 0, 1)) != 0 )
  {
    v14 = *(__int64 (**)(void))(*(_QWORD *)v13 + 112LL);
    if ( (char *)v14 == (char *)sub_2534DF0 )
    {
      v15 = *(_QWORD *)(v13 + 104);
      v16 = *(_BYTE *)(v13 + 112);
      v33 = _mm_loadu_si128((const __m128i *)(v13 + 104));
      v32 = v33;
    }
    else
    {
      v27 = v14();
      v29 = v28;
      v32.m128i_i64[0] = v27;
      v15 = v27;
      v32.m128i_i64[1] = v29;
      v16 = v29;
    }
    v17 = *(__m128i **)(a1 + 24);
    v32.m128i_i64[0] = v15;
    v32.m128i_i8[8] = v16;
    v18 = _mm_loadu_si128(&v32);
    v19 = _mm_loadu_si128(v17);
    v20 = v17->m128i_i8[8];
    v21 = v17->m128i_i64[0];
    v34[1] = v18;
    v34[0] = v19;
    if ( v20 )
    {
      if ( v16 )
      {
        if ( v21 == v15 )
        {
          v34[0].m128i_i64[0] = v15;
          v16 = v20;
          v39 = _mm_loadu_si128(v34);
        }
        else
        {
          v16 = 1;
          v15 = 0;
        }
      }
      else
      {
        v39 = v19;
        v16 = v20;
        v15 = v21;
      }
    }
    else
    {
      v39 = v18;
    }
    v39.m128i_i64[0] = v15;
    v11 = 1;
    v39.m128i_i8[8] = v16;
    v22 = _mm_loadu_si128(&v39);
    *v17 = v22;
    v23 = *(_QWORD *)(a1 + 24);
    v34[2] = v22;
    if ( *(_BYTE *)(v23 + 8) )
      LOBYTE(v11) = *(_QWORD *)v23 != 0;
  }
  else
  {
    v11 = 0;
  }
  if ( v36 != &v38 )
    _libc_free((unsigned __int64)v36);
  return v11;
}
