// Function: sub_31BA920
// Address: 0x31ba920
//
__int64 __fastcall sub_31BA920(__int64 a1, const __m128i *a2, __int64 *a3)
{
  void *v5; // r14
  __int64 v6; // r9
  __int64 v7; // rax
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // rax
  int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r15
  int v18; // eax
  _BYTE *v19; // rdi
  __int64 *v20; // rdx
  __int64 v21; // rax
  unsigned int v22; // ebx
  __m128i v23; // xmm1
  bool v25; // zf
  __int64 *v26; // rax
  bool v27; // al
  size_t v28; // rdx
  __m128i v29; // xmm2
  int v30; // eax
  __int64 v31; // rbx
  __int64 *v32; // rdx
  __int64 v33; // r10
  __int64 *v34; // rax
  __int64 v35; // [rsp+8h] [rbp-78h]
  __int64 v36; // [rsp+10h] [rbp-70h]
  __int64 v37; // [rsp+18h] [rbp-68h]
  void *src; // [rsp+20h] [rbp-60h] BYREF
  __int64 v39; // [rsp+28h] [rbp-58h]
  _BYTE v40[80]; // [rsp+30h] [rbp-50h] BYREF

  v5 = (void *)(a1 + 16);
  if ( (unsigned __int8)sub_31BC080(a2, a3) )
  {
    v23 = _mm_loadu_si128(a2);
    *(_QWORD *)a1 = v5;
    *(_QWORD *)(a1 + 8) = 0x200000001LL;
    *(__m128i *)(a1 + 16) = v23;
    return a1;
  }
  v7 = *a3;
  if ( !*a3 )
  {
    v29 = _mm_loadu_si128(a2);
    *(_QWORD *)a1 = v5;
    *(_QWORD *)(a1 + 8) = 0x200000001LL;
    *(__m128i *)(a1 + 16) = v29;
    return a1;
  }
  v8 = a2->m128i_i64[1];
  if ( v7 == a2->m128i_i64[0] )
  {
    if ( a3[1] == v8 )
    {
      *(_QWORD *)a1 = v5;
      *(_QWORD *)(a1 + 8) = 0x200000001LL;
      *(_OWORD *)(a1 + 16) = 0;
      return a1;
    }
    goto LABEL_5;
  }
  if ( a2->m128i_i64[0] )
  {
LABEL_5:
    if ( sub_B445A0(*(_QWORD *)(v8 + 16), *(_QWORD *)(v7 + 16))
      || sub_B445A0(*(_QWORD *)(a3[1] + 16), *(_QWORD *)(a2->m128i_i64[0] + 16)) )
    {
      v9 = a2->m128i_i64[0];
      v8 = 0;
      v6 = 0;
    }
    else
    {
      v25 = !sub_B445A0(*(_QWORD *)(a2->m128i_i64[0] + 16), *(_QWORD *)(*a3 + 16));
      v26 = (__int64 *)a2;
      if ( !v25 )
        v26 = a3;
      v37 = *v26;
      v27 = sub_B445A0(*(_QWORD *)(a2->m128i_i64[1] + 16), *(_QWORD *)(a3[1] + 16));
      v6 = v37;
      if ( v27 )
        v8 = a2->m128i_i64[1];
      else
        v8 = a3[1];
      v9 = a2->m128i_i64[0];
    }
    src = v40;
    v39 = 0x200000000LL;
    if ( v6 != v9 )
    {
      v6 = sub_318B520(v6);
      v11 = (unsigned int)v39;
      v12 = v39;
      if ( (unsigned int)v39 >= (unsigned __int64)HIDWORD(v39) )
      {
        v33 = a2->m128i_i64[0];
        if ( HIDWORD(v39) < (unsigned __int64)(unsigned int)v39 + 1 )
        {
          v35 = a2->m128i_i64[0];
          v36 = v6;
          sub_C8D5F0((__int64)&src, v40, (unsigned int)v39 + 1LL, 0x10u, v10, v6);
          v11 = (unsigned int)v39;
          v33 = v35;
          v6 = v36;
        }
        v34 = (__int64 *)((char *)src + 16 * v11);
        *v34 = v33;
        v34[1] = v6;
        LODWORD(v39) = v39 + 1;
      }
      else
      {
        v13 = (__int64 *)((char *)src + 16 * (unsigned int)v39);
        if ( v13 )
        {
          v14 = a2->m128i_i64[0];
          v13[1] = v6;
          *v13 = v14;
          v12 = v39;
        }
        LODWORD(v39) = v12 + 1;
      }
    }
    goto LABEL_12;
  }
  src = v40;
  v39 = 0x200000000LL;
LABEL_12:
  if ( a2->m128i_i64[1] == v8 )
  {
    v22 = v39;
    v19 = src;
  }
  else
  {
    v15 = sub_318B4B0(v8);
    v16 = (unsigned int)v39;
    v17 = v15;
    v18 = v39;
    if ( (unsigned int)v39 >= (unsigned __int64)HIDWORD(v39) )
    {
      v31 = a2->m128i_i64[1];
      if ( HIDWORD(v39) < (unsigned __int64)(unsigned int)v39 + 1 )
      {
        sub_C8D5F0((__int64)&src, v40, (unsigned int)v39 + 1LL, 0x10u, (unsigned int)v39 + 1LL, v6);
        v16 = (unsigned int)v39;
      }
      v32 = (__int64 *)((char *)src + 16 * v16);
      *v32 = v17;
      v19 = src;
      v32[1] = v31;
      v22 = v39 + 1;
      LODWORD(v39) = v39 + 1;
    }
    else
    {
      v19 = src;
      v20 = (__int64 *)((char *)src + 16 * (unsigned int)v39);
      if ( v20 )
      {
        v21 = a2->m128i_i64[1];
        *v20 = v17;
        v19 = src;
        v20[1] = v21;
        v18 = v39;
      }
      v22 = v18 + 1;
      LODWORD(v39) = v18 + 1;
    }
  }
  *(_QWORD *)a1 = v5;
  *(_QWORD *)(a1 + 8) = 0x200000000LL;
  if ( v22 )
  {
    if ( v19 != v40 )
    {
      v30 = HIDWORD(v39);
      *(_QWORD *)a1 = v19;
      *(_DWORD *)(a1 + 8) = v22;
      *(_DWORD *)(a1 + 12) = v30;
      return a1;
    }
    v28 = 16LL * v22;
    if ( v22 <= 2
      || (sub_C8D5F0(a1, v5, v22, 0x10u, v22, v6), v5 = *(void **)a1, v19 = src, (v28 = 16LL * (unsigned int)v39) != 0) )
    {
      memcpy(v5, v19, v28);
      v19 = src;
    }
    *(_DWORD *)(a1 + 8) = v22;
  }
  if ( v19 != v40 )
    _libc_free((unsigned __int64)v19);
  return a1;
}
