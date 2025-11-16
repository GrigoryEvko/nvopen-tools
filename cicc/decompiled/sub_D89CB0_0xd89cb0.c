// Function: sub_D89CB0
// Address: 0xd89cb0
//
__int64 __fastcall sub_D89CB0(__int64 a1, __m128i *a2)
{
  __m128i *v3; // rbx
  __m128i v4; // xmm1
  __int64 v5; // rax
  __int64 v6; // rcx
  __int64 v7; // rdx
  __m128i v8; // xmm0
  __m128i v9; // xmm2
  void (__fastcall *v10)(_QWORD, _QWORD, _QWORD); // rax
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // r14
  __int64 v14; // rbx
  __int64 v15; // r13
  _QWORD *v16; // rdi
  __m128i v18; // [rsp+0h] [rbp-40h] BYREF
  void (__fastcall *v19)(_QWORD, _QWORD, _QWORD); // [rsp+10h] [rbp-30h]
  __int64 v20; // [rsp+18h] [rbp-28h]

  v3 = a2;
  v4 = _mm_loadu_si128(&v18);
  *(_QWORD *)a1 = a2->m128i_i64[0];
  v5 = v20;
  v6 = a2[1].m128i_i64[1];
  v7 = a2[2].m128i_i64[0];
  a2[1].m128i_i64[1] = 0;
  a2[2].m128i_i64[0] = v5;
  v8 = _mm_loadu_si128((const __m128i *)&a2->m128i_u64[1]);
  *(__m128i *)((char *)a2 + 8) = v4;
  v9 = _mm_loadu_si128((const __m128i *)(a1 + 8));
  v10 = *(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(a1 + 24);
  *(_QWORD *)(a1 + 24) = v6;
  v11 = *(_QWORD *)(a1 + 32);
  v19 = v10;
  v20 = v11;
  *(_QWORD *)(a1 + 32) = v7;
  v18 = v9;
  *(__m128i *)(a1 + 8) = v8;
  if ( v10 )
  {
    a2 = &v18;
    v10(&v18, &v18, 3);
  }
  *(_QWORD *)(a1 + 40) = v3[2].m128i_i64[1];
  v12 = v3[3].m128i_i64[0];
  v3[3].m128i_i64[0] = 0;
  v13 = *(_QWORD *)(a1 + 48);
  *(_QWORD *)(a1 + 48) = v12;
  if ( v13 )
  {
    sub_D85A50(*(_QWORD *)(v13 + 160));
    if ( *(_BYTE *)(v13 + 76) )
    {
      v14 = *(_QWORD *)(v13 + 16);
      if ( v14 )
        goto LABEL_6;
    }
    else
    {
      _libc_free(*(_QWORD *)(v13 + 56), a2);
      v14 = *(_QWORD *)(v13 + 16);
      while ( v14 )
      {
LABEL_6:
        v15 = v14;
        sub_D86030(*(_QWORD **)(v14 + 24));
        v16 = *(_QWORD **)(v14 + 104);
        v14 = *(_QWORD *)(v14 + 16);
        sub_D85F30(v16);
        sub_D85E30(*(_QWORD **)(v15 + 56));
        j_j___libc_free_0(v15, 144);
      }
    }
    j_j___libc_free_0(v13, 192);
  }
  return a1;
}
