// Function: sub_D89990
// Address: 0xd89990
//
__int64 __fastcall sub_D89990(__int64 a1, __int64 a2)
{
  __m128i v2; // xmm1
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 v5; // rdx
  __m128i v6; // xmm0
  __m128i v7; // xmm2
  void (__fastcall *v8)(_QWORD, _QWORD, _QWORD); // rax
  __int64 v9; // rcx
  __int64 v10; // rax
  __int64 v11; // r13
  __m128i v13; // [rsp+0h] [rbp-40h] BYREF
  void (__fastcall *v14)(_QWORD, _QWORD, _QWORD); // [rsp+10h] [rbp-30h]
  __int64 v15; // [rsp+18h] [rbp-28h]

  v2 = _mm_loadu_si128(&v13);
  *(_QWORD *)a1 = *(_QWORD *)a2;
  v3 = v15;
  v4 = *(_QWORD *)(a2 + 24);
  v5 = *(_QWORD *)(a2 + 32);
  *(_QWORD *)(a2 + 24) = 0;
  *(_QWORD *)(a2 + 32) = v3;
  v6 = _mm_loadu_si128((const __m128i *)(a2 + 8));
  *(__m128i *)(a2 + 8) = v2;
  v7 = _mm_loadu_si128((const __m128i *)(a1 + 8));
  v8 = *(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(a1 + 24);
  *(_QWORD *)(a1 + 24) = v4;
  v9 = *(_QWORD *)(a1 + 32);
  v14 = v8;
  v15 = v9;
  *(_QWORD *)(a1 + 32) = v5;
  v13 = v7;
  *(__m128i *)(a1 + 8) = v6;
  if ( v8 )
    v8(&v13, &v13, 3);
  v10 = *(_QWORD *)(a2 + 40);
  *(_QWORD *)(a2 + 40) = 0;
  v11 = *(_QWORD *)(a1 + 40);
  *(_QWORD *)(a1 + 40) = v10;
  if ( v11 )
  {
    sub_D85F30(*(_QWORD **)(v11 + 64));
    sub_D85E30(*(_QWORD **)(v11 + 16));
    j_j___libc_free_0(v11, 104);
  }
  return a1;
}
