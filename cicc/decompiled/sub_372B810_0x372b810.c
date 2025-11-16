// Function: sub_372B810
// Address: 0x372b810
//
__int64 __fastcall sub_372B810(_QWORD *a1, _QWORD *a2, const __m128i **a3)
{
  __int64 v5; // r12
  __m128i v6; // xmm1
  _QWORD *v7; // rax
  _QWORD *v8; // rdx
  _QWORD *v9; // r14
  _QWORD *v10; // rcx
  char v11; // di
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rdi

  v5 = sub_22077B0(0x78u);
  v6 = _mm_loadu_si128(*a3);
  *(_OWORD *)(v5 + 80) = 0;
  *(_QWORD *)(v5 + 56) = 0x100000000LL;
  *(_QWORD *)(v5 + 112) = 0;
  *(_QWORD *)(v5 + 48) = v5 + 64;
  *(_QWORD *)(v5 + 88) = 0;
  *(_QWORD *)(v5 + 96) = v5 + 80;
  *(_QWORD *)(v5 + 104) = v5 + 80;
  *(__m128i *)(v5 + 32) = v6;
  *(_OWORD *)(v5 + 64) = 0;
  v7 = sub_372B6C0(a1, a2, (unsigned __int64 *)(v5 + 32));
  v9 = v7;
  if ( v8 )
  {
    v10 = a1 + 1;
    v11 = 1;
    if ( !v7 && v8 != v10 )
    {
      v13 = v8[4];
      if ( *(_QWORD *)(v5 + 32) >= v13 )
      {
        v11 = 0;
        if ( *(_QWORD *)(v5 + 32) == v13 )
          v11 = *(_QWORD *)(v5 + 40) < v8[5];
      }
    }
    sub_220F040(v11, v5, v8, v10);
    ++a1[5];
    return v5;
  }
  else
  {
    sub_372A530(0);
    v14 = *(_QWORD *)(v5 + 48);
    if ( v5 + 64 != v14 )
      _libc_free(v14);
    j_j___libc_free_0(v5);
    return (__int64)v9;
  }
}
