// Function: sub_26E38E0
// Address: 0x26e38e0
//
__int64 __fastcall sub_26E38E0(_QWORD *a1, __int64 a2, _QWORD **a3, const __m128i **a4)
{
  __int64 v7; // r12
  __m128i v8; // xmm0
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r13
  _QWORD *v12; // rcx
  char v13; // di
  unsigned int v15; // eax

  v7 = sub_22077B0(0x38u);
  v8 = _mm_loadu_si128(*a4);
  *(_QWORD *)(v7 + 32) = **a3;
  *(__m128i *)(v7 + 40) = v8;
  v9 = sub_26E3780(a1, a2, (unsigned int *)(v7 + 32));
  v11 = v9;
  if ( v10 )
  {
    v12 = a1 + 1;
    v13 = 1;
    if ( !v9 && (_QWORD *)v10 != v12 )
    {
      v15 = *(_DWORD *)(v10 + 32);
      if ( *(_DWORD *)(v7 + 32) >= v15 )
      {
        v13 = 0;
        if ( *(_DWORD *)(v7 + 32) == v15 )
          v13 = *(_DWORD *)(v7 + 36) < *(_DWORD *)(v10 + 36);
      }
    }
    sub_220F040(v13, v7, (_QWORD *)v10, v12);
    ++a1[5];
    return v7;
  }
  else
  {
    j_j___libc_free_0(v7);
    return v11;
  }
}
