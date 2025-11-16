// Function: sub_26E1BB0
// Address: 0x26e1bb0
//
__int64 __fastcall sub_26E1BB0(__int64 a1, __int64 *a2, const __m128i *a3)
{
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r13
  _QWORD *v10; // rcx
  char v11; // di
  unsigned int v13; // eax

  v5 = sub_22077B0(0x38u);
  v6 = *a2;
  *(__m128i *)(v5 + 40) = _mm_loadu_si128(a3);
  *(_QWORD *)(v5 + 32) = v6;
  v7 = sub_26E1A10(a1, (unsigned int *)(v5 + 32));
  v9 = v7;
  if ( v8 )
  {
    v10 = (_QWORD *)(a1 + 8);
    v11 = 1;
    if ( !v7 && (_QWORD *)v8 != v10 )
    {
      v13 = *(_DWORD *)(v8 + 32);
      if ( *(_DWORD *)(v5 + 32) >= v13 )
      {
        v11 = 0;
        if ( *(_DWORD *)(v5 + 32) == v13 )
          v11 = *(_DWORD *)(v5 + 36) < *(_DWORD *)(v8 + 36);
      }
    }
    sub_220F040(v11, v5, (_QWORD *)v8, v10);
    ++*(_QWORD *)(a1 + 40);
    return v5;
  }
  else
  {
    j_j___libc_free_0(v5);
    return v9;
  }
}
