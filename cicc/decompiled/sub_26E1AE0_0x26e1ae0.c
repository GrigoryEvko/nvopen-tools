// Function: sub_26E1AE0
// Address: 0x26e1ae0
//
__int64 __fastcall sub_26E1AE0(__int64 a1, __int64 *a2)
{
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // r13
  _QWORD *v8; // rcx
  char v9; // di
  unsigned int v11; // eax

  v3 = sub_22077B0(0x38u);
  v4 = *a2;
  *(__m128i *)(v3 + 40) = _mm_loadu_si128((const __m128i *)(a2 + 1));
  *(_QWORD *)(v3 + 32) = v4;
  v5 = sub_26E1A10(a1, (unsigned int *)(v3 + 32));
  v7 = v5;
  if ( v6 )
  {
    v8 = (_QWORD *)(a1 + 8);
    v9 = 1;
    if ( !v5 && (_QWORD *)v6 != v8 )
    {
      v11 = *(_DWORD *)(v6 + 32);
      if ( *(_DWORD *)(v3 + 32) >= v11 )
      {
        v9 = 0;
        if ( *(_DWORD *)(v3 + 32) == v11 )
          v9 = *(_DWORD *)(v3 + 36) < *(_DWORD *)(v6 + 36);
      }
    }
    sub_220F040(v9, v3, (_QWORD *)v6, v8);
    ++*(_QWORD *)(a1 + 40);
    return v3;
  }
  else
  {
    j_j___libc_free_0(v3);
    return v7;
  }
}
