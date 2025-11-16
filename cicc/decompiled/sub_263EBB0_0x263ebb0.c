// Function: sub_263EBB0
// Address: 0x263ebb0
//
__int64 __fastcall sub_263EBB0(_QWORD *a1, __int64 a2, const __m128i **a3)
{
  __int64 v5; // r12
  __m128i v6; // xmm0
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // r13
  _QWORD *v10; // rcx
  char v11; // di
  unsigned __int64 v13; // rax

  v5 = sub_22077B0(0x40u);
  v6 = _mm_loadu_si128(*a3);
  *(_QWORD *)(v5 + 48) = 0;
  *(_DWORD *)(v5 + 56) = 0;
  *(__m128i *)(v5 + 32) = v6;
  v7 = sub_263E480(a1, a2, v5 + 32);
  v9 = v7;
  if ( v8 )
  {
    v10 = a1 + 1;
    v11 = 1;
    if ( !v7 && (_QWORD *)v8 != v10 )
    {
      v13 = *(_QWORD *)(v8 + 32);
      if ( *(_QWORD *)(v5 + 32) >= v13 )
      {
        v11 = 0;
        if ( *(_QWORD *)(v5 + 32) == v13 )
          v11 = *(_DWORD *)(v5 + 40) < *(_DWORD *)(v8 + 40);
      }
    }
    sub_220F040(v11, v5, (_QWORD *)v8, v10);
    ++a1[5];
    return v5;
  }
  else
  {
    j_j___libc_free_0(v5);
    return v9;
  }
}
