// Function: sub_2CE4CC0
// Address: 0x2ce4cc0
//
__int64 __fastcall sub_2CE4CC0(_QWORD *a1, _QWORD *a2, const __m128i **a3)
{
  __int64 v5; // r12
  __m128i v6; // xmm0
  _QWORD *v7; // rax
  _QWORD *v8; // rdx
  _QWORD *v9; // r13
  _QWORD *v10; // rcx
  char v11; // di
  unsigned __int64 v13; // rax

  v5 = sub_22077B0(0x60u);
  v6 = _mm_loadu_si128(*a3);
  *(_DWORD *)(v5 + 56) = 0;
  *(_QWORD *)(v5 + 64) = 0;
  *(_QWORD *)(v5 + 72) = v5 + 56;
  *(_QWORD *)(v5 + 80) = v5 + 56;
  *(_QWORD *)(v5 + 88) = 0;
  *(__m128i *)(v5 + 32) = v6;
  v7 = sub_2CE4B30(a1, a2, (unsigned __int64 *)(v5 + 32));
  v9 = v7;
  if ( v8 )
  {
    v10 = a1 + 1;
    v11 = 1;
    if ( !v7 && v8 != v10 )
    {
      v13 = v8[4];
      if ( *(_QWORD *)(v5 + 32) == v13 )
        v11 = *(_QWORD *)(v5 + 40) < v8[5];
      else
        v11 = *(_QWORD *)(v5 + 32) < v13;
    }
    sub_220F040(v11, v5, v8, v10);
    ++a1[5];
    return v5;
  }
  else
  {
    sub_2CDF380(0);
    j_j___libc_free_0(v5);
    return (__int64)v9;
  }
}
