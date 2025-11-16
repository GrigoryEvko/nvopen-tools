// Function: sub_C6DC20
// Address: 0xc6dc20
//
__int64 __fastcall sub_C6DC20(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  __int64 v4; // rax
  _QWORD *v5; // r13
  __m128i v6; // xmm0
  __int64 v7[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( (unsigned __int8)sub_C6BF30(a1, a2, v7) )
    return v7[0] + 24;
  v3 = sub_C6D9C0(a1, a2, v7[0]);
  v4 = *(_QWORD *)a2;
  *(_QWORD *)a2 = 0;
  v5 = *(_QWORD **)v3;
  *(_QWORD *)v3 = v4;
  if ( v5 )
  {
    if ( (_QWORD *)*v5 != v5 + 2 )
      j_j___libc_free_0(*v5, v5[2] + 1LL);
    j_j___libc_free_0(v5, 32);
  }
  v6 = _mm_loadu_si128((const __m128i *)(a2 + 8));
  *(_WORD *)(v3 + 24) = 0;
  *(__m128i *)(v3 + 8) = v6;
  return v3 + 24;
}
