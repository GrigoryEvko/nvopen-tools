// Function: sub_16F4840
// Address: 0x16f4840
//
__int64 __fastcall sub_16F4840(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  __int64 v5; // rcx
  int v6; // r8d
  __int64 v8; // r12
  __int64 v9; // rax
  _QWORD *v10; // r13
  __m128i v11; // xmm0
  __int64 v12[5]; // [rsp+8h] [rbp-28h] BYREF

  if ( (unsigned __int8)sub_16F3520(a1, a2, v12, a4, a5) )
    return v12[0] + 24;
  v8 = sub_16F45F0(a1, a2, v12[0], v5, v6);
  v9 = *(_QWORD *)a2;
  *(_QWORD *)a2 = 0;
  v10 = *(_QWORD **)v8;
  *(_QWORD *)v8 = v9;
  if ( v10 )
  {
    if ( (_QWORD *)*v10 != v10 + 2 )
      j_j___libc_free_0(*v10, v10[2] + 1LL);
    j_j___libc_free_0(v10, 32);
  }
  v11 = _mm_loadu_si128((const __m128i *)(a2 + 8));
  *(_BYTE *)(v8 + 24) = 0;
  *(__m128i *)(v8 + 8) = v11;
  return v8 + 24;
}
