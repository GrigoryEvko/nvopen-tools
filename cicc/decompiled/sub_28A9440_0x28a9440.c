// Function: sub_28A9440
// Address: 0x28a9440
//
__int64 __fastcall sub_28A9440(__int64 a1, unsigned __int8 *a2)
{
  unsigned int v3; // eax
  unsigned int v4; // r12d
  const __m128i *v5; // rax
  _QWORD **v6; // rdi
  __m128i v7; // xmm2
  char v8; // al
  __m128i v10[3]; // [rsp+0h] [rbp-60h] BYREF
  char v11; // [rsp+30h] [rbp-30h]

  sub_104C6D0(*(_QWORD *)(*(_QWORD *)a1 + 32LL), **(_QWORD **)(a1 + 8), (__int64)a2);
  if ( (_BYTE)v3 )
    return 1;
  v4 = v3;
  if ( a2 == **(unsigned __int8 ***)(a1 + 8) || a2 == **(unsigned __int8 ***)(a1 + 16) )
    return 1;
  v5 = *(const __m128i **)(a1 + 32);
  v6 = *(_QWORD ***)(a1 + 24);
  v10[0] = _mm_loadu_si128(v5);
  v10[1] = _mm_loadu_si128(v5 + 1);
  v7 = _mm_loadu_si128(v5 + 2);
  v11 = 1;
  v10[2] = v7;
  v8 = sub_CF63E0(*v6, a2, v10, (__int64)(v6 + 1));
  if ( ((**(_BYTE **)(a1 + 40) & 2) == 0 || (v8 & 1) == 0) && ((**(_BYTE **)(a1 + 40) & 1) == 0 || (v8 & 2) == 0) )
    return 1;
  return v4;
}
