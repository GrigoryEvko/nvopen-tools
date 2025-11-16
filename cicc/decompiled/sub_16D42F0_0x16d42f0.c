// Function: sub_16D42F0
// Address: 0x16d42f0
//
_QWORD *__fastcall sub_16D42F0(_QWORD *a1, __m128i *a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // rax
  _QWORD *v5; // r12
  __int64 v6; // rdx
  __m128i v7; // xmm0
  __m128i v8; // xmm1
  __int64 v9; // rax
  __int64 v10; // rax

  v2 = sub_22077B0(88);
  v3 = v2;
  if ( v2 )
  {
    *(_BYTE *)(v2 + 36) = 0;
    *(_QWORD *)(v2 + 8) = 0x100000001LL;
    *(_QWORD *)(v2 + 24) = 0;
    *(_DWORD *)(v2 + 32) = 0;
    *(_QWORD *)v2 = &unk_49EF5D0;
    *(_DWORD *)(v2 + 40) = 0;
    *(_QWORD *)(v2 + 16) = &unk_49EF540;
    v4 = sub_22077B0(16);
    v5 = (_QWORD *)v4;
    if ( v4 )
    {
      sub_222D5E0(v4);
      *v5 = &unk_49EF518;
    }
    v6 = *(_QWORD *)(v3 + 80);
    *(_QWORD *)(v3 + 48) = v5;
    v7 = _mm_loadu_si128(a2);
    v8 = _mm_loadu_si128((const __m128i *)(v3 + 56));
    *(_QWORD *)(v3 + 16) = &unk_49EF588;
    v9 = a2[1].m128i_i64[0];
    a2[1].m128i_i64[0] = 0;
    *(_QWORD *)(v3 + 72) = v9;
    v10 = a2[1].m128i_i64[1];
    a2[1].m128i_i64[1] = v6;
    *(_QWORD *)(v3 + 80) = v10;
    *a2 = v8;
    *(__m128i *)(v3 + 56) = v7;
  }
  a1[1] = v3;
  *a1 = v3 + 16;
  return a1;
}
