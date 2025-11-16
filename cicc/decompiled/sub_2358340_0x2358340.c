// Function: sub_2358340
// Address: 0x2358340
//
unsigned __int64 __fastcall sub_2358340(unsigned __int64 *a1, __int64 *a2)
{
  __int64 v2; // r13
  int v3; // eax
  __int64 v4; // rbx
  __int64 v5; // rax
  __m128i v6; // xmm5
  __m128i v7; // xmm6
  __m128i v8; // xmm7
  __m128i v9; // xmm5
  int v10; // edx
  __m128i v11; // xmm6
  unsigned __int64 result; // rax
  unsigned __int64 v13[2]; // [rsp+8h] [rbp-98h] BYREF
  __m128i v14; // [rsp+18h] [rbp-88h] BYREF
  __m128i v15; // [rsp+28h] [rbp-78h] BYREF
  __m128i v16; // [rsp+38h] [rbp-68h] BYREF
  __m128i v17; // [rsp+48h] [rbp-58h] BYREF
  __m128i v18; // [rsp+58h] [rbp-48h] BYREF
  int v19; // [rsp+68h] [rbp-38h]

  v2 = *a2;
  v3 = *((_DWORD *)a2 + 22);
  *a2 = 0;
  v19 = v3;
  v4 = *(__int64 *)((char *)a2 + 92);
  v14 = _mm_loadu_si128((const __m128i *)(a2 + 1));
  v15 = _mm_loadu_si128((const __m128i *)(a2 + 3));
  v16 = _mm_loadu_si128((const __m128i *)(a2 + 5));
  v17 = _mm_loadu_si128((const __m128i *)(a2 + 7));
  v18 = _mm_loadu_si128((const __m128i *)(a2 + 9));
  v5 = sub_22077B0(0x70u);
  if ( v5 )
  {
    v6 = _mm_loadu_si128(&v14);
    *(_QWORD *)(v5 + 8) = v2;
    v7 = _mm_loadu_si128(&v15);
    v8 = _mm_loadu_si128(&v16);
    *(_QWORD *)(v5 + 100) = v4;
    *(__m128i *)(v5 + 16) = v6;
    v9 = _mm_loadu_si128(&v17);
    *(_QWORD *)v5 = &unk_4A0D978;
    v10 = v19;
    *(__m128i *)(v5 + 32) = v7;
    v11 = _mm_loadu_si128(&v18);
    *(_DWORD *)(v5 + 96) = v10;
    *(__m128i *)(v5 + 48) = v8;
    *(__m128i *)(v5 + 64) = v9;
    *(__m128i *)(v5 + 80) = v11;
    v13[0] = v5;
    result = sub_2356EF0(a1, v13);
    if ( v13[0] )
      return (*(__int64 (__fastcall **)(unsigned __int64))(*(_QWORD *)v13[0] + 8LL))(v13[0]);
  }
  else
  {
    v13[0] = 0;
    result = sub_2356EF0(a1, v13);
    if ( v13[0] )
      result = (*(__int64 (__fastcall **)(unsigned __int64))(*(_QWORD *)v13[0] + 8LL))(v13[0]);
    if ( v2 )
      return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v2 + 8LL))(v2);
  }
  return result;
}
