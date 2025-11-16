// Function: sub_1287240
// Address: 0x1287240
//
__int64 __fastcall sub_1287240(__int64 a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // r15
  char v4; // al
  __int64 v5; // rbx
  __int64 v8; // r12
  __int64 v9; // rcx
  __int64 v10; // r8
  __m128i v11; // xmm1
  __m128i v12; // xmm2
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // rax
  __m128i v16; // [rsp+0h] [rbp-60h] BYREF
  __m128i v17; // [rsp+10h] [rbp-50h] BYREF
  __m128i v18; // [rsp+20h] [rbp-40h] BYREF

  v3 = *(_QWORD *)(a3 + 72);
  v4 = *(_BYTE *)(a3 + 56);
  v5 = *(_QWORD *)(v3 + 16);
  if ( v4 == 91 )
  {
    sub_127FF60((__int64)&v16, (__int64)a2, *(__int64 **)(a3 + 72), 0, 0, 0);
    if ( !a2[7] )
    {
      v15 = sub_12A4D50(a2, byte_3F871B3, 0, 0);
      sub_1290AF0(a2, v15, 0);
    }
    sub_1286D80(a1, a2, v5, v13, v14);
  }
  else
  {
    if ( v4 != 73 )
      sub_127B550("can't generate l-value for this binary expression!", (_DWORD *)(a3 + 36), 1);
    if ( sub_127B420(*(_QWORD *)a3) )
    {
      sub_12A6CC0(a1, a2, a3);
    }
    else
    {
      v8 = sub_128F980(a2, v5);
      sub_1286D80((__int64)&v16, a2, v3, v9, v10);
      sub_1280F50(a2, v8, v16.m128i_u64[1], v17.m128i_u32[0], v18.m128i_i8[8] & 1);
      v11 = _mm_loadu_si128(&v17);
      v12 = _mm_loadu_si128(&v18);
      *(__m128i *)a1 = _mm_loadu_si128(&v16);
      *(__m128i *)(a1 + 16) = v11;
      *(__m128i *)(a1 + 32) = v12;
    }
  }
  return a1;
}
