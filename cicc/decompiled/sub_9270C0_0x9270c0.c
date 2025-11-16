// Function: sub_9270C0
// Address: 0x9270c0
//
__int64 __fastcall sub_9270C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  char v4; // al
  __int64 v5; // rbx
  __int64 v8; // r12
  __m128i v9; // xmm1
  __m128i v10; // xmm2
  __int64 v11; // rax
  __int64 v12; // rax
  __m128i v13; // [rsp+0h] [rbp-70h] BYREF
  __m128i v14; // [rsp+10h] [rbp-60h] BYREF
  __m128i v15; // [rsp+20h] [rbp-50h] BYREF
  __int64 v16; // [rsp+30h] [rbp-40h]

  v3 = *(_QWORD *)(a3 + 72);
  v4 = *(_BYTE *)(a3 + 56);
  v5 = *(_QWORD *)(v3 + 16);
  if ( v4 == 91 )
  {
    sub_921EA0((__int64)&v13, a2, *(__int64 **)(a3 + 72), 0, 0, 0);
    if ( !*(_QWORD *)(a2 + 96) )
    {
      v12 = sub_945CA0(a2, byte_3F871B3, 0, 0);
      sub_92FEA0(a2, v12, 0);
    }
    sub_926800(a1, a2, v5);
  }
  else
  {
    if ( v4 != 73 )
      sub_91B8A0("can't generate l-value for this binary expression!", (_DWORD *)(a3 + 36), 1);
    if ( sub_91B770(*(_QWORD *)a3) )
    {
      sub_947F00(a1, a2, a3);
    }
    else
    {
      v8 = sub_92F410(a2, v5);
      sub_926800((__int64)&v13, a2, v3);
      sub_923130(a2, v8, v13.m128i_u64[1], v14.m128i_u32[2], v16 & 1);
      v9 = _mm_loadu_si128(&v14);
      v10 = _mm_loadu_si128(&v15);
      v11 = v16;
      *(__m128i *)a1 = _mm_loadu_si128(&v13);
      *(_QWORD *)(a1 + 48) = v11;
      *(__m128i *)(a1 + 16) = v9;
      *(__m128i *)(a1 + 32) = v10;
    }
  }
  return a1;
}
