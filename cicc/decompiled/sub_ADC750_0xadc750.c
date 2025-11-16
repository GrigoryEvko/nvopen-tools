// Function: sub_ADC750
// Address: 0xadc750
//
__int64 __fastcall sub_ADC750(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        _DWORD a6,
        __int128 a7,
        __int128 a8,
        __int64 a9,
        __int64 a10,
        char a11)
{
  char v14; // bl
  __int64 v15; // r14
  __int32 v16; // ecx
  __int64 v17; // r10
  __int64 v18; // r8
  char v19; // r15
  __int64 v20; // rax
  int v21; // ecx
  int v22; // r15d
  int v23; // eax
  int v24; // esi
  int v25; // eax
  __m128i v26; // xmm3
  int v28; // eax
  __int64 v29; // [rsp+8h] [rbp-D8h]
  __int64 v30; // [rsp+10h] [rbp-D0h]
  __int64 v31; // [rsp+18h] [rbp-C8h]
  __int32 v32; // [rsp+20h] [rbp-C0h]
  int v33; // [rsp+20h] [rbp-C0h]
  int v34; // [rsp+20h] [rbp-C0h]
  __int64 v35; // [rsp+20h] [rbp-C0h]
  __m128i v37; // [rsp+50h] [rbp-90h] BYREF
  __int64 v38; // [rsp+60h] [rbp-80h]
  __m128i v39; // [rsp+90h] [rbp-50h]
  __m128i v40; // [rsp+A0h] [rbp-40h]

  v14 = 0;
  v15 = *(_QWORD *)(a1 + 8);
  v38 = 0;
  v16 = a7;
  v39 = _mm_loadu_si128((const __m128i *)&a7);
  v17 = a9;
  v18 = a10;
  v40 = _mm_loadu_si128((const __m128i *)&a8);
  v19 = a11;
  v37 = 0;
  if ( BYTE8(a8) )
  {
    v20 = 0;
    if ( (_QWORD)a8 )
    {
      v29 = a10;
      v30 = a9;
      v31 = a4;
      v32 = a7;
      v20 = sub_B9B140(v15, v39.m128i_i64[1], v40.m128i_i64[0]);
      v18 = v29;
      v17 = v30;
      a4 = v31;
      v16 = v32;
    }
    v37.m128i_i32[0] = v16;
    v14 = 1;
    v37.m128i_i64[1] = v20;
  }
  v21 = 0;
  if ( v19 )
  {
    v35 = a4;
    v28 = sub_B9B140(v15, v17, v18);
    a4 = v35;
    v21 = v28;
  }
  v22 = 0;
  if ( a5 )
  {
    v33 = v21;
    v23 = sub_B9B140(v15, a4, a5);
    v21 = v33;
    v22 = v23;
  }
  v24 = 0;
  if ( a3 )
  {
    v34 = v21;
    v25 = sub_B9B140(v15, a2, a3);
    v21 = v34;
    v24 = v25;
  }
  LOBYTE(v38) = v14;
  v26 = _mm_loadu_si128(&v37);
  return sub_B07920(v15, v24, v22, v21, 0, 1, v26.m128i_i32[0], v26.m128i_i64[1], v38);
}
