// Function: sub_39BA6B0
// Address: 0x39ba6b0
//
__int64 __fastcall sub_39BA6B0(__int64 a1, _QWORD *a2, __int64 a3)
{
  __int64 v3; // rax
  __int16 v6; // dx
  bool v7; // al
  __m128i v8; // xmm3
  __m128i v9; // xmm2
  __m128i v10; // xmm1
  __m128i v11; // xmm0
  unsigned int v12; // r14d
  __m128i v13[4]; // [rsp+0h] [rbp-150h] BYREF
  __int64 v14; // [rsp+40h] [rbp-110h]
  __m128i v15; // [rsp+48h] [rbp-108h]
  __m128i v16; // [rsp+58h] [rbp-F8h]
  __m128i v17; // [rsp+68h] [rbp-E8h]
  __m128i v18; // [rsp+78h] [rbp-D8h]
  __int64 v19; // [rsp+88h] [rbp-C8h]
  __int64 v20; // [rsp+90h] [rbp-C0h]
  __int64 v21; // [rsp+98h] [rbp-B8h]
  __int64 v22; // [rsp+A0h] [rbp-B0h]
  __int64 v23; // [rsp+A8h] [rbp-A8h]
  __int64 v24; // [rsp+B0h] [rbp-A0h]
  __int64 v25; // [rsp+B8h] [rbp-98h]
  _BYTE *v26; // [rsp+C0h] [rbp-90h]
  __int64 v27; // [rsp+C8h] [rbp-88h]
  _BYTE v28[128]; // [rsp+D0h] [rbp-80h] BYREF

  v3 = *(_QWORD *)(*(_QWORD *)(a3 + 16) + 8LL);
  if ( (v3 & 4) != 0
    || ((v6 = *(_WORD *)(a3 + 46), (v6 & 4) != 0) || (v6 & 8) == 0
      ? (v7 = (v3 & 0x40) != 0)
      : (v7 = sub_1E15D00(a3, 0x40u, 1)),
        v7) )
  {
    *(_BYTE *)(a1 + 16) = 0;
    *(_QWORD *)a1 = a1 + 16;
    *(_QWORD *)(a1 + 8) = 0;
  }
  else
  {
    v20 = 0;
    v8 = _mm_loadu_si128(xmmword_452E800);
    v9 = _mm_loadu_si128(&xmmword_452E800[1]);
    v21 = 0;
    v10 = _mm_loadu_si128(&xmmword_452E800[2]);
    v11 = _mm_loadu_si128(&xmmword_452E800[3]);
    v22 = 0;
    v13[0] = v8;
    v13[1] = v9;
    v14 = unk_452E840;
    v19 = unk_452E840;
    v13[2] = v10;
    v15 = v8;
    v16 = v9;
    v17 = v10;
    v13[3] = v11;
    v18 = v11;
    v27 = 0x1000000000LL;
    v23 = 0;
    v24 = 0;
    v25 = 0;
    v26 = v28;
    sub_1F4B6B0(v13, a2);
    v12 = sub_1F4BF20((__int64)v13, a3, 1);
    sub_1F4C150((__int64)v13, a3);
    sub_39BA2A0(a1, v12, *(double *)v11.m128i_i64);
    if ( v26 != v28 )
      _libc_free((unsigned __int64)v26);
  }
  return a1;
}
