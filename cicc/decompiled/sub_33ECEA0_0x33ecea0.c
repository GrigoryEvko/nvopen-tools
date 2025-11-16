// Function: sub_33ECEA0
// Address: 0x33ecea0
//
void __fastcall sub_33ECEA0(const __m128i *a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // rax
  __m128i si128; // xmm0
  __int64 v5; // rax
  __m128i v6[2]; // [rsp+0h] [rbp-160h] BYREF
  _QWORD v7[8]; // [rsp+20h] [rbp-140h] BYREF
  __int64 v8; // [rsp+60h] [rbp-100h]
  int v9; // [rsp+68h] [rbp-F8h]
  __int64 v10; // [rsp+70h] [rbp-F0h]
  __int64 v11; // [rsp+78h] [rbp-E8h]
  __int64 v12; // [rsp+80h] [rbp-E0h] BYREF
  __int32 v13; // [rsp+88h] [rbp-D8h]
  _QWORD *v14; // [rsp+90h] [rbp-D0h]
  __int64 v15; // [rsp+98h] [rbp-C8h]
  __int64 v16; // [rsp+A0h] [rbp-C0h] BYREF
  unsigned __int64 v17[2]; // [rsp+B0h] [rbp-B0h] BYREF
  _QWORD v18[20]; // [rsp+C0h] [rbp-A0h] BYREF

  v2 = a1[24].m128i_i64[0];
  v17[0] = (unsigned __int64)v18;
  v6[0] = _mm_loadu_si128(a1 + 24);
  v18[0] = a2;
  v17[1] = 0x1000000001LL;
  v3 = sub_33ECD10(1u);
  v14 = v7;
  si128 = _mm_load_si128(v6);
  v7[6] = v3;
  v8 = 0x100000000LL;
  v11 = 0xFFFFFFFFLL;
  v6[1] = si128;
  v16 = 0;
  v7[7] = 0;
  v9 = 0;
  v10 = 0;
  v15 = 0;
  v13 = si128.m128i_i32[2];
  v12 = si128.m128i_i64[0];
  v5 = *(_QWORD *)(v2 + 56);
  memset(v7, 0, 24);
  v7[3] = 328;
  v7[4] = -65536;
  v16 = v5;
  if ( v5 )
    *(_QWORD *)(v5 + 24) = &v16;
  v15 = v2 + 56;
  *(_QWORD *)(v2 + 56) = &v12;
  v7[5] = &v12;
  LODWORD(v8) = 1;
  sub_33EBD60((__int64)a1, (__int64)v17);
  sub_33CF710((__int64)v7);
  if ( (_QWORD *)v17[0] != v18 )
    _libc_free(v17[0]);
}
