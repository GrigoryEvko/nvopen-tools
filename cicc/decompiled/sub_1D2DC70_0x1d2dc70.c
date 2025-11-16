// Function: sub_1D2DC70
// Address: 0x1d2dc70
//
void __fastcall sub_1D2DC70(const __m128i *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __m128i v6; // xmm0
  __int64 v7; // rbx
  __int64 v8; // rax
  __m128i si128; // xmm0
  __int64 v10; // rax
  __m128i v11[2]; // [rsp+0h] [rbp-150h] BYREF
  _QWORD v12[7]; // [rsp+20h] [rbp-130h] BYREF
  __int64 v13; // [rsp+58h] [rbp-F8h]
  int v14; // [rsp+60h] [rbp-F0h]
  __int64 v15; // [rsp+68h] [rbp-E8h]
  int v16; // [rsp+70h] [rbp-E0h]
  __int64 v17; // [rsp+78h] [rbp-D8h] BYREF
  __int32 v18; // [rsp+80h] [rbp-D0h]
  _QWORD *v19; // [rsp+88h] [rbp-C8h]
  __int64 v20; // [rsp+90h] [rbp-C0h]
  __int64 v21; // [rsp+98h] [rbp-B8h] BYREF
  unsigned __int64 v22[2]; // [rsp+A0h] [rbp-B0h] BYREF
  _QWORD v23[20]; // [rsp+B0h] [rbp-A0h] BYREF

  v6 = _mm_loadu_si128(a1 + 11);
  v7 = a1[11].m128i_i64[0];
  v23[0] = a2;
  v22[0] = (unsigned __int64)v23;
  v11[0] = v6;
  v22[1] = 0x1000000001LL;
  v8 = sub_1D274F0(1u, a3, a4, a5, a6);
  v19 = v12;
  si128 = _mm_load_si128(v11);
  v12[5] = v8;
  v13 = 0x100000000LL;
  v11[1] = si128;
  v21 = 0;
  v12[6] = 0;
  v14 = 0;
  v20 = 0;
  v15 = 0;
  v16 = -65536;
  v18 = si128.m128i_i32[2];
  v17 = si128.m128i_i64[0];
  v10 = *(_QWORD *)(v7 + 48);
  memset(v12, 0, 24);
  v12[3] = -4294967084LL;
  v21 = v10;
  if ( v10 )
    *(_QWORD *)(v10 + 24) = &v21;
  v20 = v7 + 48;
  *(_QWORD *)(v7 + 48) = &v17;
  v12[4] = &v17;
  LODWORD(v13) = 1;
  sub_1D2D860((__int64)a1, (__int64)v22);
  sub_1D189A0((__int64)v12);
  if ( (_QWORD *)v22[0] != v23 )
    _libc_free(v22[0]);
}
