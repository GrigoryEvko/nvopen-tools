// Function: sub_15099C0
// Address: 0x15099c0
//
__int64 __fastcall sub_15099C0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __m128i *a8,
        unsigned __int64 a9)
{
  unsigned __int8 v9; // r13
  char v10; // bl
  char v11; // dl
  char v12; // al
  __int64 v13; // rax
  __int64 v15; // rdx
  __m128i v16[4]; // [rsp+0h] [rbp-80h] BYREF
  char v17; // [rsp+40h] [rbp-40h]

  v9 = a4;
  v10 = a3;
  sub_14F5920(v16, a2, a3, a4, a5, a6, a7, a8, a9);
  v11 = v17 & 1;
  v12 = (2 * (v17 & 1)) | v17 & 0xFD;
  v17 = v12;
  if ( v11 )
  {
    *(_BYTE *)(a1 + 8) |= 3u;
    v17 = v12 & 0xFD;
    v13 = v16[0].m128i_i64[0];
    v16[0].m128i_i64[0] = 0;
    *(_QWORD *)a1 = v13 & 0xFFFFFFFFFFFFFFFELL;
LABEL_3:
    if ( v16[0].m128i_i64[0] )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v16[0].m128i_i64[0] + 8LL))(v16[0].m128i_i64[0]);
    return a1;
  }
  sub_1509990(a1, v16, a2, v10, v9);
  if ( (v17 & 2) != 0 )
    sub_14F5A70(v16, (__int64)v16, v15);
  if ( (v17 & 1) != 0 )
    goto LABEL_3;
  return a1;
}
