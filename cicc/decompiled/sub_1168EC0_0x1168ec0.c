// Function: sub_1168EC0
// Address: 0x1168ec0
//
_QWORD *__fastcall sub_1168EC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, char a5)
{
  __int64 (__fastcall *v7)(__int64 *, __int64 *, __int64, __int64, __int64, __int64); // rcx
  __m128i v8; // xmm1
  __m128i v9; // xmm0
  _QWORD *result; // rax
  __m128i v11; // [rsp+10h] [rbp-70h] BYREF
  void (__fastcall *v12)(__m128i *, __m128i *, __int64); // [rsp+20h] [rbp-60h]
  __int64 (__fastcall *v13)(__int64 *, __int64 *, __int64, __int64, __int64, __int64); // [rsp+28h] [rbp-58h]
  void *v14; // [rsp+30h] [rbp-50h] BYREF
  __m128i v15; // [rsp+38h] [rbp-48h] BYREF
  __int64 (__fastcall *v16)(_QWORD *, _QWORD *, int); // [rsp+48h] [rbp-38h]
  __int64 (__fastcall *v17)(__int64 *, __int64 *, __int64, __int64, __int64, __int64); // [rsp+50h] [rbp-30h]

  v16 = sub_1168C60;
  v7 = v17;
  *(_QWORD *)a1 = a1 + 16;
  v8 = _mm_loadu_si128(&v15);
  v13 = v7;
  *(_QWORD *)(a1 + 8) = 0x1000000000LL;
  v11.m128i_i64[0] = a1;
  v9 = _mm_loadu_si128(&v11);
  v17 = sub_1168C00;
  *(_QWORD *)(a1 + 216) = a2;
  *(_QWORD *)(a1 + 144) = a1 + 160;
  *(_QWORD *)(a1 + 224) = a1 + 272;
  *(_QWORD *)(a1 + 152) = 0x200000000LL;
  *(_QWORD *)(a1 + 232) = a1 + 288;
  v11 = v8;
  v15 = v9;
  v14 = &unk_49DA0D8;
  *(_WORD *)(a1 + 252) = 512;
  v12 = 0;
  *(_QWORD *)(a1 + 240) = 0;
  *(_DWORD *)(a1 + 248) = 0;
  *(_BYTE *)(a1 + 254) = 7;
  *(_QWORD *)(a1 + 256) = 0;
  *(_QWORD *)(a1 + 272) = &unk_49D94D0;
  *(_WORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 280) = a3;
  *(_QWORD *)(a1 + 264) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_QWORD *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 288) = &unk_49DA0D8;
  *(_QWORD *)(a1 + 312) = 0;
  sub_1168C60((_QWORD *)(a1 + 296), &v15, 2);
  *(_QWORD *)(a1 + 320) = v17;
  *(_QWORD *)(a1 + 312) = v16;
  nullsub_63();
  sub_B32BF0(&v14);
  if ( v12 )
    v12(&v11, &v11, 3);
  *(_QWORD *)(a1 + 328) = a4;
  result = (_QWORD *)(a1 + 360);
  *(_BYTE *)(a1 + 336) = a5;
  *(_QWORD *)(a1 + 344) = 0;
  *(_QWORD *)(a1 + 352) = 1;
  do
  {
    if ( result )
      *result = -4096;
    result += 2;
  }
  while ( result != (_QWORD *)(a1 + 424) );
  return result;
}
