// Function: sub_2E92060
// Address: 0x2e92060
//
__int64 sub_2E92060()
{
  __int64 v0; // rax
  __int64 v1; // r12
  __m128i v2; // xmm1
  void (__fastcall *v3)(_QWORD, _QWORD, _QWORD); // rax
  __m128i v4; // xmm0
  __int64 v5; // rdx
  __int128 *v6; // rax
  __m128i v8; // [rsp+0h] [rbp-30h] BYREF
  void (__fastcall *v9)(_QWORD, _QWORD, _QWORD); // [rsp+10h] [rbp-20h]
  __int64 v10; // [rsp+18h] [rbp-18h]

  v9 = 0;
  v0 = sub_22077B0(0xE8u);
  v1 = v0;
  if ( v0 )
  {
    *(_QWORD *)(v0 + 8) = 0;
    v2 = _mm_loadu_si128((const __m128i *)(v0 + 200));
    *(_QWORD *)(v0 + 16) = &unk_50201D4;
    *(_QWORD *)(v0 + 56) = v0 + 104;
    *(_QWORD *)(v0 + 112) = v0 + 160;
    *(_QWORD *)v0 = off_4A29080;
    v3 = v9;
    *(_DWORD *)(v1 + 88) = 1065353216;
    *(_DWORD *)(v1 + 144) = 1065353216;
    v4 = _mm_loadu_si128(&v8);
    *(_DWORD *)(v1 + 24) = 2;
    v8 = v2;
    *(__m128i *)(v1 + 200) = v4;
    *(_QWORD *)(v1 + 32) = 0;
    *(_QWORD *)(v1 + 40) = 0;
    *(_QWORD *)(v1 + 48) = 0;
    *(_QWORD *)(v1 + 64) = 1;
    *(_QWORD *)(v1 + 72) = 0;
    *(_QWORD *)(v1 + 80) = 0;
    *(_QWORD *)(v1 + 96) = 0;
    *(_QWORD *)(v1 + 104) = 0;
    *(_QWORD *)(v1 + 120) = 1;
    *(_QWORD *)(v1 + 128) = 0;
    *(_QWORD *)(v1 + 136) = 0;
    *(_QWORD *)(v1 + 152) = 0;
    *(_QWORD *)(v1 + 160) = 0;
    *(_BYTE *)(v1 + 168) = 0;
    *(_QWORD *)(v1 + 176) = 0;
    *(_QWORD *)(v1 + 184) = 0;
    *(_QWORD *)(v1 + 192) = 0;
    v9 = 0;
    *(_QWORD *)(v1 + 216) = v3;
    v5 = *(_QWORD *)(v1 + 224);
    *(_QWORD *)(v1 + 224) = v10;
    v10 = v5;
    v6 = sub_BC2B00();
    sub_2E91FE0((__int64)v6);
  }
  if ( v9 )
    v9(&v8, &v8, 3);
  return v1;
}
