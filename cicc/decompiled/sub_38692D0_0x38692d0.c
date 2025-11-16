// Function: sub_38692D0
// Address: 0x38692d0
//
void __fastcall sub_38692D0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 *a4,
        __int64 a5,
        __int64 a6,
        __m128i a7,
        __m128i a8,
        __int64 a9)
{
  __int64 v10; // rax
  __int64 v11; // r14
  __int64 v12; // rax
  __int64 v13; // r12
  __int64 v14; // rax

  v10 = sub_22077B0(0x168u);
  v11 = v10;
  if ( v10 )
    sub_14586F0(v10, a3, a2);
  *(_QWORD *)a1 = v11;
  v12 = sub_22077B0(0x160u);
  if ( v12 )
  {
    *(_BYTE *)v12 = 0;
    *(_QWORD *)(v12 + 8) = v12 + 24;
    *(_QWORD *)(v12 + 16) = 0x200000000LL;
    *(_QWORD *)(v12 + 160) = 0x200000000LL;
    *(_QWORD *)(v12 + 152) = v12 + 168;
    *(_QWORD *)(v12 + 264) = a3;
    *(_QWORD *)(v12 + 272) = v12 + 288;
    *(_QWORD *)(v12 + 280) = 0x400000000LL;
  }
  *(_QWORD *)(a1 + 8) = v12;
  v13 = *(_QWORD *)a1;
  v14 = sub_22077B0(0x150u);
  if ( v14 )
  {
    *(_QWORD *)v14 = v13;
    *(_QWORD *)(v14 + 48) = v14 + 64;
    *(_QWORD *)(v14 + 56) = 0x1000000000LL;
    *(_WORD *)(v14 + 216) = 256;
    *(_QWORD *)(v14 + 8) = a2;
    *(_QWORD *)(v14 + 16) = 0;
    *(_QWORD *)(v14 + 24) = 0;
    *(_QWORD *)(v14 + 32) = 0;
    *(_DWORD *)(v14 + 40) = 0;
    *(_DWORD *)(v14 + 192) = 0;
    *(_QWORD *)(v14 + 208) = 0xFFFFFFFFLL;
    *(_BYTE *)(v14 + 218) = 1;
    *(_QWORD *)(v14 + 224) = v14 + 240;
    *(_QWORD *)(v14 + 232) = 0x800000000LL;
  }
  *(_QWORD *)(a1 + 16) = v14;
  *(_WORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 24) = a2;
  *(_QWORD *)(a1 + 32) = 0;
  *(_QWORD *)(a1 + 40) = -1;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_DWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 104) = a1 + 136;
  *(_QWORD *)(a1 + 112) = a1 + 136;
  *(_QWORD *)(a1 + 120) = 8;
  *(_DWORD *)(a1 + 128) = 0;
  if ( (unsigned __int8)sub_38636B0((__int64 *)a1, a7, a8) )
    sub_3867190(a1, a5, a9, a4, a6, a7, a8);
}
