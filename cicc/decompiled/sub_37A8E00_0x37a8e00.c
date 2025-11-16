// Function: sub_37A8E00
// Address: 0x37a8e00
//
__m128i *__fastcall sub_37A8E00(__int64 a1, __int64 a2)
{
  _QWORD *v3; // rax
  __int64 v4; // rsi
  unsigned __int64 v5; // r12
  __int64 v6; // r13
  unsigned __int64 v7; // r8
  __int64 v8; // r14
  unsigned int v9; // edx
  unsigned __int64 v10; // r14
  unsigned int v11; // edx
  __int64 v12; // rcx
  __m128i *v13; // r12
  __int128 v15; // [rsp-50h] [rbp-C0h]
  unsigned __int64 v16; // [rsp+0h] [rbp-70h]
  __int64 v17; // [rsp+10h] [rbp-60h]
  __int64 v18; // [rsp+20h] [rbp-50h]
  __int64 v19; // [rsp+30h] [rbp-40h] BYREF
  int v20; // [rsp+38h] [rbp-38h]

  v3 = *(_QWORD **)(a2 + 40);
  v4 = *(_QWORD *)(a2 + 80);
  v5 = v3[25];
  v6 = v3[26];
  v19 = v4;
  v7 = v3[5];
  v8 = v3[6];
  if ( v4 )
  {
    v16 = v3[5];
    sub_B96E90((__int64)&v19, v4, 1);
    v7 = v16;
  }
  v20 = *(_DWORD *)(a2 + 72);
  v18 = sub_379AB60(a1, v7, v8);
  v10 = v9 | v8 & 0xFFFFFFFF00000000LL;
  v17 = sub_379AB60(a1, v5, v6);
  v12 = *(_QWORD *)(a2 + 40);
  *((_QWORD *)&v15 + 1) = v11 | v6 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v15 = v17;
  v13 = sub_33F5F90(
          *(__int64 **)(a1 + 8),
          *(_QWORD *)v12,
          *(_QWORD *)(v12 + 8),
          (__int64)&v19,
          v18,
          v10,
          *(_QWORD *)(v12 + 80),
          *(_QWORD *)(v12 + 88),
          *(_OWORD *)(v12 + 120),
          *(_OWORD *)(v12 + 160),
          v15,
          *(_OWORD *)(v12 + 240),
          *(unsigned __int16 *)(a2 + 96),
          *(_QWORD *)(a2 + 104),
          *(const __m128i **)(a2 + 112),
          (*(_WORD *)(a2 + 32) >> 7) & 7,
          (*(_BYTE *)(a2 + 33) & 4) != 0,
          (*(_BYTE *)(a2 + 33) & 8) != 0);
  if ( v19 )
    sub_B91220((__int64)&v19, v19);
  return v13;
}
