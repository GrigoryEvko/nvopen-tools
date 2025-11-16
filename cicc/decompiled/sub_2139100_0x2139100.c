// Function: sub_2139100
// Address: 0x2139100
//
__int64 *__fastcall sub_2139100(__int64 a1, unsigned __int64 a2, __int64 a3, double a4, double a5, __m128i a6)
{
  __int64 v6; // rcx
  unsigned __int8 *v10; // rax
  __int64 v11; // rsi
  unsigned __int64 v12; // rdx
  unsigned int v13; // r14d
  __int64 v14; // rax
  unsigned int v15; // edx
  unsigned int v16; // ebx
  unsigned __int64 v17; // rdx
  __int64 v18; // r8
  __int64 v19; // r9
  __int128 v20; // rax
  __int64 *v21; // r12
  __int64 v23; // [rsp+8h] [rbp-58h]
  __int64 v24; // [rsp+10h] [rbp-50h]
  unsigned __int64 v25; // [rsp+18h] [rbp-48h]
  __int64 *v26; // [rsp+18h] [rbp-48h]
  __int64 v27; // [rsp+20h] [rbp-40h] BYREF
  int v28; // [rsp+28h] [rbp-38h]

  v6 = a1;
  v10 = (unsigned __int8 *)(*(_QWORD *)(a2 + 40) + 16LL * (unsigned int)a3);
  v11 = *(_QWORD *)(a2 + 72);
  v12 = *((_QWORD *)v10 + 1);
  v13 = *v10;
  v27 = v11;
  v25 = v12;
  if ( v11 )
  {
    sub_1623A60((__int64)&v27, v11, 2);
    v6 = a1;
  }
  v23 = v6;
  v28 = *(_DWORD *)(a2 + 64);
  v14 = sub_2138AD0(v6, a2, a3);
  v16 = v15;
  v17 = v25;
  v24 = v14;
  v26 = *(__int64 **)(v23 + 8);
  *(_QWORD *)&v20 = sub_1D2EF30(v26, v13, v17, v23, v18, v19);
  v21 = sub_1D332F0(
          v26,
          148,
          (__int64)&v27,
          *(unsigned __int8 *)(*(_QWORD *)(v24 + 40) + 16LL * v16),
          *(const void ***)(*(_QWORD *)(v24 + 40) + 16LL * v16 + 8),
          0,
          a4,
          a5,
          a6,
          v24,
          v16 | a3 & 0xFFFFFFFF00000000LL,
          v20);
  if ( v27 )
    sub_161E7C0((__int64)&v27, v27);
  return v21;
}
