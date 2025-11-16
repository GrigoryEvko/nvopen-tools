// Function: sub_303DF80
// Address: 0x303df80
//
__int64 __fastcall sub_303DF80(__int64 a1, __int64 a2, __int64 a3, int a4, __int64 a5, int a6)
{
  __int64 v8; // rsi
  __int64 *v9; // rax
  int v10; // eax
  int v11; // r12d
  __int64 v12; // rdi
  int v13; // edx
  int v14; // r13d
  __int64 v15; // r12
  char v17; // [rsp-28h] [rbp-C8h]
  __int64 v18; // [rsp+8h] [rbp-98h]
  __int16 v19; // [rsp+14h] [rbp-8Ch]
  __int64 v20; // [rsp+18h] [rbp-88h]
  __m128i v21; // [rsp+20h] [rbp-80h]
  __int64 v22; // [rsp+40h] [rbp-60h] BYREF
  int v23; // [rsp+48h] [rbp-58h]
  _QWORD v24[10]; // [rsp+50h] [rbp-50h] BYREF

  v8 = *(_QWORD *)(a2 + 80);
  v22 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v22, v8, 1);
  v23 = *(_DWORD *)(a2 + 72);
  v9 = *(__int64 **)(a2 + 40);
  v20 = v9[1];
  v18 = *v9;
  v21 = _mm_loadu_si128((const __m128i *)v9 + 5);
  v10 = sub_33FAF80(a4, 214, (unsigned int)&v22, 6, 0, a6, *(_OWORD *)(v9 + 5));
  memset(v24, 0, 32);
  v11 = v10;
  v12 = *(_QWORD *)(a2 + 112);
  v14 = v13;
  v19 = *(_WORD *)(v12 + 32);
  v17 = sub_2EAC4F0(v12);
  v15 = sub_33F5040(
          a4,
          v18,
          v20,
          (unsigned int)&v22,
          v11,
          v14,
          v21.m128i_i64[0],
          v21.m128i_i64[1],
          *(_OWORD *)*(_QWORD *)(a2 + 112),
          *(_QWORD *)(*(_QWORD *)(a2 + 112) + 16LL),
          5,
          0,
          v17,
          v19,
          (__int64)v24);
  if ( v22 )
    sub_B91220((__int64)&v22, v22);
  return v15;
}
