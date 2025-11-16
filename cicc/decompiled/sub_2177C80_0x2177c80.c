// Function: sub_2177C80
// Address: 0x2177c80
//
__int64 *__fastcall sub_2177C80(__int64 a1, double a2, double a3, __m128i a4, __int64 a5, __int64 *a6, char a7)
{
  __int64 *v8; // rax
  __int64 v9; // rsi
  __int64 v10; // r14
  __int64 v11; // r15
  const void ***v12; // rax
  int v13; // edx
  __int64 v14; // r9
  unsigned __int64 v15; // rdx
  __int64 *v16; // rbx
  __int64 v17; // r8
  unsigned __int64 v18; // r11
  unsigned __int8 v19; // al
  __int64 v20; // rdx
  int v21; // r8d
  __int64 *v22; // r14
  unsigned int v24; // edx
  __int128 v25; // [rsp-10h] [rbp-90h]
  unsigned __int64 v27; // [rsp+8h] [rbp-78h]
  __int64 v28; // [rsp+20h] [rbp-60h] BYREF
  int v29; // [rsp+28h] [rbp-58h]
  _QWORD v30[10]; // [rsp+30h] [rbp-50h] BYREF

  v8 = *(__int64 **)(a1 + 32);
  v9 = *(_QWORD *)(a1 + 72);
  v10 = *v8;
  v11 = v8[1];
  v28 = v9;
  if ( v9 )
    sub_1623A60((__int64)&v28, v9, 2);
  v29 = *(_DWORD *)(a1 + 64);
  v12 = (const void ***)sub_1D252B0((__int64)a6, 5, 0, 1, 0);
  *((_QWORD *)&v25 + 1) = v11;
  *(_QWORD *)&v25 = v10;
  v16 = sub_1D37410(a6, 305, (__int64)&v28, v12, v13, v14, a2, a3, a4, v25);
  v17 = (__int64)v16;
  v18 = v15;
  v19 = 5;
  if ( a7 )
  {
    v27 = v15;
    v17 = sub_1D323C0(a6, (__int64)v16, 0, (__int64)&v28, 6, 0, a2, a3, *(double *)a4.m128i_i64);
    v19 = 6;
    v18 = v24 | v27 & 0xFFFFFFFF00000000LL;
  }
  v30[0] = sub_1D2B130(a6, (__int64)&v28, v19, 0, v17, v18, 5u, 0);
  v30[1] = v20;
  v30[2] = v16;
  v30[3] = v11 & 0xFFFFFFFF00000000LL | 1;
  v22 = sub_1D37190((__int64)a6, (__int64)v30, 2u, (__int64)&v28, v21, a2, a3, a4);
  if ( v28 )
    sub_161E7C0((__int64)&v28, v28);
  return v22;
}
