// Function: sub_382AC40
// Address: 0x382ac40
//
unsigned __int8 *__fastcall sub_382AC40(__int64 *a1, __int64 a2, __m128i a3)
{
  __int16 *v4; // rax
  unsigned __int16 v5; // si
  __int64 v6; // r8
  __int64 (__fastcall *v7)(__int64, __int64, unsigned int, __int64); // rax
  unsigned int v8; // r15d
  __int64 v9; // r8
  __int64 v10; // rsi
  _QWORD *v11; // r9
  __int64 v12; // r12
  __int64 v13; // rdx
  unsigned __int64 v14; // r13
  unsigned __int8 *v15; // r12
  __int64 v17; // rdx
  __int128 v18; // [rsp-10h] [rbp-70h]
  __int64 v19; // [rsp+0h] [rbp-60h]
  _QWORD *v20; // [rsp+8h] [rbp-58h]
  __int64 v21; // [rsp+10h] [rbp-50h] BYREF
  int v22; // [rsp+18h] [rbp-48h]
  __int64 v23; // [rsp+20h] [rbp-40h]

  v4 = *(__int16 **)(a2 + 48);
  v5 = *v4;
  v6 = *((_QWORD *)v4 + 1);
  v7 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v7 == sub_2D56A50 )
  {
    HIWORD(v8) = 0;
    sub_2FE6CC0((__int64)&v21, *a1, *(_QWORD *)(a1[1] + 64), v5, v6);
    LOWORD(v8) = v22;
    v9 = v23;
  }
  else
  {
    v8 = v7(*a1, *(_QWORD *)(a1[1] + 64), v5, v6);
    v9 = v17;
  }
  v10 = *(_QWORD *)(a2 + 80);
  v11 = (_QWORD *)a1[1];
  v12 = *(_QWORD *)(a2 + 40);
  v13 = 5LL * *(unsigned int *)(a2 + 64);
  v21 = v10;
  v14 = 0xCCCCCCCCCCCCCCCDLL * v13;
  if ( v10 )
  {
    v19 = v9;
    v20 = v11;
    sub_B96E90((__int64)&v21, v10, 1);
    v9 = v19;
    v11 = v20;
  }
  *((_QWORD *)&v18 + 1) = v14;
  *(_QWORD *)&v18 = v12;
  v22 = *(_DWORD *)(a2 + 72);
  v15 = sub_34102A0(v11, 498, (__int64)&v21, v8, v9, (__int64)v11, a3, v18);
  if ( v21 )
    sub_B91220((__int64)&v21, v21);
  return v15;
}
