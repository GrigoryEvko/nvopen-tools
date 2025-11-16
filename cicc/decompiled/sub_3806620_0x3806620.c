// Function: sub_3806620
// Address: 0x3806620
//
unsigned __int8 *__fastcall sub_3806620(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 (__fastcall *v4)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v5; // rax
  unsigned __int16 v6; // si
  __int64 v7; // r8
  __int64 v8; // rax
  unsigned int v9; // r13d
  __int64 v10; // r15
  int v11; // edx
  __int64 v12; // rsi
  int v13; // r9d
  unsigned __int8 *v14; // r12
  unsigned int v16; // eax
  __int64 v17; // rdx
  int v18; // [rsp+8h] [rbp-68h]
  __int64 v19; // [rsp+18h] [rbp-58h]
  __int64 v20; // [rsp+20h] [rbp-50h] BYREF
  int v21; // [rsp+28h] [rbp-48h]
  __int64 v22; // [rsp+30h] [rbp-40h]

  v4 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v5 = *(__int16 **)(a2 + 48);
  v6 = *v5;
  v7 = *((_QWORD *)v5 + 1);
  v8 = a1[1];
  if ( v4 == sub_2D56A50 )
  {
    HIWORD(v9) = 0;
    sub_2FE6CC0((__int64)&v20, *a1, *(_QWORD *)(v8 + 64), v6, v7);
    LOWORD(v9) = v21;
    v19 = v22;
  }
  else
  {
    v16 = v4(*a1, *(_QWORD *)(v8 + 64), v6, v7);
    v19 = v17;
    v9 = v16;
  }
  v10 = a1[1];
  sub_3805E70((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v12 = *(_QWORD *)(a2 + 80);
  v13 = v11;
  v20 = v12;
  if ( v12 )
  {
    v18 = v11;
    sub_B96E90((__int64)&v20, v12, 1);
    v13 = v18;
  }
  v21 = *(_DWORD *)(a2 + 72);
  v14 = sub_33FAF80(v10, 52, (__int64)&v20, v9, v19, v13, a3);
  if ( v20 )
    sub_B91220((__int64)&v20, v20);
  return v14;
}
