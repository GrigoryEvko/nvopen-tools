// Function: sub_38154A0
// Address: 0x38154a0
//
unsigned __int8 *__fastcall sub_38154A0(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 (__fastcall *v4)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v5; // rax
  unsigned __int16 v6; // si
  __int64 v7; // r8
  __int64 v8; // rax
  int v9; // r9d
  unsigned int v10; // r13d
  __int64 v11; // r15
  __int64 v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // rsi
  unsigned __int8 *v15; // r12
  __int64 v17; // rdx
  __int64 v18; // [rsp+0h] [rbp-50h] BYREF
  int v19; // [rsp+8h] [rbp-48h]
  __int64 v20; // [rsp+10h] [rbp-40h]

  v4 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v5 = *(__int16 **)(a2 + 48);
  v6 = *v5;
  v7 = *((_QWORD *)v5 + 1);
  v8 = a1[1];
  if ( v4 == sub_2D56A50 )
  {
    HIWORD(v10) = 0;
    sub_2FE6CC0((__int64)&v18, *a1, *(_QWORD *)(v8 + 64), v6, v7);
    LOWORD(v10) = v19;
    v11 = v20;
  }
  else
  {
    v10 = v4(*a1, *(_QWORD *)(v8 + 64), v6, v7);
    v11 = v17;
  }
  v12 = *(_QWORD *)(a2 + 80);
  v18 = v12;
  if ( v12 )
    sub_B96E90((__int64)&v18, v12, 1);
  v13 = a1[1];
  v14 = *(unsigned int *)(a2 + 24);
  v19 = *(_DWORD *)(a2 + 72);
  v15 = sub_33FAF80(v13, v14, (__int64)&v18, v10, v11, v9, a3);
  if ( v18 )
    sub_B91220((__int64)&v18, v18);
  return v15;
}
