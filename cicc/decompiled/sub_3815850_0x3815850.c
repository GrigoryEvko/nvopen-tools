// Function: sub_3815850
// Address: 0x3815850
//
__m128i *__fastcall sub_3815850(__int64 *a1, unsigned __int64 a2)
{
  __int64 (__fastcall *v4)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v5; // rax
  unsigned __int16 v6; // si
  __int64 v7; // r8
  __int64 v8; // rax
  unsigned int v9; // ebx
  __int64 v10; // r8
  __int64 v11; // rsi
  char v12; // r13
  int v13; // r13d
  __int64 v14; // rcx
  __int64 v15; // rdx
  unsigned __int16 v16; // si
  __int64 *v17; // rdi
  __m128i *v18; // r13
  __int64 v20; // rdx
  const __m128i *v21; // [rsp-10h] [rbp-70h]
  __int64 v22; // [rsp+8h] [rbp-58h]
  __int64 v23; // [rsp+10h] [rbp-50h] BYREF
  int v24; // [rsp+18h] [rbp-48h]
  __int64 v25; // [rsp+20h] [rbp-40h]

  v4 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v5 = *(__int16 **)(a2 + 48);
  v6 = *v5;
  v7 = *((_QWORD *)v5 + 1);
  v8 = a1[1];
  if ( v4 == sub_2D56A50 )
  {
    HIWORD(v9) = 0;
    sub_2FE6CC0((__int64)&v23, *a1, *(_QWORD *)(v8 + 64), v6, v7);
    LOWORD(v9) = v24;
    v10 = v25;
  }
  else
  {
    v9 = v4(*a1, *(_QWORD *)(v8 + 64), v6, v7);
    v10 = v20;
  }
  v11 = *(_QWORD *)(a2 + 80);
  v12 = *(_BYTE *)(a2 + 33) >> 2;
  v23 = v11;
  v13 = v12 & 3;
  if ( !v13 )
    LOBYTE(v13) = 1;
  if ( v11 )
  {
    v22 = v10;
    sub_B96E90((__int64)&v23, v11, 1);
    v10 = v22;
  }
  v14 = *(_QWORD *)(a2 + 40);
  v15 = *(_QWORD *)(a2 + 104);
  v16 = *(_WORD *)(a2 + 32);
  v21 = *(const __m128i **)(a2 + 112);
  v17 = (__int64 *)a1[1];
  v24 = *(_DWORD *)(a2 + 72);
  v18 = sub_33E9660(
          v17,
          (v16 >> 7) & 7,
          v13,
          v9,
          v10,
          (__int64)&v23,
          *(_OWORD *)v14,
          *(_QWORD *)(v14 + 40),
          *(_QWORD *)(v14 + 48),
          *(_OWORD *)(v14 + 80),
          *(_OWORD *)(v14 + 120),
          *(_OWORD *)(v14 + 160),
          *(unsigned __int16 *)(a2 + 96),
          v15,
          v21,
          0);
  sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v18, 1);
  if ( v23 )
    sub_B91220((__int64)&v23, v23);
  return v18;
}
