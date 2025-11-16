// Function: sub_38156E0
// Address: 0x38156e0
//
__m128i *__fastcall sub_38156E0(__int64 *a1, unsigned __int64 a2)
{
  __int64 (__fastcall *v4)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v5; // rax
  unsigned __int16 v6; // si
  __int64 v7; // r8
  __int64 v8; // rax
  __int64 v9; // r15
  unsigned int v10; // ecx
  char v11; // r14
  __int64 v12; // rsi
  __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 *v15; // rdi
  __m128i *v16; // r14
  __int64 v18; // rdx
  unsigned int v19; // [rsp+8h] [rbp-58h]
  __int64 v20; // [rsp+10h] [rbp-50h] BYREF
  int v21; // [rsp+18h] [rbp-48h]
  __int64 v22; // [rsp+20h] [rbp-40h]

  v4 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v5 = *(__int16 **)(a2 + 48);
  v6 = *v5;
  v7 = *((_QWORD *)v5 + 1);
  v8 = a1[1];
  if ( v4 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v20, *a1, *(_QWORD *)(v8 + 64), v6, v7);
    v9 = v22;
    v10 = (unsigned __int16)v21;
  }
  else
  {
    v10 = v4(*a1, *(_QWORD *)(v8 + 64), v6, v7);
    v9 = v18;
  }
  if ( *(_DWORD *)(a2 + 24) != 298 || (v11 = 1, ((*(_BYTE *)(a2 + 33) >> 2) & 3) != 0) )
    v11 = (*(_BYTE *)(a2 + 33) >> 2) & 3;
  v12 = *(_QWORD *)(a2 + 80);
  v20 = v12;
  if ( v12 )
  {
    v19 = v10;
    sub_B96E90((__int64)&v20, v12, 1);
    v10 = v19;
  }
  v13 = *(_QWORD *)(a2 + 40);
  v14 = *(_QWORD *)(a2 + 104);
  v15 = (__int64 *)a1[1];
  v21 = *(_DWORD *)(a2 + 72);
  v16 = sub_33F1B30(
          v15,
          v11,
          (__int64)&v20,
          v10,
          v9,
          *(const __m128i **)(a2 + 112),
          *(_OWORD *)v13,
          *(_QWORD *)(v13 + 40),
          *(_QWORD *)(v13 + 48),
          *(unsigned __int16 *)(a2 + 96),
          v14);
  sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v16, 1);
  if ( v20 )
    sub_B91220((__int64)&v20, v20);
  return v16;
}
