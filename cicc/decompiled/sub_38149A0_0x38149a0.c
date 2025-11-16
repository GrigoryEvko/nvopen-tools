// Function: sub_38149A0
// Address: 0x38149a0
//
unsigned __int8 *__fastcall sub_38149A0(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 v4; // rsi
  __int64 v5; // r9
  __int64 (__fastcall *v6)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v7; // rax
  unsigned __int16 v8; // si
  __int64 v9; // r8
  __int64 v10; // rax
  __int64 v11; // r8
  __int64 v12; // rcx
  unsigned __int8 *v13; // r12
  __int64 v15; // rdx
  __int128 v16; // [rsp-10h] [rbp-60h]
  __int64 v17; // [rsp+0h] [rbp-50h] BYREF
  int v18; // [rsp+8h] [rbp-48h]
  _BYTE v19[8]; // [rsp+10h] [rbp-40h] BYREF
  unsigned __int16 v20; // [rsp+18h] [rbp-38h]
  __int64 v21; // [rsp+20h] [rbp-30h]

  v4 = *(_QWORD *)(a2 + 80);
  v17 = v4;
  if ( v4 )
    sub_B96E90((__int64)&v17, v4, 1);
  v5 = *a1;
  v18 = *(_DWORD *)(a2 + 72);
  v6 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v5 + 592LL);
  v7 = *(__int16 **)(a2 + 48);
  v8 = *v7;
  v9 = *((_QWORD *)v7 + 1);
  v10 = a1[1];
  if ( v6 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)v19, v5, *(_QWORD *)(v10 + 64), v8, v9);
    v11 = v21;
    v12 = v20;
  }
  else
  {
    v12 = v6(v5, *(_QWORD *)(v10 + 64), v8, v9);
    v11 = v15;
  }
  *((_QWORD *)&v16 + 1) = *(unsigned int *)(a2 + 64);
  *(_QWORD *)&v16 = *(_QWORD *)(a2 + 40);
  v13 = sub_34102A0(
          (_QWORD *)a1[1],
          *(unsigned int *)(a2 + 24),
          (__int64)&v17,
          v12,
          v11,
          *((__int64 *)&v16 + 1),
          a3,
          v16);
  if ( v17 )
    sub_B91220((__int64)&v17, v17);
  return v13;
}
