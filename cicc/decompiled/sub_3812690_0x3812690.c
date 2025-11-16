// Function: sub_3812690
// Address: 0x3812690
//
unsigned __int8 *__fastcall sub_3812690(__int64 *a1, __int64 a2, __m128i a3)
{
  unsigned __int64 *v5; // rax
  __int64 v6; // rsi
  unsigned __int64 v7; // r12
  __int64 v8; // r13
  __int64 v9; // rcx
  __int64 v10; // rax
  unsigned __int16 v11; // r14
  __int64 v12; // r10
  unsigned __int16 v13; // di
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 (__fastcall *v16)(__int64, __int64, unsigned int, __int64); // r11
  __int64 v17; // r8
  __int64 v18; // rcx
  int v19; // r9d
  __int64 v20; // rsi
  unsigned __int8 *v21; // r10
  __int64 v22; // rdx
  __int64 v23; // r9
  unsigned __int8 *v24; // r12
  __int64 v26; // rdx
  __int128 v27; // [rsp-30h] [rbp-C0h]
  __int16 v28; // [rsp+Eh] [rbp-82h]
  __int64 v29; // [rsp+10h] [rbp-80h]
  __int64 v30; // [rsp+10h] [rbp-80h]
  unsigned __int64 v31; // [rsp+18h] [rbp-78h]
  __int64 v32; // [rsp+18h] [rbp-78h]
  __int64 v33; // [rsp+30h] [rbp-60h] BYREF
  int v34; // [rsp+38h] [rbp-58h]
  char v35[8]; // [rsp+40h] [rbp-50h] BYREF
  unsigned __int16 v36; // [rsp+48h] [rbp-48h]
  __int64 v37; // [rsp+50h] [rbp-40h]

  v28 = **(_WORD **)(a2 + 48);
  v5 = *(unsigned __int64 **)(a2 + 40);
  v6 = *(_QWORD *)(a2 + 80);
  v7 = *v5;
  v8 = v5[1];
  v9 = 16LL * *((unsigned int *)v5 + 2);
  v31 = *v5;
  v10 = *(_QWORD *)(*v5 + 48) + v9;
  v29 = v9;
  v11 = *(_WORD *)v10;
  v33 = v6;
  if ( v6 )
  {
    sub_B96E90((__int64)&v33, v6, 1);
    v10 = *(_QWORD *)(v31 + 48) + v29;
  }
  v12 = *a1;
  v34 = *(_DWORD *)(a2 + 72);
  v13 = *(_WORD *)v10;
  v14 = *(_QWORD *)(v10 + 8);
  v15 = a1[1];
  v16 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v12 + 592LL);
  if ( v16 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)v35, v12, *(_QWORD *)(v15 + 64), v13, v14);
    v17 = v37;
    v18 = v36;
  }
  else
  {
    v18 = v16(v12, *(_QWORD *)(v15 + 64), v13, v14);
    v17 = v26;
  }
  v30 = v17;
  v32 = v18;
  sub_380F170((__int64)a1, v7, v8);
  if ( v11 == 11 )
  {
    v20 = 236;
  }
  else if ( v28 == 11 )
  {
    v20 = 237;
  }
  else if ( v11 == 10 )
  {
    v20 = 240;
  }
  else
  {
    if ( v28 != 10 )
      sub_C64ED0("Attempt at an invalid promotion-related conversion", 1u);
    v20 = 241;
  }
  v21 = sub_33FAF80(a1[1], v20, (__int64)&v33, v32, v30, v19, a3);
  *((_QWORD *)&v27 + 1) = v22;
  *(_QWORD *)&v27 = v21;
  v24 = sub_3406EB0(
          (_QWORD *)a1[1],
          *(_DWORD *)(a2 + 24),
          (__int64)&v33,
          **(unsigned __int16 **)(a2 + 48),
          *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
          v23,
          v27,
          *(_OWORD *)(*(_QWORD *)(a2 + 40) + 40LL));
  if ( v33 )
    sub_B91220((__int64)&v33, v33);
  return v24;
}
