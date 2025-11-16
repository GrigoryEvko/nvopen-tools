// Function: sub_37FCAF0
// Address: 0x37fcaf0
//
unsigned __int8 *__fastcall sub_37FCAF0(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 v3; // r11
  __int64 v4; // rdx
  __int64 (__fastcall *v5)(__int64, __int64, unsigned int, __int64); // rax
  int v6; // r9d
  unsigned int v7; // r14d
  __int64 v8; // r10
  __int64 v9; // rsi
  __int64 v10; // r13
  int v11; // r9d
  unsigned __int8 *v12; // r12
  unsigned int v13; // edx
  _QWORD *v14; // rbx
  unsigned __int64 v15; // r13
  __int128 v16; // rax
  __int64 v17; // r9
  unsigned __int8 *v18; // r12
  unsigned int v20; // eax
  __int64 v21; // rdx
  __int128 v22; // [rsp-20h] [rbp-90h]
  __int64 v23; // [rsp+0h] [rbp-70h]
  __int64 v24; // [rsp+0h] [rbp-70h]
  __int64 v25; // [rsp+8h] [rbp-68h]
  __int64 v26; // [rsp+20h] [rbp-50h] BYREF
  int v27; // [rsp+28h] [rbp-48h]
  __int64 v28; // [rsp+30h] [rbp-40h]

  v3 = *a1;
  v4 = a1[1];
  v5 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v5 == sub_2D56A50 )
  {
    HIWORD(v7) = 0;
    sub_2FE6CC0((__int64)&v26, v3, *(_QWORD *)(v4 + 64), 12, 0);
    LOWORD(v7) = v27;
    v8 = a2;
    v25 = v28;
  }
  else
  {
    v20 = v5(v3, *(_QWORD *)(v4 + 64), 12u, 0);
    v8 = a2;
    v25 = v21;
    v7 = v20;
  }
  v9 = *(_QWORD *)(v8 + 80);
  v10 = *(_QWORD *)(*(_QWORD *)(v8 + 40) + 8LL);
  v26 = v9;
  if ( v9 )
  {
    v23 = v8;
    sub_B96E90((__int64)&v26, v9, 1);
    v8 = v23;
  }
  v24 = a1[1];
  v27 = *(_DWORD *)(v8 + 72);
  sub_33FAF80(v24, 234, (__int64)&v26, 6, 0, v6, a3);
  v12 = sub_33FAF80(v24, 215, (__int64)&v26, v7, v25, v11, a3);
  v14 = (_QWORD *)a1[1];
  v15 = v13 | v10 & 0xFFFFFFFF00000000LL;
  *(_QWORD *)&v16 = sub_3400E40((__int64)v14, 16, v7, v25, (__int64)&v26, a3);
  *((_QWORD *)&v22 + 1) = v15;
  *(_QWORD *)&v22 = v12;
  v18 = sub_3406EB0(v14, 0xBEu, (__int64)&v26, v7, v25, v17, v22, v16);
  if ( v26 )
    sub_B91220((__int64)&v26, v26);
  return v18;
}
