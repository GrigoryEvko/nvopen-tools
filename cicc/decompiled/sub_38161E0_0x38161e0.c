// Function: sub_38161E0
// Address: 0x38161e0
//
unsigned __int8 *__fastcall sub_38161E0(__int64 *a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 *v3; // r9
  __int64 v4; // rsi
  __int64 v5; // r10
  __int64 *v6; // rax
  __int64 v7; // r12
  __int64 v8; // r13
  __int64 v9; // r14
  __int64 v10; // r15
  __int64 (__fastcall *v11)(__int64, __int64, unsigned int, __int64); // r11
  __int16 *v12; // rax
  unsigned __int16 v13; // di
  __int64 v14; // r8
  __int64 v15; // rax
  __int64 v16; // r8
  __int64 v17; // rcx
  __int64 v18; // r9
  unsigned __int8 *v19; // r12
  __int64 v21; // rax
  __int64 v22; // rdx
  __int128 v23; // [rsp-20h] [rbp-90h]
  __int128 v24; // [rsp-10h] [rbp-80h]
  __int64 v25; // [rsp+0h] [rbp-70h]
  __int64 *v26; // [rsp+8h] [rbp-68h]
  __int64 v27; // [rsp+10h] [rbp-60h] BYREF
  int v28; // [rsp+18h] [rbp-58h]
  _BYTE v29[8]; // [rsp+20h] [rbp-50h] BYREF
  unsigned __int16 v30; // [rsp+28h] [rbp-48h]
  __int64 v31; // [rsp+30h] [rbp-40h]

  v2 = a2;
  v3 = a1;
  v4 = *(_QWORD *)(a2 + 80);
  v27 = v4;
  if ( v4 )
  {
    v25 = v2;
    sub_B96E90((__int64)&v27, v4, 1);
    v2 = v25;
    v3 = a1;
  }
  v5 = *v3;
  v26 = v3;
  v28 = *(_DWORD *)(v2 + 72);
  v6 = *(__int64 **)(v2 + 40);
  v7 = *v6;
  v8 = v6[1];
  v9 = v6[5];
  v10 = v6[6];
  v11 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v5 + 592LL);
  v12 = *(__int16 **)(v2 + 48);
  v13 = *v12;
  v14 = *((_QWORD *)v12 + 1);
  v15 = v3[1];
  if ( v11 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)v29, v5, *(_QWORD *)(v15 + 64), v13, v14);
    v16 = v31;
    v17 = v30;
    v18 = (__int64)v26;
  }
  else
  {
    v21 = v11(v5, *(_QWORD *)(v15 + 64), v13, v14);
    v18 = (__int64)v26;
    v17 = v21;
    v16 = v22;
  }
  *((_QWORD *)&v24 + 1) = v10;
  *(_QWORD *)&v24 = v9;
  *((_QWORD *)&v23 + 1) = v8;
  *(_QWORD *)&v23 = v7;
  v19 = sub_3406EB0(*(_QWORD **)(v18 + 8), 0x9Bu, (__int64)&v27, v17, v16, v18, v23, v24);
  if ( v27 )
    sub_B91220((__int64)&v27, v27);
  return v19;
}
