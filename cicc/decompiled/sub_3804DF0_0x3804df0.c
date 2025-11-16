// Function: sub_3804DF0
// Address: 0x3804df0
//
unsigned __int8 *__fastcall sub_3804DF0(__int64 *a1, __int64 a2, __m128i a3)
{
  unsigned int v3; // r15d
  __int64 v6; // rsi
  __int64 v7; // r9
  __int64 v8; // rdx
  __int16 *v9; // rax
  unsigned __int16 v10; // di
  __int64 v11; // r10
  __int64 (__fastcall *v12)(__int64, __int64, unsigned int, __int64); // rax
  unsigned int v13; // r13d
  __int64 v14; // r9
  __int64 v15; // r10
  __int128 v16; // rax
  _QWORD *v17; // rbx
  __int128 v18; // rax
  __int64 v19; // r9
  unsigned __int8 *v20; // r14
  unsigned int v22; // eax
  __int64 v23; // rdx
  __int64 v24; // [rsp+8h] [rbp-88h]
  __int64 v25; // [rsp+10h] [rbp-80h]
  __int128 v26; // [rsp+10h] [rbp-80h]
  __int64 v27; // [rsp+10h] [rbp-80h]
  __int64 v28; // [rsp+20h] [rbp-70h]
  unsigned __int16 v29; // [rsp+2Eh] [rbp-62h]
  __int64 v30; // [rsp+30h] [rbp-60h] BYREF
  int v31; // [rsp+38h] [rbp-58h]
  _BYTE v32[8]; // [rsp+40h] [rbp-50h] BYREF
  __int16 v33; // [rsp+48h] [rbp-48h]
  __int64 v34; // [rsp+50h] [rbp-40h]

  v6 = *(_QWORD *)(a2 + 80);
  v30 = v6;
  if ( v6 )
    sub_B96E90((__int64)&v30, v6, 1);
  v7 = *a1;
  v8 = a1[1];
  v31 = *(_DWORD *)(a2 + 72);
  v9 = *(__int16 **)(a2 + 48);
  v10 = *v9;
  v11 = *((_QWORD *)v9 + 1);
  v29 = *v9;
  v12 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v7 + 592LL);
  if ( v12 == sub_2D56A50 )
  {
    v25 = v11;
    HIWORD(v13) = 0;
    sub_2FE6CC0((__int64)v32, v7, *(_QWORD *)(v8 + 64), v10, v11);
    LOWORD(v13) = v33;
    v14 = v34;
    v15 = v25;
  }
  else
  {
    v3 = v29;
    v27 = v11;
    v22 = v12(v7, *(_QWORD *)(v8 + 64), v29, v11);
    v15 = v27;
    v13 = v22;
    v14 = v23;
  }
  v28 = v14;
  v24 = v15;
  *(_QWORD *)&v16 = sub_33FAF80(a1[1], *(unsigned int *)(a2 + 24), (__int64)&v30, v13, v14, v14, a3);
  v17 = (_QWORD *)a1[1];
  v26 = v16;
  *(_QWORD *)&v18 = sub_3400D50((__int64)v17, 0, (__int64)&v30, 1u, a3);
  LOWORD(v3) = v29;
  sub_3406EB0(v17, 0xE6u, (__int64)&v30, v3, v24, v19, v26, v18);
  v20 = sub_33FAF80((__int64)v17, 233, (__int64)&v30, v13, v28, v28, a3);
  if ( v30 )
    sub_B91220((__int64)&v30, v30);
  return v20;
}
