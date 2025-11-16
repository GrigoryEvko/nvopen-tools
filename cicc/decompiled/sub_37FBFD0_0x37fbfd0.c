// Function: sub_37FBFD0
// Address: 0x37fbfd0
//
unsigned __int8 *__fastcall sub_37FBFD0(__int64 *a1, __int64 a2, __m128i a3)
{
  _QWORD *v4; // r13
  unsigned __int8 *v5; // r14
  __int64 v6; // rdx
  __int64 v7; // r15
  __int128 v8; // rax
  __int64 v9; // r11
  __int64 (__fastcall *v10)(__int64, __int64, unsigned int, __int64); // r10
  unsigned __int16 v11; // dx
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // r9
  __int64 v15; // r8
  __int64 v16; // rcx
  __int64 v17; // rsi
  unsigned __int8 *v18; // r12
  __int64 v20; // rdx
  __int128 v21; // [rsp-10h] [rbp-80h]
  __int64 v22; // [rsp+0h] [rbp-70h]
  __int64 v23; // [rsp+8h] [rbp-68h]
  __int128 v24; // [rsp+10h] [rbp-60h]
  __int64 v25; // [rsp+20h] [rbp-50h] BYREF
  int v26; // [rsp+28h] [rbp-48h]
  __int64 v27; // [rsp+30h] [rbp-40h]

  v4 = (_QWORD *)a1[1];
  v5 = sub_375A6A0((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL), a3);
  v7 = v6;
  *(_QWORD *)&v8 = sub_375A6A0((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), a3);
  v9 = *a1;
  v24 = v8;
  v10 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  *(_QWORD *)&v8 = *(_QWORD *)(a2 + 48);
  v11 = *(_WORD *)v8;
  v12 = *(_QWORD *)(v8 + 8);
  v13 = a1[1];
  if ( v10 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v25, v9, *(_QWORD *)(v13 + 64), v11, v12);
    v15 = v27;
    v16 = (unsigned __int16)v26;
  }
  else
  {
    v16 = v10(v9, *(_QWORD *)(v13 + 64), v11, v12);
    v15 = v20;
  }
  v17 = *(_QWORD *)(a2 + 80);
  v25 = v17;
  if ( v17 )
  {
    v22 = v15;
    v23 = v16;
    sub_B96E90((__int64)&v25, v17, 1);
    v15 = v22;
    v16 = v23;
  }
  *((_QWORD *)&v21 + 1) = v7;
  *(_QWORD *)&v21 = v5;
  v26 = *(_DWORD *)(a2 + 72);
  v18 = sub_3406EB0(v4, 0x36u, (__int64)&v25, v16, v15, v14, v24, v21);
  if ( v25 )
    sub_B91220((__int64)&v25, v25);
  return v18;
}
