// Function: sub_380DBE0
// Address: 0x380dbe0
//
unsigned __int8 *__fastcall sub_380DBE0(__int64 *a1, __int64 a2, __m128i a3)
{
  __int16 *v4; // rax
  unsigned __int16 v5; // si
  __int64 v6; // r8
  __int64 (__fastcall *v7)(__int64, __int64, unsigned int, __int64); // rax
  __int64 v8; // r8
  __int64 v9; // rcx
  __int64 v10; // rdx
  __int64 v11; // rax
  _QWORD *v12; // r11
  int v13; // r9d
  __int64 v14; // rcx
  __int64 v15; // r14
  __int64 v16; // rdx
  __int64 v17; // r15
  __int64 v18; // r8
  unsigned int v19; // esi
  unsigned __int8 *v20; // r12
  __int64 v22; // rdx
  __int128 v23; // [rsp-10h] [rbp-90h]
  __int64 v24; // [rsp+0h] [rbp-80h]
  __int64 v25; // [rsp+8h] [rbp-78h]
  __int64 v26; // [rsp+10h] [rbp-70h]
  int v27; // [rsp+10h] [rbp-70h]
  __int64 v28; // [rsp+18h] [rbp-68h]
  _QWORD *v29; // [rsp+18h] [rbp-68h]
  __int128 v30; // [rsp+20h] [rbp-60h]
  __int64 v31; // [rsp+30h] [rbp-50h] BYREF
  int v32; // [rsp+38h] [rbp-48h]
  __int64 v33; // [rsp+40h] [rbp-40h]

  v4 = *(__int16 **)(a2 + 48);
  v5 = *v4;
  v6 = *((_QWORD *)v4 + 1);
  v7 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v7 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v31, *a1, *(_QWORD *)(a1[1] + 64), v5, v6);
    v8 = v33;
    v9 = (unsigned __int16)v32;
  }
  else
  {
    v9 = v7(*a1, *(_QWORD *)(a1[1] + 64), v5, v6);
    v8 = v22;
  }
  v26 = v8;
  v28 = v9;
  *(_QWORD *)&v30 = sub_380AAE0((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  *((_QWORD *)&v30 + 1) = v10;
  v11 = sub_380AAE0((__int64)a1, *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  v12 = (_QWORD *)a1[1];
  v13 = *(_DWORD *)(a2 + 28);
  v14 = v28;
  v15 = v11;
  v17 = v16;
  v31 = *(_QWORD *)(a2 + 80);
  v18 = v26;
  if ( v31 )
  {
    v24 = v26;
    v25 = v28;
    v27 = v13;
    v29 = v12;
    sub_B96E90((__int64)&v31, v31, 1);
    v18 = v24;
    v14 = v25;
    v13 = v27;
    v12 = v29;
  }
  v19 = *(_DWORD *)(a2 + 24);
  *((_QWORD *)&v23 + 1) = v17;
  *(_QWORD *)&v23 = v15;
  v32 = *(_DWORD *)(a2 + 72);
  v20 = sub_3405C90(v12, v19, (__int64)&v31, v14, v18, v13, a3, v30, v23);
  if ( v31 )
    sub_B91220((__int64)&v31, v31);
  return v20;
}
