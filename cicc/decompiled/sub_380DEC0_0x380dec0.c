// Function: sub_380DEC0
// Address: 0x380dec0
//
unsigned __int8 *__fastcall sub_380DEC0(__int64 *a1, __int64 a2)
{
  __int16 *v3; // rax
  __int64 v4; // r9
  unsigned __int16 v5; // si
  __int64 v6; // r8
  __int64 (__fastcall *v7)(__int64, __int64, unsigned int, __int64); // rax
  unsigned int v8; // ebx
  __int64 v9; // r10
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // r12
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r13
  _QWORD *v16; // r10
  __int64 v17; // r8
  __int64 v18; // r9
  unsigned int v19; // esi
  unsigned __int8 *v20; // r12
  unsigned int v22; // eax
  __int64 v23; // rdx
  __int128 v24; // [rsp-20h] [rbp-90h]
  __int128 v25; // [rsp-10h] [rbp-80h]
  __int64 v26; // [rsp+0h] [rbp-70h]
  __int64 v27; // [rsp+8h] [rbp-68h]
  __int64 v28; // [rsp+10h] [rbp-60h]
  _QWORD *v29; // [rsp+10h] [rbp-60h]
  __int64 v30; // [rsp+18h] [rbp-58h]
  __int64 v31; // [rsp+20h] [rbp-50h] BYREF
  int v32; // [rsp+28h] [rbp-48h]
  __int64 v33; // [rsp+30h] [rbp-40h]

  v3 = *(__int16 **)(a2 + 48);
  v4 = *a1;
  v5 = *v3;
  v6 = *((_QWORD *)v3 + 1);
  v7 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v7 == sub_2D56A50 )
  {
    HIWORD(v8) = 0;
    sub_2FE6CC0((__int64)&v31, v4, *(_QWORD *)(a1[1] + 64), v5, v6);
    LOWORD(v8) = v32;
    v9 = (__int64)a1;
    v30 = v33;
  }
  else
  {
    v22 = v7(v4, *(_QWORD *)(a1[1] + 64), v5, v6);
    v9 = (__int64)a1;
    v30 = v23;
    v8 = v22;
  }
  v28 = v9;
  v10 = sub_380AAE0(v9, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
  v11 = *(_QWORD *)(a2 + 80);
  v12 = v10;
  v13 = *(_QWORD *)(a2 + 40);
  v15 = v14;
  v16 = *(_QWORD **)(v28 + 8);
  v17 = *(_QWORD *)(v13 + 40);
  v18 = *(_QWORD *)(v13 + 48);
  v31 = v11;
  if ( v11 )
  {
    v26 = v17;
    v27 = v18;
    v29 = v16;
    sub_B96E90((__int64)&v31, v11, 1);
    v17 = v26;
    v18 = v27;
    v16 = v29;
  }
  *((_QWORD *)&v25 + 1) = v18;
  *(_QWORD *)&v25 = v17;
  v19 = *(_DWORD *)(a2 + 24);
  *((_QWORD *)&v24 + 1) = v15;
  *(_QWORD *)&v24 = v12;
  v32 = *(_DWORD *)(a2 + 72);
  v20 = sub_3406EB0(v16, v19, (__int64)&v31, v8, v30, v18, v24, v25);
  if ( v31 )
    sub_B91220((__int64)&v31, v31);
  return v20;
}
