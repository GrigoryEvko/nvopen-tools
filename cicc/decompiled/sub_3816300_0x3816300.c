// Function: sub_3816300
// Address: 0x3816300
//
unsigned __int8 *__fastcall sub_3816300(__int64 *a1, unsigned __int64 a2)
{
  __int64 (__fastcall *v3)(__int64, __int64, unsigned int, __int64); // r10
  __int64 v4; // rax
  unsigned __int16 v5; // si
  __int64 v6; // r8
  __int64 v7; // rax
  __int64 v8; // r8
  __int64 v9; // rcx
  __int64 v10; // rsi
  __int64 v11; // r10
  unsigned int v12; // ebx
  __int64 *v13; // r14
  unsigned int *v14; // rax
  __int64 v15; // rdx
  unsigned __int8 *v16; // r14
  __int64 v17; // rdx
  __int64 v19; // rdx
  __int64 v20; // [rsp+8h] [rbp-68h]
  __int64 v21; // [rsp+10h] [rbp-60h]
  __int64 v22; // [rsp+18h] [rbp-58h]
  __int128 *v23; // [rsp+18h] [rbp-58h]
  __int64 v24; // [rsp+20h] [rbp-50h] BYREF
  int v25; // [rsp+28h] [rbp-48h]
  __int64 v26; // [rsp+30h] [rbp-40h]

  v3 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v4 = *(_QWORD *)(a2 + 48);
  v5 = *(_WORD *)(v4 + 16);
  v6 = *(_QWORD *)(v4 + 24);
  v7 = a1[1];
  if ( v3 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v24, *a1, *(_QWORD *)(v7 + 64), v5, v6);
    v8 = v26;
    v9 = (unsigned __int16)v25;
  }
  else
  {
    v9 = v3(*a1, *(_QWORD *)(v7 + 64), v5, v6);
    v8 = v19;
  }
  v10 = *(_QWORD *)(a2 + 80);
  v11 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
  v12 = **(unsigned __int16 **)(a2 + 48);
  v24 = v10;
  if ( v10 )
  {
    v20 = v8;
    v21 = v9;
    v22 = v11;
    sub_B96E90((__int64)&v24, v10, 1);
    v8 = v20;
    v9 = v21;
    v11 = v22;
  }
  v13 = (__int64 *)a1[1];
  v23 = *(__int128 **)(a2 + 40);
  v25 = *(_DWORD *)(a2 + 72);
  v14 = (unsigned int *)sub_33E5110(v13, v12, v11, v9, v8);
  v16 = sub_3411EF0(v13, *(unsigned int *)(a2 + 24), (__int64)&v24, v14, v15, (__int64)v23, *v23);
  sub_3760E70((__int64)a1, a2, 0, (unsigned __int64)v16, v17);
  if ( v24 )
    sub_B91220((__int64)&v24, v24);
  return v16;
}
