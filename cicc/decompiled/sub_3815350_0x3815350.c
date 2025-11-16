// Function: sub_3815350
// Address: 0x3815350
//
unsigned __int8 *__fastcall sub_3815350(__int64 *a1, unsigned __int64 a2)
{
  __int64 (__fastcall *v3)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v4; // rax
  unsigned __int16 v5; // si
  __int64 v6; // r8
  __int64 v7; // rax
  unsigned int v8; // ebx
  __int64 v9; // r10
  __int64 v10; // rsi
  __int64 *v11; // r14
  unsigned int *v12; // rax
  __int64 v13; // rdx
  unsigned __int8 *v14; // r14
  __int64 v16; // rdx
  __int64 v17; // [rsp+8h] [rbp-58h]
  __int128 *v18; // [rsp+8h] [rbp-58h]
  __int64 v19; // [rsp+10h] [rbp-50h] BYREF
  int v20; // [rsp+18h] [rbp-48h]
  __int64 v21; // [rsp+20h] [rbp-40h]

  v3 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v4 = *(__int16 **)(a2 + 48);
  v5 = *v4;
  v6 = *((_QWORD *)v4 + 1);
  v7 = a1[1];
  if ( v3 == sub_2D56A50 )
  {
    HIWORD(v8) = 0;
    sub_2FE6CC0((__int64)&v19, *a1, *(_QWORD *)(v7 + 64), v5, v6);
    LOWORD(v8) = v20;
    v9 = v21;
  }
  else
  {
    v8 = v3(*a1, *(_QWORD *)(v7 + 64), v5, v6);
    v9 = v16;
  }
  v10 = *(_QWORD *)(a2 + 80);
  v19 = v10;
  if ( v10 )
  {
    v17 = v9;
    sub_B96E90((__int64)&v19, v10, 1);
    v9 = v17;
  }
  v11 = (__int64 *)a1[1];
  v18 = *(__int128 **)(a2 + 40);
  v20 = *(_DWORD *)(a2 + 72);
  v12 = (unsigned int *)sub_33E5110(v11, v8, v9, 1, 0);
  v14 = sub_3411F20(
          v11,
          *(unsigned int *)(a2 + 24),
          (__int64)&v19,
          v12,
          v13,
          (__int64)v18,
          *v18,
          *(__int128 *)((char *)v18 + 40));
  sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v14, 1);
  if ( v19 )
    sub_B91220((__int64)&v19, v19);
  return v14;
}
