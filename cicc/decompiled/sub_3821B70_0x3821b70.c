// Function: sub_3821B70
// Address: 0x3821b70
//
void __fastcall sub_3821B70(__int64 *a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rsi
  __int64 v8; // r9
  __int64 (__fastcall *v9)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v10; // rax
  unsigned __int16 v11; // si
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v16; // rsi
  unsigned int *v17; // rcx
  __int64 v18; // rdx
  __int64 v19; // r9
  unsigned __int8 *v20; // rax
  __int64 v21; // [rsp+0h] [rbp-60h] BYREF
  int v22; // [rsp+8h] [rbp-58h]
  _BYTE v23[8]; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int16 v24; // [rsp+18h] [rbp-48h]
  __int64 v25; // [rsp+20h] [rbp-40h]

  v7 = *(_QWORD *)(a2 + 80);
  v21 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v21, v7, 1);
  v8 = *a1;
  v22 = *(_DWORD *)(a2 + 72);
  v9 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)v8 + 592LL);
  v10 = *(__int16 **)(a2 + 48);
  v11 = *v10;
  v12 = *((_QWORD *)v10 + 1);
  v13 = a1[1];
  if ( v9 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)v23, v8, *(_QWORD *)(v13 + 64), v11, v12);
    v15 = v25;
    v16 = v24;
  }
  else
  {
    v16 = v9(v8, *(_QWORD *)(v13 + 64), v11, v12);
  }
  v17 = (unsigned int *)sub_33E5B50((_QWORD *)a1[1], v16, v15, (unsigned int)v16, v15, v14, 1, 0);
  v20 = sub_3411EF0(
          (_QWORD *)a1[1],
          *(unsigned int *)(a2 + 24),
          (__int64)&v21,
          v17,
          v18,
          v19,
          *(_OWORD *)*(_QWORD *)(a2 + 40));
  *(_QWORD *)a3 = v20;
  *(_DWORD *)(a3 + 8) = 0;
  *(_QWORD *)a4 = v20;
  *(_DWORD *)(a4 + 8) = 1;
  sub_3760E70((__int64)a1, a2, 1, (unsigned __int64)v20, 2);
  if ( v21 )
    sub_B91220((__int64)&v21, v21);
}
