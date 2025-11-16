// Function: sub_3785130
// Address: 0x3785130
//
unsigned __int8 *__fastcall sub_3785130(__int64 a1, __int64 a2, unsigned int a3, __m128i a4)
{
  unsigned __int16 *v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rcx
  unsigned int v9; // r14d
  unsigned __int64 *v10; // rax
  unsigned __int64 v11; // rsi
  __int64 v12; // rdx
  __int64 v13; // rax
  __int16 v14; // cx
  __int64 v15; // rax
  __int64 v16; // r13
  unsigned int v17; // eax
  unsigned __int8 *v18; // rax
  __int64 v19; // rdx
  unsigned __int8 *v20; // r14
  __int64 v22; // [rsp+0h] [rbp-A0h]
  __int64 v23; // [rsp+8h] [rbp-98h]
  __int128 v24; // [rsp+10h] [rbp-90h] BYREF
  __int128 v25; // [rsp+20h] [rbp-80h] BYREF
  __int64 v26; // [rsp+30h] [rbp-70h] BYREF
  int v27; // [rsp+38h] [rbp-68h]
  __int64 v28[2]; // [rsp+40h] [rbp-60h] BYREF
  _QWORD v29[10]; // [rsp+50h] [rbp-50h] BYREF

  v6 = *(unsigned __int16 **)(a2 + 48);
  v7 = *(_QWORD *)(a2 + 80);
  v8 = *((_QWORD *)v6 + 1);
  v9 = *v6;
  *(_QWORD *)&v25 = 0;
  *(_QWORD *)&v24 = 0;
  v23 = v8;
  DWORD2(v24) = 0;
  DWORD2(v25) = 0;
  v26 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v26, v7, 1);
  v27 = *(_DWORD *)(a2 + 72);
  v10 = (unsigned __int64 *)(*(_QWORD *)(a2 + 40) + 40LL * a3);
  v11 = *v10;
  v12 = v10[1];
  v13 = *(_QWORD *)(*v10 + 48) + 16LL * *((unsigned int *)v10 + 2);
  v14 = *(_WORD *)v13;
  v15 = *(_QWORD *)(v13 + 8);
  LOWORD(v28[0]) = v14;
  v28[1] = v15;
  sub_375E8D0(a1, v11, v12, (__int64)&v24, (__int64)&v25);
  sub_33D0340((__int64)v29, *(_QWORD *)(a1 + 8), v28);
  v16 = v29[1];
  v22 = v29[0];
  v17 = sub_33CB000(*(_DWORD *)(a2 + 24));
  v18 = sub_3405C90(*(_QWORD **)(a1 + 8), v17, (__int64)&v26, v22, v16, *(_DWORD *)(a2 + 28), a4, v24, v25);
  v20 = sub_33FA050(
          *(_QWORD *)(a1 + 8),
          *(unsigned int *)(a2 + 24),
          (__int64)&v26,
          v9,
          v23,
          *(_DWORD *)(a2 + 28),
          v18,
          v19);
  if ( v26 )
    sub_B91220((__int64)&v26, v26);
  return v20;
}
