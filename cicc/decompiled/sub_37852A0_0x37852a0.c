// Function: sub_37852A0
// Address: 0x37852a0
//
unsigned __int8 *__fastcall sub_37852A0(__int64 a1, __int64 a2, __m128i a3)
{
  unsigned __int16 *v4; // rax
  __int64 v5; // rsi
  __int64 v6; // rcx
  unsigned int v7; // r14d
  int v8; // r15d
  __int64 v9; // rax
  unsigned __int64 v10; // rsi
  __int64 v11; // rdx
  __int64 v12; // rax
  __int16 v13; // cx
  __int64 v14; // rax
  __int128 v15; // rax
  unsigned __int8 *v16; // r14
  __int128 v18; // [rsp+0h] [rbp-B0h]
  __int64 v19; // [rsp+18h] [rbp-98h]
  __int128 v20; // [rsp+20h] [rbp-90h] BYREF
  __int128 v21; // [rsp+30h] [rbp-80h] BYREF
  __int64 v22; // [rsp+40h] [rbp-70h] BYREF
  int v23; // [rsp+48h] [rbp-68h]
  __int64 v24[2]; // [rsp+50h] [rbp-60h] BYREF
  _BYTE v25[80]; // [rsp+60h] [rbp-50h] BYREF

  v4 = *(unsigned __int16 **)(a2 + 48);
  v5 = *(_QWORD *)(a2 + 80);
  v6 = *((_QWORD *)v4 + 1);
  v7 = *v4;
  *(_QWORD *)&v21 = 0;
  *(_QWORD *)&v20 = 0;
  v19 = v6;
  DWORD2(v20) = 0;
  DWORD2(v21) = 0;
  v22 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v22, v5, 1);
  v8 = *(_DWORD *)(a2 + 28);
  v23 = *(_DWORD *)(a2 + 72);
  v9 = *(_QWORD *)(a2 + 40);
  v10 = *(_QWORD *)(v9 + 40);
  v11 = *(_QWORD *)(v9 + 48);
  v18 = *(_OWORD *)v9;
  v12 = *(_QWORD *)(v10 + 48) + 16LL * *(unsigned int *)(v9 + 48);
  v13 = *(_WORD *)v12;
  v14 = *(_QWORD *)(v12 + 8);
  LOWORD(v24[0]) = v13;
  v24[1] = v14;
  sub_375E8D0(a1, v10, v11, (__int64)&v20, (__int64)&v21);
  sub_33D0340((__int64)v25, *(_QWORD *)(a1 + 8), v24);
  *(_QWORD *)&v15 = sub_3405C90(*(_QWORD **)(a1 + 8), *(_DWORD *)(a2 + 24), (__int64)&v22, v7, v19, v8, a3, v18, v20);
  v16 = sub_3405C90(*(_QWORD **)(a1 + 8), *(_DWORD *)(a2 + 24), (__int64)&v22, v7, v19, v8, a3, v15, v21);
  if ( v22 )
    sub_B91220((__int64)&v22, v22);
  return v16;
}
