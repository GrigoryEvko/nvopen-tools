// Function: sub_377DAA0
// Address: 0x377daa0
//
void __fastcall sub_377DAA0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5)
{
  __int64 v7; // rsi
  __int64 v8; // rsi
  __int16 *v9; // rax
  __int16 v10; // dx
  __int64 v11; // rax
  int v12; // r9d
  unsigned __int8 *v13; // rax
  int v14; // edx
  _QWORD *v15; // rdi
  _QWORD *v16; // rbx
  int v17; // edx
  int v18; // r12d
  __int64 v19; // [rsp+8h] [rbp-A8h]
  unsigned __int16 v20; // [rsp+16h] [rbp-9Ah]
  __int64 v22; // [rsp+40h] [rbp-70h] BYREF
  int v23; // [rsp+48h] [rbp-68h]
  __int64 v24[2]; // [rsp+50h] [rbp-60h] BYREF
  __int64 v25; // [rsp+60h] [rbp-50h] BYREF
  __int64 v26; // [rsp+68h] [rbp-48h]
  unsigned __int16 v27; // [rsp+70h] [rbp-40h]
  __int64 v28; // [rsp+78h] [rbp-38h]

  v7 = *(_QWORD *)(a2 + 80);
  v22 = v7;
  if ( v7 )
    sub_B96E90((__int64)&v22, v7, 1);
  v8 = *(_QWORD *)(a1 + 8);
  v23 = *(_DWORD *)(a2 + 72);
  v9 = *(__int16 **)(a2 + 48);
  v10 = *v9;
  v11 = *((_QWORD *)v9 + 1);
  LOWORD(v24[0]) = v10;
  v24[1] = v11;
  sub_33D0340((__int64)&v25, v8, v24);
  v20 = v27;
  v19 = v28;
  v13 = sub_33FAF80(*(_QWORD *)(a1 + 8), *(unsigned int *)(a2 + 24), (__int64)&v22, v25, v26, v12, a5);
  *(_DWORD *)(a3 + 8) = v14;
  *(_QWORD *)a3 = v13;
  if ( *(_DWORD *)(a2 + 24) == 167 )
  {
    v15 = *(_QWORD **)(a1 + 8);
    v25 = 0;
    LODWORD(v26) = 0;
    v16 = sub_33F17F0(v15, 51, (__int64)&v25, v20, v19);
    v18 = v17;
    if ( v25 )
      sub_B91220((__int64)&v25, v25);
    *(_QWORD *)a4 = v16;
    *(_DWORD *)(a4 + 8) = v18;
  }
  else
  {
    *(_QWORD *)a4 = v13;
    *(_DWORD *)(a4 + 8) = *(_DWORD *)(a3 + 8);
  }
  if ( v22 )
    sub_B91220((__int64)&v22, v22);
}
