// Function: sub_37836C0
// Address: 0x37836c0
//
void __fastcall sub_37836C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5)
{
  unsigned __int64 *v9; // rax
  __int64 v10; // rdx
  int v11; // r9d
  __int64 v12; // rsi
  __int64 v13; // rdi
  unsigned __int16 *v14; // rax
  unsigned __int8 *v15; // rax
  int v16; // edx
  __int64 v17; // rdx
  unsigned __int16 *v18; // rax
  int v19; // r9d
  unsigned __int8 *v20; // rax
  __int64 v21; // rsi
  int v22; // edx
  int v23; // [rsp+18h] [rbp-68h]
  __int64 v24; // [rsp+20h] [rbp-60h] BYREF
  __int64 v25; // [rsp+28h] [rbp-58h]
  __int64 v26; // [rsp+30h] [rbp-50h] BYREF
  __int64 v27; // [rsp+38h] [rbp-48h]
  __int64 v28; // [rsp+40h] [rbp-40h] BYREF
  int v29; // [rsp+48h] [rbp-38h]

  v9 = *(unsigned __int64 **)(a2 + 40);
  LODWORD(v25) = 0;
  LODWORD(v27) = 0;
  v24 = 0;
  v10 = v9[1];
  v26 = 0;
  sub_375E8D0(a1, *v9, v10, (__int64)&v24, (__int64)&v26);
  v12 = *(_QWORD *)(a2 + 80);
  v28 = v12;
  if ( v12 )
    sub_B96E90((__int64)&v28, v12, 1);
  v13 = *(_QWORD *)(a1 + 8);
  v29 = *(_DWORD *)(a2 + 72);
  v14 = (unsigned __int16 *)(*(_QWORD *)(v26 + 48) + 16LL * (unsigned int)v27);
  v15 = sub_33FAF80(v13, 164, (__int64)&v28, *v14, *((_QWORD *)v14 + 1), v11, a5);
  v23 = v16;
  v17 = v24;
  *(_QWORD *)a3 = v15;
  *(_DWORD *)(a3 + 8) = v23;
  v18 = (unsigned __int16 *)(*(_QWORD *)(v17 + 48) + 16LL * (unsigned int)v25);
  v20 = sub_33FAF80(*(_QWORD *)(a1 + 8), 164, (__int64)&v28, *v18, *((_QWORD *)v18 + 1), v19, a5);
  v21 = v28;
  *(_QWORD *)a4 = v20;
  *(_DWORD *)(a4 + 8) = v22;
  if ( v21 )
    sub_B91220((__int64)&v28, v21);
}
