// Function: sub_3778790
// Address: 0x3778790
//
void __fastcall sub_3778790(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 *v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // rsi
  unsigned int v11; // esi
  _QWORD *v12; // rdi
  __int64 v13; // rax
  __int64 v14; // r14
  __int64 v15; // r15
  unsigned __int16 *v16; // rax
  __int64 v17; // rax
  int v18; // edx
  __int64 v19; // rdx
  unsigned __int16 *v20; // rax
  __int64 v21; // rax
  __int64 v22; // rsi
  int v23; // edx
  __int128 v24; // [rsp-10h] [rbp-D0h]
  __int128 v25; // [rsp-10h] [rbp-D0h]
  int v28; // [rsp+38h] [rbp-88h]
  __int64 v29; // [rsp+40h] [rbp-80h] BYREF
  __int64 v30; // [rsp+48h] [rbp-78h]
  __int64 v31; // [rsp+50h] [rbp-70h] BYREF
  __int64 v32; // [rsp+58h] [rbp-68h]
  __int128 v33; // [rsp+60h] [rbp-60h] BYREF
  __int128 v34; // [rsp+70h] [rbp-50h] BYREF
  __int64 v35; // [rsp+80h] [rbp-40h] BYREF
  int v36; // [rsp+88h] [rbp-38h]

  v6 = *(unsigned __int64 **)(a2 + 40);
  LODWORD(v30) = 0;
  LODWORD(v32) = 0;
  v7 = v6[1];
  v29 = 0;
  v31 = 0;
  sub_375E8D0(a1, *v6, v7, (__int64)&v29, (__int64)&v31);
  v8 = *(_QWORD *)(a2 + 40);
  DWORD2(v33) = 0;
  DWORD2(v34) = 0;
  v9 = *(_QWORD *)(v8 + 48);
  *(_QWORD *)&v33 = 0;
  *(_QWORD *)&v34 = 0;
  sub_375E8D0(a1, *(_QWORD *)(v8 + 40), v9, (__int64)&v33, (__int64)&v34);
  v10 = *(_QWORD *)(a2 + 80);
  v35 = v10;
  if ( v10 )
    sub_B96E90((__int64)&v35, v10, 1);
  v11 = *(_DWORD *)(a2 + 24);
  v12 = *(_QWORD **)(a1 + 8);
  v36 = *(_DWORD *)(a2 + 72);
  v13 = *(_QWORD *)(a2 + 40);
  v14 = *(_QWORD *)(v13 + 80);
  v15 = *(_QWORD *)(v13 + 88);
  v16 = (unsigned __int16 *)(*(_QWORD *)(v29 + 48) + 16LL * (unsigned int)v30);
  *((_QWORD *)&v24 + 1) = v15;
  *(_QWORD *)&v24 = v14;
  v17 = sub_340EC60(v12, v11, (__int64)&v35, *v16, *((_QWORD *)v16 + 1), *(unsigned int *)(a2 + 28), v29, v30, v33, v24);
  v28 = v18;
  v19 = v31;
  *(_QWORD *)a3 = v17;
  *(_DWORD *)(a3 + 8) = v28;
  v20 = (unsigned __int16 *)(*(_QWORD *)(v19 + 48) + 16LL * (unsigned int)v32);
  *((_QWORD *)&v25 + 1) = v15;
  *(_QWORD *)&v25 = v14;
  v21 = sub_340EC60(
          *(_QWORD **)(a1 + 8),
          v11,
          (__int64)&v35,
          *v20,
          *((_QWORD *)v20 + 1),
          *(unsigned int *)(a2 + 28),
          v31,
          v32,
          v34,
          v25);
  v22 = v35;
  *(_QWORD *)a4 = v21;
  *(_DWORD *)(a4 + 8) = v23;
  if ( v22 )
    sub_B91220((__int64)&v35, v22);
}
