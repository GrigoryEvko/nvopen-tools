// Function: sub_381E300
// Address: 0x381e300
//
void __fastcall sub_381E300(__int64 a1, __int64 a2, unsigned int *a3, __int64 a4, __m128i a5)
{
  unsigned int v5; // r15d
  __int64 v9; // rsi
  __int64 v10; // rax
  unsigned __int16 v11; // dx
  __int64 v12; // rax
  __int64 v13; // rax
  unsigned __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rdx
  char v17; // al
  unsigned int v18; // r12d
  __int64 v19; // rax
  __int64 v20; // rdx
  unsigned int v21; // eax
  unsigned int v22; // eax
  __int64 v23; // r14
  int v24; // eax
  unsigned __int64 v25; // rdx
  _QWORD *v26; // r15
  __int128 v27; // rax
  __int64 v28; // r9
  unsigned __int8 *v29; // rax
  __int64 v30; // rcx
  __int64 v31; // r8
  unsigned int v32; // edx
  int v33; // edx
  __int64 v34; // rax
  __int128 v35; // rax
  __int64 v36; // r9
  int v37; // edx
  __int64 v38; // [rsp+30h] [rbp-90h] BYREF
  int v39; // [rsp+38h] [rbp-88h]
  unsigned int v40; // [rsp+40h] [rbp-80h] BYREF
  __int64 v41; // [rsp+48h] [rbp-78h]
  unsigned int v42; // [rsp+50h] [rbp-70h] BYREF
  unsigned __int64 v43; // [rsp+58h] [rbp-68h]
  __int64 v44; // [rsp+60h] [rbp-60h] BYREF
  char v45; // [rsp+68h] [rbp-58h]
  __int64 v46; // [rsp+70h] [rbp-50h] BYREF
  __int64 v47; // [rsp+78h] [rbp-48h]
  __int64 v48; // [rsp+80h] [rbp-40h]
  __int64 v49; // [rsp+88h] [rbp-38h]

  v9 = *(_QWORD *)(a2 + 80);
  v38 = v9;
  if ( v9 )
    sub_B96E90((__int64)&v38, v9, 1);
  v39 = *(_DWORD *)(a2 + 72);
  sub_375E510(a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL), (__int64)a3, a4);
  v10 = *(_QWORD *)(*(_QWORD *)a3 + 48LL) + 16LL * a3[2];
  v11 = *(_WORD *)v10;
  v41 = *(_QWORD *)(v10 + 8);
  v12 = *(_QWORD *)(a2 + 40);
  LOWORD(v40) = v11;
  v13 = *(_QWORD *)(v12 + 40);
  v14 = *(_QWORD *)(v13 + 104);
  LOWORD(v13) = *(_WORD *)(v13 + 96);
  v43 = v14;
  LOWORD(v42) = v13;
  if ( v11 )
  {
    if ( v11 == 1 || (unsigned __int16)(v11 - 504) <= 7u )
      goto LABEL_37;
    v34 = 16LL * (v11 - 1);
    v16 = *(_QWORD *)&byte_444C4A0[v34];
    v17 = byte_444C4A0[v34 + 8];
  }
  else
  {
    v48 = sub_3007260((__int64)&v40);
    v49 = v15;
    v16 = v48;
    v17 = v49;
  }
  v46 = v16;
  LOBYTE(v47) = v17;
  v18 = sub_CA1930(&v46);
  if ( !(_WORD)v42 )
  {
    v19 = sub_3007260((__int64)&v42);
    v46 = v19;
    v47 = v20;
    goto LABEL_7;
  }
  if ( (_WORD)v42 == 1 || (unsigned __int16)(v42 - 504) <= 7u )
LABEL_37:
    BUG();
  v20 = 16LL * ((unsigned __int16)v42 - 1);
  v19 = *(_QWORD *)&byte_444C4A0[v20];
  LOBYTE(v20) = byte_444C4A0[v20 + 8];
LABEL_7:
  v44 = v19;
  v45 = v20;
  v21 = sub_CA1930(&v44);
  if ( v18 >= v21 )
  {
    v26 = *(_QWORD **)(a1 + 8);
    *(_QWORD *)&v27 = sub_33F7D60(v26, v42, v43);
    v29 = sub_3406EB0(v26, 4u, (__int64)&v38, v40, v41, v28, *(_OWORD *)a3, v27);
    v30 = v40;
    v31 = v41;
    *(_QWORD *)a3 = v29;
    a3[2] = v32;
    *(_QWORD *)a4 = sub_3400BD0(*(_QWORD *)(a1 + 8), 0, (__int64)&v38, v30, v31, 0, a5, 0);
    *(_DWORD *)(a4 + 8) = v33;
    goto LABEL_18;
  }
  v22 = v21 - v18;
  v23 = *(_QWORD *)(a1 + 8);
  switch ( v22 )
  {
    case 1u:
      LOWORD(v24) = 2;
LABEL_28:
      v25 = 0;
      goto LABEL_29;
    case 2u:
      LOWORD(v24) = 3;
      goto LABEL_28;
    case 4u:
      LOWORD(v24) = 4;
      goto LABEL_28;
    case 8u:
      LOWORD(v24) = 5;
      goto LABEL_28;
    case 0x10u:
      LOWORD(v24) = 6;
      goto LABEL_28;
    case 0x20u:
      LOWORD(v24) = 7;
      goto LABEL_28;
    case 0x40u:
      LOWORD(v24) = 8;
      goto LABEL_28;
    case 0x80u:
      LOWORD(v24) = 9;
      goto LABEL_28;
  }
  v24 = sub_3007020(*(_QWORD **)(v23 + 64), v22);
  HIWORD(v5) = HIWORD(v24);
LABEL_29:
  LOWORD(v5) = v24;
  *(_QWORD *)&v35 = sub_33F7D60((_QWORD *)v23, v5, v25);
  *(_QWORD *)a4 = sub_3406EB0((_QWORD *)v23, 4u, (__int64)&v38, v40, v41, v36, *(_OWORD *)a4, v35);
  *(_DWORD *)(a4 + 8) = v37;
LABEL_18:
  if ( v38 )
    sub_B91220((__int64)&v38, v38);
}
