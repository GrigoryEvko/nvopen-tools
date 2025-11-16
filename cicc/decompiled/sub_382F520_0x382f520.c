// Function: sub_382F520
// Address: 0x382f520
//
void __fastcall sub_382F520(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __m128i a5)
{
  __int64 v7; // r10
  __int64 (__fastcall *v8)(__int64, __int64, unsigned int, __int64); // r11
  __int16 *v9; // rax
  unsigned __int16 v10; // si
  __int64 v11; // r8
  __int64 v12; // rax
  unsigned int v13; // r15d
  unsigned __int16 v14; // cx
  __int64 v15; // r14
  __int64 v16; // rsi
  unsigned __int64 *v17; // rax
  unsigned __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // rax
  unsigned __int16 v21; // si
  __int64 v22; // rax
  int v23; // edx
  _QWORD *v24; // rdi
  _QWORD *v25; // rax
  int v26; // edx
  int v27; // r14d
  _QWORD *v28; // r12
  __int64 v29; // rax
  unsigned __int64 v30; // r10
  __int64 v31; // rdx
  char v32; // r11
  __int64 v33; // rax
  __int64 v34; // rsi
  __int64 v35; // rdx
  __int64 v36; // rax
  unsigned __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rdx
  int v40; // eax
  __int64 v41; // rdx
  unsigned __int64 v42; // [rsp+0h] [rbp-E0h]
  __int64 v43; // [rsp+8h] [rbp-D8h]
  unsigned __int64 v44; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v45; // [rsp+10h] [rbp-D0h]
  __int64 v46; // [rsp+18h] [rbp-C8h]
  unsigned __int16 v47; // [rsp+20h] [rbp-C0h]
  char v48; // [rsp+20h] [rbp-C0h]
  unsigned __int16 v49; // [rsp+24h] [rbp-BCh]
  unsigned __int16 v50; // [rsp+24h] [rbp-BCh]
  __int64 v52; // [rsp+50h] [rbp-90h] BYREF
  int v53; // [rsp+58h] [rbp-88h]
  unsigned __int16 v54; // [rsp+60h] [rbp-80h] BYREF
  __int64 v55; // [rsp+68h] [rbp-78h]
  __int64 v56; // [rsp+70h] [rbp-70h]
  __int64 v57; // [rsp+78h] [rbp-68h]
  __int64 v58; // [rsp+80h] [rbp-60h]
  __int64 v59; // [rsp+88h] [rbp-58h]
  __int64 v60; // [rsp+90h] [rbp-50h] BYREF
  __int64 v61; // [rsp+98h] [rbp-48h]
  __int64 v62; // [rsp+A0h] [rbp-40h]

  v7 = *a1;
  v8 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v9 = *(__int16 **)(a2 + 48);
  v10 = *v9;
  v11 = *((_QWORD *)v9 + 1);
  v12 = a1[1];
  if ( v8 == sub_2D56A50 )
  {
    HIWORD(v13) = 0;
    sub_2FE6CC0((__int64)&v60, v7, *(_QWORD *)(v12 + 64), v10, v11);
    v14 = v61;
    v15 = v62;
  }
  else
  {
    v40 = v8(v7, *(_QWORD *)(v12 + 64), v10, v11);
    HIWORD(v13) = HIWORD(v40);
    v14 = v40;
    v15 = v41;
  }
  v16 = *(_QWORD *)(a2 + 80);
  v52 = v16;
  if ( v16 )
  {
    v49 = v14;
    sub_B96E90((__int64)&v52, v16, 1);
    v14 = v49;
  }
  v53 = *(_DWORD *)(a2 + 72);
  v17 = *(unsigned __int64 **)(a2 + 40);
  v18 = *v17;
  v19 = v17[1];
  v20 = *(_QWORD *)(*v17 + 48) + 16LL * *((unsigned int *)v17 + 2);
  v21 = *(_WORD *)v20;
  v22 = *(_QWORD *)(v20 + 8);
  v54 = v21;
  v55 = v22;
  if ( v21 == v14 )
  {
    if ( v14 || v22 == v15 )
      goto LABEL_7;
    v61 = v15;
    LOWORD(v60) = 0;
LABEL_14:
    v44 = v18;
    v46 = v19;
    v47 = v14;
    v29 = sub_3007260((__int64)&v60);
    v18 = v44;
    v58 = v29;
    v14 = v47;
    v30 = v29;
    v59 = v31;
    v19 = v46;
    v32 = v31;
    if ( !v21 )
    {
LABEL_15:
      v42 = v18;
      v43 = v19;
      v45 = v30;
      v48 = v32;
      v50 = v14;
      v33 = sub_3007260((__int64)&v54);
      v18 = v42;
      v19 = v43;
      v34 = v33;
      v36 = v35;
      v30 = v45;
      v32 = v48;
      v56 = v34;
      v14 = v50;
      v37 = v34;
      v57 = v36;
      goto LABEL_16;
    }
    goto LABEL_23;
  }
  LOWORD(v60) = v14;
  v61 = v15;
  if ( !v14 )
    goto LABEL_14;
  if ( v14 == 1 || (unsigned __int16)(v14 - 504) <= 7u )
LABEL_29:
    BUG();
  v30 = *(_QWORD *)&byte_444C4A0[16 * v14 - 16];
  v32 = byte_444C4A0[16 * v14 - 8];
  if ( !v21 )
    goto LABEL_15;
LABEL_23:
  if ( v21 == 1 || (unsigned __int16)(v21 - 504) <= 7u )
    goto LABEL_29;
  v37 = *(_QWORD *)&byte_444C4A0[16 * v21 - 16];
  LOBYTE(v36) = byte_444C4A0[16 * v21 - 8];
LABEL_16:
  if ( (_BYTE)v36 && !v32 || v30 < v37 )
  {
    v38 = sub_37AE0F0((__int64)a1, v18, v19);
    sub_375BC20(a1, v38, v39, a3, a4, a5);
    goto LABEL_10;
  }
LABEL_7:
  LOWORD(v13) = v14;
  *(_QWORD *)a3 = sub_33FAF80(a1[1], 215, (__int64)&v52, v13, v15, v19, a5);
  v60 = 0;
  *(_DWORD *)(a3 + 8) = v23;
  v24 = (_QWORD *)a1[1];
  LODWORD(v61) = 0;
  v25 = sub_33F17F0(v24, 51, (__int64)&v60, v13, v15);
  v27 = v26;
  v28 = v25;
  if ( v60 )
    sub_B91220((__int64)&v60, v60);
  *(_QWORD *)a4 = v28;
  *(_DWORD *)(a4 + 8) = v27;
LABEL_10:
  if ( v52 )
    sub_B91220((__int64)&v52, v52);
}
