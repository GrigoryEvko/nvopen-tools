// Function: sub_382FDF0
// Address: 0x382fdf0
//
void __fastcall sub_382FDF0(__int64 *a1, __int64 a2, __int64 a3, __int64 *a4, __m128i a5)
{
  unsigned int v5; // r15d
  __int64 (__fastcall *v9)(__int64, __int64, unsigned int, __int64); // r10
  __int16 *v10; // rax
  unsigned __int16 v11; // si
  __int64 v12; // r8
  __int64 v13; // rax
  unsigned __int64 v14; // r9
  __int64 v15; // rsi
  unsigned __int64 *v16; // rcx
  unsigned __int64 v17; // r8
  unsigned __int64 v18; // r14
  __int64 v19; // rax
  unsigned __int16 v20; // si
  __int64 v21; // rdx
  unsigned __int8 *v22; // rax
  __int64 v23; // rcx
  __int64 v24; // r8
  int v25; // edx
  int v26; // edx
  __int64 v27; // rax
  __int64 v28; // rdx
  char v29; // r10
  __int64 v30; // rax
  __int64 v31; // rsi
  __int64 v32; // rdx
  unsigned __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rdx
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rax
  __int64 v39; // rdx
  char v40; // al
  unsigned int v41; // eax
  __int64 v42; // r13
  int v43; // eax
  __int64 v44; // rdx
  __int64 v45; // r9
  __int64 v46; // rdx
  int v47; // edx
  unsigned __int64 v48; // [rsp+10h] [rbp-110h]
  unsigned __int64 v49; // [rsp+18h] [rbp-108h]
  char v50; // [rsp+18h] [rbp-108h]
  unsigned __int64 v51; // [rsp+20h] [rbp-100h]
  __int64 v52; // [rsp+30h] [rbp-F0h]
  __int64 v53; // [rsp+38h] [rbp-E8h]
  unsigned int v54; // [rsp+70h] [rbp-B0h] BYREF
  __int64 v55; // [rsp+78h] [rbp-A8h]
  __int64 v56; // [rsp+80h] [rbp-A0h] BYREF
  int v57; // [rsp+88h] [rbp-98h]
  __int64 v58; // [rsp+90h] [rbp-90h]
  __int64 v59; // [rsp+98h] [rbp-88h]
  __int64 v60; // [rsp+A0h] [rbp-80h]
  __int64 v61; // [rsp+A8h] [rbp-78h]
  __int64 v62; // [rsp+B0h] [rbp-70h]
  __int64 v63; // [rsp+B8h] [rbp-68h]
  __int64 v64; // [rsp+C0h] [rbp-60h] BYREF
  __int64 v65; // [rsp+C8h] [rbp-58h]
  __int64 v66; // [rsp+D0h] [rbp-50h] BYREF
  __int64 v67; // [rsp+D8h] [rbp-48h]
  __int64 v68; // [rsp+E0h] [rbp-40h]

  v9 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  v10 = *(__int16 **)(a2 + 48);
  v11 = *v10;
  v12 = *((_QWORD *)v10 + 1);
  v13 = a1[1];
  if ( v9 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v66, *a1, *(_QWORD *)(v13 + 64), v11, v12);
    LOWORD(v54) = v67;
    v55 = v68;
  }
  else
  {
    v54 = v9(*a1, *(_QWORD *)(v13 + 64), v11, v12);
    v55 = v46;
  }
  v15 = *(_QWORD *)(a2 + 80);
  v56 = v15;
  if ( v15 )
    sub_B96E90((__int64)&v56, v15, 1);
  v16 = *(unsigned __int64 **)(a2 + 40);
  v57 = *(_DWORD *)(a2 + 72);
  v17 = *v16;
  v52 = v16[1];
  v18 = *v16;
  v53 = 16LL * *((unsigned int *)v16 + 2);
  v19 = *(_QWORD *)(*v16 + 48) + v53;
  v20 = *(_WORD *)v19;
  v21 = *(_QWORD *)(v19 + 8);
  LOWORD(v64) = v20;
  v65 = v21;
  if ( v20 == (_WORD)v54 )
  {
    if ( v20 || v21 == v55 )
      goto LABEL_7;
    v67 = v55;
    LOWORD(v66) = 0;
LABEL_12:
    v49 = v17;
    v27 = sub_3007260((__int64)&v66);
    v17 = v49;
    v62 = v27;
    v14 = v27;
    v63 = v28;
    v29 = v28;
    if ( !v20 )
      goto LABEL_13;
    goto LABEL_29;
  }
  LOWORD(v66) = v54;
  v67 = v55;
  if ( !(_WORD)v54 )
    goto LABEL_12;
  if ( (_WORD)v54 == 1 || (unsigned __int16)(v54 - 504) <= 7u )
    goto LABEL_47;
  v14 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v54 - 16];
  v29 = byte_444C4A0[16 * (unsigned __int16)v54 - 8];
  if ( v20 )
  {
LABEL_29:
    if ( v20 != 1 && (unsigned __int16)(v20 - 504) > 7u )
    {
      v33 = *(_QWORD *)&byte_444C4A0[16 * v20 - 16];
      if ( !byte_444C4A0[16 * v20 - 8] )
        goto LABEL_32;
      goto LABEL_14;
    }
LABEL_47:
    BUG();
  }
LABEL_13:
  v48 = v17;
  v50 = v29;
  v51 = v14;
  v30 = sub_3007260((__int64)&v64);
  v17 = v48;
  v29 = v50;
  v31 = v30;
  v14 = v51;
  v61 = v32;
  LOBYTE(v30) = v32;
  v33 = v31;
  v60 = v31;
  if ( !(_BYTE)v30 )
    goto LABEL_32;
LABEL_14:
  if ( !v29 )
    goto LABEL_15;
LABEL_32:
  if ( v14 >= v33 )
  {
LABEL_7:
    v22 = sub_33FAF80(a1[1], 214, (__int64)&v56, v54, v55, v14, a5);
    v23 = v54;
    v24 = v55;
    *(_QWORD *)a3 = v22;
    *(_DWORD *)(a3 + 8) = v25;
    *a4 = (__int64)sub_3400BD0(a1[1], 0, (__int64)&v56, v23, v24, 0, a5, 0);
    *((_DWORD *)a4 + 2) = v26;
    goto LABEL_8;
  }
LABEL_15:
  v34 = sub_37AE0F0((__int64)a1, v17, v52);
  sub_375BC20(a1, v34, v35, a3, (__int64)a4, a5);
  v58 = sub_2D5B750((unsigned __int16 *)&v54);
  v36 = *(_QWORD *)(v18 + 48) + v53;
  v59 = v37;
  LOWORD(v37) = *(_WORD *)v36;
  v38 = *(_QWORD *)(v36 + 8);
  LOWORD(v66) = v37;
  v67 = v38;
  v64 = sub_2D5B750((unsigned __int16 *)&v66);
  v65 = v39;
  v40 = v39;
  v66 = v64 - v58;
  if ( v58 )
    v40 = v59;
  LOBYTE(v67) = v40;
  v41 = sub_CA1930(&v66);
  v42 = a1[1];
  switch ( v41 )
  {
    case 1u:
      LOWORD(v43) = 2;
      goto LABEL_38;
    case 2u:
      LOWORD(v43) = 3;
      goto LABEL_38;
    case 4u:
      LOWORD(v43) = 4;
LABEL_38:
      v45 = 0;
      goto LABEL_39;
    case 8u:
      LOWORD(v43) = 5;
      goto LABEL_38;
    case 0x10u:
      LOWORD(v43) = 6;
      goto LABEL_38;
    case 0x20u:
      LOWORD(v43) = 7;
      goto LABEL_38;
    case 0x40u:
      LOWORD(v43) = 8;
      goto LABEL_38;
    case 0x80u:
      LOWORD(v43) = 9;
      goto LABEL_38;
  }
  v43 = sub_3007020(*(_QWORD **)(v42 + 64), v41);
  HIWORD(v5) = HIWORD(v43);
  v45 = v44;
LABEL_39:
  LOWORD(v5) = v43;
  *a4 = (__int64)sub_34070B0((_QWORD *)v42, *a4, a4[1], (__int64)&v56, v5, v45, a5);
  *((_DWORD *)a4 + 2) = v47;
LABEL_8:
  if ( v56 )
    sub_B91220((__int64)&v56, v56);
}
