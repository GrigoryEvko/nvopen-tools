// Function: sub_37675C0
// Address: 0x37675c0
//
unsigned __int8 *__fastcall sub_37675C0(
        __int64 *a1,
        __int64 a2,
        __m128i a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  unsigned int v7; // r12d
  __int16 *v10; // rax
  unsigned __int16 v11; // r15
  __int64 v12; // rax
  __int64 v14; // r13
  int v15; // eax
  int v16; // r9d
  __int64 v17; // rdi
  __int64 (*v18)(); // rax
  __int64 v19; // rsi
  __int64 v20; // rdi
  unsigned __int8 *v21; // r14
  __int64 v22; // rdx
  __int64 v23; // r15
  _QWORD *v24; // rdi
  _QWORD *v25; // rax
  __int64 v26; // rdx
  __int64 v27; // r10
  int v28; // r9d
  unsigned __int8 *v29; // r13
  __int64 v30; // rdx
  _BYTE *v31; // rdi
  __int64 v32; // rdx
  char v33; // al
  char v34; // al
  unsigned int v35; // esi
  __int64 v36; // rax
  char v37; // cl
  unsigned __int8 v38; // al
  char v39; // al
  char v40; // al
  _QWORD *v41; // [rsp+0h] [rbp-100h]
  __int64 v42; // [rsp+8h] [rbp-F8h]
  __int64 v43; // [rsp+18h] [rbp-E8h]
  __int64 *v44; // [rsp+20h] [rbp-E0h]
  unsigned int v45; // [rsp+30h] [rbp-D0h]
  const void *v46; // [rsp+30h] [rbp-D0h]
  __int64 v47; // [rsp+38h] [rbp-C8h]
  unsigned int v48; // [rsp+50h] [rbp-B0h] BYREF
  __int64 v49; // [rsp+58h] [rbp-A8h]
  __int64 v50; // [rsp+60h] [rbp-A0h] BYREF
  int v51; // [rsp+68h] [rbp-98h]
  __int64 v52; // [rsp+70h] [rbp-90h] BYREF
  int v53; // [rsp+78h] [rbp-88h]
  _BYTE *v54; // [rsp+80h] [rbp-80h] BYREF
  __int64 v55; // [rsp+88h] [rbp-78h]
  _BYTE v56[112]; // [rsp+90h] [rbp-70h] BYREF

  v10 = *(__int16 **)(a2 + 48);
  v11 = *v10;
  v12 = *((_QWORD *)v10 + 1);
  LOWORD(v48) = v11;
  v49 = v12;
  if ( v11 )
  {
    if ( (unsigned __int16)(v11 - 176) <= 0x34u )
      return sub_345C4B0(a1[1], a2, *a1);
  }
  else if ( sub_3007100((__int64)&v48) )
  {
    return sub_345C4B0(a1[1], a2, *a1);
  }
  v14 = 0;
  v54 = v56;
  v55 = 0x1000000000LL;
  sub_3763720(v48, v49, (__int64)&v54, a5, a6, a7);
  v45 = v55;
  v44 = *(__int64 **)(*a1 + 64);
  LOWORD(v15) = sub_2D43050(5, v55);
  if ( !(_WORD)v15 )
  {
    v15 = sub_3009400(v44, 5, 0, v45, 0);
    HIWORD(v7) = HIWORD(v15);
    v14 = v30;
  }
  v17 = a1[1];
  LOWORD(v7) = v15;
  v18 = *(__int64 (**)())(*(_QWORD *)v17 + 624LL);
  if ( v18 == sub_2FE3180
    || ((unsigned __int8 (__fastcall *)(__int64, _BYTE *, _QWORD, _QWORD, __int64))v18)(
         v17,
         v54,
         (unsigned int)v55,
         v7,
         v14) )
  {
    v19 = *(_QWORD *)(a2 + 80);
    v50 = v19;
    if ( v19 )
      sub_B96E90((__int64)&v50, v19, 1);
    v20 = *a1;
    v51 = *(_DWORD *)(a2 + 72);
    v21 = sub_33FAF80(v20, 234, (__int64)&v50, v7, v14, v16, a3);
    v23 = v22;
    v24 = (_QWORD *)*a1;
    v43 = *a1;
    v46 = v54;
    v52 = 0;
    v47 = (unsigned int)v55;
    v53 = 0;
    v25 = sub_33F17F0(v24, 51, (__int64)&v52, v7, v14);
    v27 = v43;
    if ( v52 )
    {
      v41 = v25;
      v42 = v26;
      sub_B91220((__int64)&v52, v52);
      v25 = v41;
      v26 = v42;
      v27 = v43;
    }
    sub_33FCE10(v27, v7, v14, (__int64)&v50, (__int64)v21, v23, a3, (__int64)v25, v26, v46, v47);
    v29 = sub_33FAF80(*a1, 234, (__int64)&v50, v48, v49, v28, a3);
    if ( v50 )
      sub_B91220((__int64)&v50, v50);
    goto LABEL_14;
  }
  v31 = (_BYTE *)a1[1];
  if ( v11 == 1 )
  {
    v33 = v31[7104];
    if ( v33 && v33 != 4 )
      goto LABEL_21;
    v34 = v31[7106];
    if ( v34 )
    {
      if ( v34 != 4 )
        goto LABEL_21;
    }
    v35 = 1;
    v36 = 1;
    v37 = v31[7100];
    if ( !v37 || v37 == 4 )
      goto LABEL_26;
LABEL_32:
    if ( v37 != 1 )
      goto LABEL_21;
    goto LABEL_33;
  }
  if ( !v11 )
    goto LABEL_21;
  v32 = v11;
  if ( !*(_QWORD *)&v31[8 * v11 + 112] )
    goto LABEL_21;
  v35 = v11;
  v39 = v31[500 * v11 + 6604];
  if ( v39 )
  {
    if ( v39 != 4 || !*(_QWORD *)&v31[8 * v11 + 112] )
      goto LABEL_21;
  }
  v40 = v31[500 * v11 + 6606];
  if ( v40 )
  {
    if ( v40 != 4 || !*(_QWORD *)&v31[8 * v11 + 112] )
      goto LABEL_21;
  }
  v37 = v31[500 * v11 + 6600];
  if ( !v37 )
  {
LABEL_35:
    if ( !*(_QWORD *)&v31[8 * v32 + 112] )
      goto LABEL_21;
    goto LABEL_36;
  }
  if ( v37 != 4 )
    goto LABEL_32;
LABEL_33:
  if ( v11 != 1 )
  {
    v32 = v11;
    goto LABEL_35;
  }
LABEL_36:
  v36 = v35;
LABEL_26:
  v38 = v31[500 * v36 + 6601];
  if ( v38 <= 1u || v38 == 4 )
  {
    v29 = sub_345C4B0((__int64)v31, a2, *a1);
    goto LABEL_14;
  }
LABEL_21:
  v29 = 0;
LABEL_14:
  if ( v54 != v56 )
    _libc_free((unsigned __int64)v54);
  return v29;
}
