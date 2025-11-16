// Function: sub_3795990
// Address: 0x3795990
//
unsigned __int8 *__fastcall sub_3795990(__int64 *a1, unsigned __int64 a2, int a3, __m128i a4)
{
  unsigned int v4; // ebx
  unsigned int v5; // r15d
  __int64 v8; // rsi
  __int16 *v9; // rax
  __int64 v10; // rsi
  unsigned __int16 v11; // dx
  __int64 v12; // r8
  __int64 v13; // rcx
  _QWORD *v14; // rdi
  __int64 *v15; // rdx
  __int64 v16; // rdx
  unsigned __int64 v17; // rsi
  _BYTE *v18; // rdi
  __int64 v19; // r8
  __int64 v20; // rdx
  __int64 v21; // rcx
  int v22; // eax
  __int64 *v23; // r9
  _WORD *v24; // rdx
  __int64 v25; // r8
  int v26; // eax
  __int64 v27; // rdx
  unsigned int *v28; // rax
  __int64 v29; // rdx
  __int64 v30; // r9
  __int64 v31; // rbx
  unsigned __int8 *v32; // r15
  unsigned __int16 *v33; // rax
  unsigned __int8 *v34; // rax
  __int64 v35; // rdx
  unsigned int v37; // edx
  __int64 v38; // rax
  __int64 v39; // rax
  unsigned int v40; // eax
  __int64 v41; // [rsp+0h] [rbp-100h]
  __int64 *v42; // [rsp+8h] [rbp-F8h]
  __int64 *v43; // [rsp+8h] [rbp-F8h]
  __int128 v45; // [rsp+20h] [rbp-E0h]
  int v46; // [rsp+20h] [rbp-E0h]
  __int128 v47; // [rsp+30h] [rbp-D0h]
  __int64 v48; // [rsp+30h] [rbp-D0h]
  __int64 v49; // [rsp+60h] [rbp-A0h] BYREF
  int v50; // [rsp+68h] [rbp-98h]
  unsigned __int16 v51; // [rsp+70h] [rbp-90h] BYREF
  __int64 v52; // [rsp+78h] [rbp-88h]
  unsigned __int16 v53; // [rsp+80h] [rbp-80h] BYREF
  __int64 v54; // [rsp+88h] [rbp-78h]
  _QWORD v55[2]; // [rsp+90h] [rbp-70h] BYREF
  _BYTE v56[16]; // [rsp+A0h] [rbp-60h] BYREF
  unsigned __int64 v57[2]; // [rsp+B0h] [rbp-50h] BYREF
  _BYTE v58[64]; // [rsp+C0h] [rbp-40h] BYREF

  v8 = *(_QWORD *)(a2 + 80);
  v49 = v8;
  if ( v8 )
    sub_B96E90((__int64)&v49, v8, 1);
  v50 = *(_DWORD *)(a2 + 72);
  v9 = *(__int16 **)(a2 + 48);
  v10 = *a1;
  v11 = *v9;
  v12 = *((_QWORD *)v9 + 1);
  v13 = *((_QWORD *)v9 + 3);
  LOWORD(v9) = v9[8];
  v51 = v11;
  v53 = (unsigned __int16)v9;
  v54 = v13;
  v52 = v12;
  sub_2FE6CC0((__int64)v57, v10, *(_QWORD *)(a1[1] + 64), v11, v12);
  if ( LOBYTE(v57[0]) == 5 )
  {
    *(_QWORD *)&v47 = sub_37946F0((__int64)a1, **(_QWORD **)(a2 + 40), *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL));
    *((_QWORD *)&v47 + 1) = v37;
    v38 = *(_QWORD *)(a2 + 40);
    v17 = *(_QWORD *)(v38 + 40);
    v39 = sub_37946F0((__int64)a1, v17, *(_QWORD *)(v38 + 48));
    v23 = (__int64 *)a1[1];
    *(_QWORD *)&v45 = v39;
    v20 = (unsigned int)v20;
    v22 = v53;
    *((_QWORD *)&v45 + 1) = (unsigned int)v20;
    if ( v53 )
      goto LABEL_9;
LABEL_17:
    v42 = v23;
    v40 = sub_3009970((__int64)&v53, v17, v20, v21, v19);
    v23 = v42;
    v25 = (__int64)v24;
    v5 = v40;
    v26 = v51;
    if ( v51 )
      goto LABEL_10;
    goto LABEL_18;
  }
  v14 = (_QWORD *)a1[1];
  v55[0] = v56;
  v55[1] = 0x100000000LL;
  v57[1] = 0x100000000LL;
  v15 = *(__int64 **)(a2 + 40);
  v57[0] = (unsigned __int64)v58;
  sub_3408690(v14, *v15, v15[1], (unsigned __int16 *)v55, 0, 0, a4, 0, 0);
  v16 = *(_QWORD *)(a2 + 40);
  v17 = *(_QWORD *)(v16 + 40);
  sub_3408690((_QWORD *)a1[1], v17, *(_QWORD *)(v16 + 48), (unsigned __int16 *)v57, 0, 0, a4, 0, 0);
  v18 = (_BYTE *)v55[0];
  v19 = v57[0];
  *(_QWORD *)&v47 = *(_QWORD *)v55[0];
  *((_QWORD *)&v47 + 1) = *(unsigned int *)(v55[0] + 8LL);
  v20 = 0;
  v21 = *(unsigned int *)(v57[0] + 8);
  *(_QWORD *)&v45 = *(_QWORD *)v57[0];
  *((_QWORD *)&v45 + 1) = v21;
  if ( (_BYTE *)v57[0] != v58 )
  {
    _libc_free(v57[0]);
    v18 = (_BYTE *)v55[0];
  }
  if ( v18 != v56 )
    _libc_free((unsigned __int64)v18);
  v22 = v53;
  v23 = (__int64 *)a1[1];
  if ( !v53 )
    goto LABEL_17;
LABEL_9:
  v24 = word_4456580;
  v25 = 0;
  LOWORD(v5) = word_4456580[v22 - 1];
  v26 = v51;
  if ( v51 )
  {
LABEL_10:
    LOWORD(v26) = word_4456580[v26 - 1];
    v27 = 0;
    goto LABEL_11;
  }
LABEL_18:
  v41 = v25;
  v43 = v23;
  v26 = sub_3009970((__int64)&v51, v17, (__int64)v24, v21, v25);
  v25 = v41;
  v23 = v43;
  HIWORD(v4) = HIWORD(v26);
LABEL_11:
  LOWORD(v4) = v26;
  v28 = (unsigned int *)sub_33E5110(v23, v4, v27, v5, v25);
  v31 = (unsigned int)(1 - a3);
  v32 = sub_3411F20((_QWORD *)a1[1], *(unsigned int *)(a2 + 24), (__int64)&v49, v28, v29, v30, v47, v45);
  *((_DWORD *)v32 + 7) = *(_DWORD *)(a2 + 28);
  v33 = (unsigned __int16 *)(*(_QWORD *)(a2 + 48) + 16 * v31);
  v46 = *v33;
  v48 = *((_QWORD *)v33 + 1);
  sub_2FE6CC0((__int64)v57, *a1, *(_QWORD *)(a1[1] + 64), (unsigned __int16)v46, v48);
  if ( LOBYTE(v57[0]) == 5 )
  {
    sub_375FC90((__int64)a1, a2, v31, (unsigned __int64)v32, v31);
  }
  else
  {
    v34 = sub_33FAF80(a1[1], 167, (__int64)&v49, (unsigned __int16)v46, v48, v46, a4);
    sub_3760E70((__int64)a1, a2, v31, (unsigned __int64)v34, v35);
  }
  if ( v49 )
    sub_B91220((__int64)&v49, v49);
  return v32;
}
