// Function: sub_382E5B0
// Address: 0x382e5b0
//
unsigned __int8 *__fastcall sub_382E5B0(
        __int64 a1,
        unsigned __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __m128i a6,
        __int64 a7,
        __int128 a8)
{
  __int64 v11; // rax
  __int64 v12; // rsi
  unsigned __int16 v13; // dx
  __int64 v14; // rax
  unsigned int v15; // edx
  unsigned __int16 *v16; // rdx
  int v17; // eax
  __int64 v18; // rbx
  __int64 v19; // rdx
  __int64 v20; // rax
  unsigned __int16 v21; // r15
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rbx
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdx
  __int128 v29; // rax
  __int64 v30; // r9
  __int128 v31; // rax
  __int64 v32; // r9
  unsigned __int8 *v33; // r12
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // rdx
  __int64 v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // r8
  __int128 v42; // [rsp-40h] [rbp-100h]
  __int64 v43; // [rsp+8h] [rbp-B8h]
  __int64 v44; // [rsp+10h] [rbp-B0h]
  __int128 v45; // [rsp+10h] [rbp-B0h]
  __int128 v46; // [rsp+20h] [rbp-A0h]
  unsigned __int16 v47; // [rsp+30h] [rbp-90h] BYREF
  __int64 v48; // [rsp+38h] [rbp-88h]
  __int64 v49; // [rsp+40h] [rbp-80h] BYREF
  int v50; // [rsp+48h] [rbp-78h]
  unsigned int v51; // [rsp+50h] [rbp-70h] BYREF
  __int64 v52; // [rsp+58h] [rbp-68h]
  unsigned __int16 v53; // [rsp+60h] [rbp-60h] BYREF
  __int64 v54; // [rsp+68h] [rbp-58h]
  __int64 v55; // [rsp+70h] [rbp-50h]
  __int64 v56; // [rsp+78h] [rbp-48h]
  __int64 v57; // [rsp+80h] [rbp-40h] BYREF
  __int64 v58; // [rsp+88h] [rbp-38h]

  v11 = *(_QWORD *)(a2 + 48) + 16LL * (unsigned int)a3;
  v12 = *(_QWORD *)(a2 + 80);
  *(_QWORD *)&v46 = a4;
  v13 = *(_WORD *)v11;
  v14 = *(_QWORD *)(v11 + 8);
  *((_QWORD *)&v46 + 1) = a5;
  v49 = v12;
  v47 = v13;
  v48 = v14;
  if ( v12 )
    sub_B96E90((__int64)&v49, v12, 1);
  v50 = *(_DWORD *)(a2 + 72);
  v44 = sub_37AE0F0(a1, a2, a3);
  v43 = v15;
  v16 = (unsigned __int16 *)(*(_QWORD *)(v44 + 48) + 16LL * v15);
  v17 = *v16;
  v18 = *((_QWORD *)v16 + 1);
  LOWORD(v51) = v17;
  v52 = v18;
  if ( !(_WORD)v17 )
  {
    if ( !sub_30070B0((__int64)&v51) )
    {
      v58 = v18;
      LOWORD(v57) = 0;
      goto LABEL_11;
    }
    LOWORD(v17) = sub_3009970((__int64)&v51, a2, v39, v40, v41);
LABEL_10:
    LOWORD(v57) = v17;
    v58 = v19;
    if ( (_WORD)v17 )
      goto LABEL_6;
LABEL_11:
    v20 = sub_3007260((__int64)&v57);
    v21 = v47;
    v25 = v22;
    v23 = v20;
    v24 = v25;
    v55 = v23;
    LODWORD(v25) = v23;
    v56 = v24;
    if ( !v47 )
      goto LABEL_20;
LABEL_12:
    if ( (unsigned __int16)(v21 - 17) <= 0xD3u )
    {
      v21 = word_4456580[v21 - 1];
      v26 = 0;
LABEL_14:
      v53 = v21;
      v54 = v26;
      if ( v21 )
        goto LABEL_22;
LABEL_15:
      v27 = sub_3007260((__int64)&v53);
      v57 = v27;
      v58 = v28;
      goto LABEL_16;
    }
LABEL_13:
    v26 = v48;
    goto LABEL_14;
  }
  if ( (unsigned __int16)(v17 - 17) <= 0xD3u )
  {
    LOWORD(v17) = word_4456580[v17 - 1];
    v19 = 0;
    goto LABEL_10;
  }
  LOWORD(v57) = v17;
  v58 = v18;
LABEL_6:
  if ( (_WORD)v17 == 1 || (unsigned __int16)(v17 - 504) <= 7u )
LABEL_8:
    BUG();
  v21 = v47;
  v25 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v17 - 16];
  if ( v47 )
    goto LABEL_12;
LABEL_20:
  if ( !sub_30070B0((__int64)&v47) )
    goto LABEL_13;
  v21 = sub_3009970((__int64)&v47, a2, v35, v36, v37);
  v53 = v21;
  v54 = v38;
  if ( !v21 )
    goto LABEL_15;
LABEL_22:
  if ( v21 == 1 || (unsigned __int16)(v21 - 504) <= 7u )
    goto LABEL_8;
  v27 = *(_QWORD *)&byte_444C4A0[16 * v21 - 16];
LABEL_16:
  *(_QWORD *)&v29 = sub_3400E40(*(_QWORD *)(a1 + 8), (unsigned int)(v25 - v27), v51, v52, (__int64)&v49, a6);
  *((_QWORD *)&v42 + 1) = a3 & 0xFFFFFFFF00000000LL | v43;
  *(_QWORD *)&v42 = v44;
  v45 = v29;
  *(_QWORD *)&v31 = sub_33FC130(*(_QWORD **)(a1 + 8), 402, (__int64)&v49, v51, v52, v30, v42, v29, v46, a8);
  v33 = sub_33FC130(*(_QWORD **)(a1 + 8), 397, (__int64)&v49, v51, v52, v32, v31, v45, v46, a8);
  if ( v49 )
    sub_B91220((__int64)&v49, v49);
  return v33;
}
