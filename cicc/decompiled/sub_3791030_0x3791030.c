// Function: sub_3791030
// Address: 0x3791030
//
__int64 __fastcall sub_3791030(__int64 *a1, __int64 a2, __m128i a3)
{
  unsigned int v3; // r14d
  __int64 *v4; // rax
  __int64 v5; // r9
  __int64 v6; // r15
  __int64 v7; // r13
  __int64 v8; // r12
  __int64 v9; // rax
  unsigned __int16 v10; // si
  __int64 v11; // r8
  __int64 v12; // rax
  __int64 (__fastcall *v13)(__int64, __int64, unsigned int, __int64); // r10
  __int64 v14; // rcx
  unsigned int v15; // eax
  __int64 v16; // r8
  unsigned __int64 v17; // rsi
  unsigned __int16 *v18; // r13
  int v19; // eax
  __int64 v20; // rdx
  unsigned __int16 v21; // ax
  unsigned int v22; // r12d
  __int64 *v23; // r13
  int v24; // eax
  __int64 v25; // r8
  __int64 v26; // r9
  __int64 v27; // r10
  unsigned __int8 *v28; // rax
  __int64 v29; // rdx
  __int64 v30; // r13
  unsigned __int8 *v31; // r12
  unsigned __int8 *v32; // rax
  __int64 v33; // rdx
  __int64 v34; // r15
  unsigned __int8 *v35; // r14
  unsigned __int8 *v36; // rax
  __int64 v37; // rdx
  _QWORD *v38; // rbx
  unsigned __int8 *v39; // r8
  __int64 v40; // r9
  __int64 v41; // rsi
  __int64 v42; // r12
  __int64 v44; // rdx
  __int64 v45; // rdx
  __int64 v46; // rdx
  __int128 v47; // [rsp-30h] [rbp-F0h]
  __int128 v48; // [rsp-20h] [rbp-E0h]
  __int128 v49; // [rsp-10h] [rbp-D0h]
  char v50; // [rsp+6h] [rbp-BAh]
  char v51; // [rsp+7h] [rbp-B9h]
  __int64 v52; // [rsp+8h] [rbp-B8h]
  __int64 v53; // [rsp+8h] [rbp-B8h]
  __int64 v54; // [rsp+10h] [rbp-B0h]
  __int64 v55; // [rsp+18h] [rbp-A8h]
  __int64 v56; // [rsp+20h] [rbp-A0h]
  __int64 v57; // [rsp+28h] [rbp-98h]
  __int64 v58; // [rsp+30h] [rbp-90h]
  unsigned __int8 *v59; // [rsp+30h] [rbp-90h]
  __int64 v60; // [rsp+38h] [rbp-88h]
  __int64 v62; // [rsp+50h] [rbp-70h]
  unsigned int v63; // [rsp+60h] [rbp-60h] BYREF
  __int64 v64; // [rsp+68h] [rbp-58h]
  __int64 v65; // [rsp+70h] [rbp-50h] BYREF
  __int64 v66; // [rsp+78h] [rbp-48h]
  __int64 v67; // [rsp+80h] [rbp-40h]

  v4 = *(__int64 **)(a2 + 40);
  v5 = *a1;
  v6 = *v4;
  v7 = *((unsigned int *)v4 + 12);
  v58 = v4[1];
  v8 = v4[5];
  v57 = v8;
  v56 = v4[6];
  v55 = v4[10];
  v54 = v4[11];
  v9 = *(_QWORD *)(*v4 + 48) + 16LL * *((unsigned int *)v4 + 2);
  v10 = *(_WORD *)v9;
  v11 = *(_QWORD *)(v9 + 8);
  v12 = a1[1];
  v13 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(*(_QWORD *)*a1 + 592LL);
  if ( v13 == sub_2D56A50 )
  {
    sub_2FE6CC0((__int64)&v65, v5, *(_QWORD *)(v12 + 64), v10, v11);
    LOWORD(v15) = v66;
    LOWORD(v63) = v66;
    v64 = v67;
  }
  else
  {
    v15 = v13(v5, *(_QWORD *)(v12 + 64), v10, v11);
    v14 = (__int64)&v65;
    v63 = v15;
    v64 = v46;
  }
  if ( (_WORD)v15 )
  {
    LOBYTE(v14) = (unsigned __int16)(v15 - 176) <= 0x34u;
    v16 = (unsigned int)v14;
    v17 = word_4456340[(unsigned __int16)v15 - 1];
  }
  else
  {
    v17 = sub_3007240((__int64)&v63);
    v16 = HIDWORD(v17);
    v14 = HIDWORD(v17);
  }
  v18 = (unsigned __int16 *)(*(_QWORD *)(v8 + 48) + 16 * v7);
  v19 = *v18;
  v20 = *((_QWORD *)v18 + 1);
  LOWORD(v65) = v19;
  v66 = v20;
  if ( (_WORD)v19 )
  {
    v52 = 0;
    v21 = word_4456580[v19 - 1];
  }
  else
  {
    v50 = v16;
    v51 = v14;
    v21 = sub_3009970((__int64)&v65, v17, v20, v14, v16);
    LOBYTE(v16) = v50;
    v52 = v45;
    LOBYTE(v14) = v51;
  }
  LODWORD(v62) = v17;
  v22 = v21;
  BYTE4(v62) = v16;
  v23 = *(__int64 **)(a1[1] + 64);
  if ( (_BYTE)v14 )
  {
    LOWORD(v24) = sub_2D43AD0(v21, v17);
    v27 = 0;
    if ( (_WORD)v24 )
      goto LABEL_9;
  }
  else
  {
    LOWORD(v24) = sub_2D43050(v21, v17);
    v27 = 0;
    if ( (_WORD)v24 )
      goto LABEL_9;
  }
  v24 = sub_3009450(v23, v22, v52, v62, v25, v26);
  HIWORD(v3) = HIWORD(v24);
  v27 = v44;
LABEL_9:
  LOWORD(v3) = v24;
  v53 = v27;
  v28 = sub_3790540((__int64)a1, v6, v58, v63, v64, 0, a3);
  v30 = v29;
  v31 = v28;
  v32 = sub_3790540((__int64)a1, v57, v56, v3, v53, 1, a3);
  v34 = v33;
  v35 = v32;
  v36 = sub_3790540((__int64)a1, v55, v54, v63, v64, 0, a3);
  v38 = (_QWORD *)a1[1];
  v39 = v36;
  v40 = v37;
  v41 = *(_QWORD *)(a2 + 80);
  v65 = v41;
  if ( v41 )
  {
    v60 = v37;
    v59 = v36;
    sub_B96E90((__int64)&v65, v41, 1);
    v39 = v59;
    v40 = v60;
  }
  *((_QWORD *)&v49 + 1) = v40;
  *(_QWORD *)&v49 = v39;
  *((_QWORD *)&v48 + 1) = v34;
  *(_QWORD *)&v48 = v35;
  *((_QWORD *)&v47 + 1) = v30;
  *(_QWORD *)&v47 = v31;
  LODWORD(v66) = *(_DWORD *)(a2 + 72);
  v42 = sub_340F900(v38, 0xABu, (__int64)&v65, v63, v64, v40, v47, v48, v49);
  if ( v65 )
    sub_B91220((__int64)&v65, v65);
  return v42;
}
