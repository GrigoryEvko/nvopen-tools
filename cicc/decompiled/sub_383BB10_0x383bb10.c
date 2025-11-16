// Function: sub_383BB10
// Address: 0x383bb10
//
unsigned __int8 *__fastcall sub_383BB10(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 v5; // rsi
  __int64 v6; // rdx
  int v7; // eax
  unsigned __int8 *v8; // r14
  unsigned int v9; // edx
  __int64 v10; // r13
  unsigned int v11; // edx
  unsigned __int16 *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  _QWORD *v16; // r8
  __int64 v17; // rdi
  __int64 v18; // rsi
  unsigned int v19; // r9d
  char v20; // al
  __int64 (*v21)(); // rax
  __int64 v22; // rcx
  unsigned __int64 v23; // r14
  unsigned __int8 *v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rsi
  unsigned __int8 *v27; // r13
  unsigned __int8 *v28; // rbx
  unsigned int v30; // edx
  unsigned int v31; // edx
  __int64 v32; // rax
  _QWORD *v33; // rbx
  unsigned __int16 v34; // dx
  __int64 v35; // rax
  unsigned int v36; // eax
  __int128 v37; // rdi
  int v38; // r15d
  __int64 v39; // rax
  unsigned __int16 v40; // dx
  __int64 v41; // rax
  int v42; // eax
  __int64 v43; // r9
  _QWORD *v44; // rdi
  __int64 v45; // rax
  __int64 v46; // rax
  unsigned __int16 v47; // dx
  __int64 v48; // rax
  unsigned int v49; // eax
  char v50; // al
  __int64 v51; // rsi
  __int128 v52; // rax
  unsigned __int8 *v53; // rax
  unsigned int v54; // edx
  __int64 v55; // r9
  __int64 v56; // rax
  _QWORD *v57; // rbx
  __int64 v58; // rdx
  __int64 v59; // r13
  __int64 v60; // r12
  __int128 v61; // rax
  __int64 v62; // r9
  __int128 v63; // [rsp-30h] [rbp-110h]
  __int128 v64; // [rsp-30h] [rbp-110h]
  __int128 v65; // [rsp-20h] [rbp-100h]
  __int128 v66; // [rsp-20h] [rbp-100h]
  char v67; // [rsp+Ch] [rbp-D4h]
  unsigned int v68; // [rsp+10h] [rbp-D0h]
  unsigned int v69; // [rsp+10h] [rbp-D0h]
  bool v70; // [rsp+18h] [rbp-C8h]
  _QWORD *v71; // [rsp+18h] [rbp-C8h]
  __int128 v72; // [rsp+20h] [rbp-C0h]
  __int64 v73; // [rsp+30h] [rbp-B0h]
  __int64 v74; // [rsp+30h] [rbp-B0h]
  __int64 v75; // [rsp+38h] [rbp-A8h]
  unsigned __int64 v76; // [rsp+38h] [rbp-A8h]
  __int64 v77; // [rsp+80h] [rbp-60h] BYREF
  int v78; // [rsp+88h] [rbp-58h]
  __int64 v79; // [rsp+90h] [rbp-50h] BYREF
  __int64 v80; // [rsp+98h] [rbp-48h]
  unsigned __int16 v81; // [rsp+A0h] [rbp-40h] BYREF
  __int64 v82; // [rsp+A8h] [rbp-38h]

  v5 = *(_QWORD *)(a2 + 80);
  v77 = v5;
  if ( v5 )
    sub_B96E90((__int64)&v77, v5, 1);
  v78 = *(_DWORD *)(a2 + 72);
  v6 = *(_QWORD *)(a2 + 40);
  v7 = *(_DWORD *)(a2 + 24);
  if ( v7 == 92 )
  {
    v70 = 0;
  }
  else
  {
    if ( v7 != 94 )
    {
      v70 = v7 == 95;
      v8 = sub_37AF270((__int64)a1, *(_QWORD *)v6, *(_QWORD *)(v6 + 8), a3);
      v10 = v9;
      v75 = v9;
      v67 = 0;
      *(_QWORD *)&v72 = sub_37AF270(
                          (__int64)a1,
                          *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                          *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL),
                          a3);
      *((_QWORD *)&v72 + 1) = v11;
      goto LABEL_6;
    }
    v70 = 1;
  }
  v8 = sub_383B380((__int64)a1, *(_QWORD *)v6, *(_QWORD *)(v6 + 8));
  v10 = v30;
  v75 = v30;
  v67 = 1;
  *(_QWORD *)&v72 = sub_383B380(
                      (__int64)a1,
                      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 40LL),
                      *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL));
  *((_QWORD *)&v72 + 1) = v31;
LABEL_6:
  v12 = (unsigned __int16 *)(*((_QWORD *)v8 + 6) + 16 * v10);
  v13 = *v12;
  v80 = *((_QWORD *)v12 + 1);
  v14 = *(_QWORD *)(a2 + 40);
  LOWORD(v79) = v13;
  v15 = *(_QWORD *)(*(_QWORD *)(v14 + 80) + 96LL);
  v16 = *(_QWORD **)(v15 + 24);
  if ( *(_DWORD *)(v15 + 32) > 0x40u )
    v16 = (_QWORD *)*v16;
  v17 = *a1;
  v18 = *(unsigned int *)(a2 + 24);
  v19 = (unsigned int)v16;
  if ( !(_WORD)v13 || !*(_QWORD *)(v17 + 8 * v13 + 112) )
    goto LABEL_14;
  if ( (unsigned int)v18 > 0x1F3 )
    goto LABEL_25;
  v20 = *(_BYTE *)((unsigned int)v18 + v17 + 500LL * (unsigned __int16)v13 + 6414);
  if ( v20 )
  {
    if ( v20 != 4 )
      goto LABEL_14;
LABEL_25:
    v38 = sub_32844A0((unsigned __int16 *)&v79, v18);
    v39 = *(_QWORD *)(a2 + 48);
    v40 = *(_WORD *)v39;
    v41 = *(_QWORD *)(v39 + 8);
    v81 = v40;
    v82 = v41;
    v42 = sub_32844A0(&v81, v18);
    v44 = (_QWORD *)a1[1];
    if ( v70 )
    {
      v71 = (_QWORD *)a1[1];
      v51 = (unsigned int)(v38 - v42);
      *(_QWORD *)&v52 = sub_3400E40((__int64)v44, v51, v79, v80, (__int64)&v77, a3);
      v76 = v10 | v75 & 0xFFFFFFFF00000000LL;
      *((_QWORD *)&v65 + 1) = v76;
      *(_QWORD *)&v65 = v8;
      v53 = sub_3406EB0(v71, 0xBEu, (__int64)&v77, (unsigned int)v79, v80, *((__int64 *)&v52 + 1), v65, v52);
      *((_QWORD *)&v64 + 1) = v54 | v76 & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v64 = v53;
      v56 = sub_340F900(
              (_QWORD *)a1[1],
              *(_DWORD *)(a2 + 24),
              (__int64)&v77,
              v79,
              v80,
              v55,
              v64,
              v72,
              *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL));
      v57 = (_QWORD *)a1[1];
      v59 = v58;
      v60 = v56;
      *(_QWORD *)&v61 = sub_3400E40((__int64)v57, (unsigned int)v51, v79, v80, (__int64)&v77, a3);
      *((_QWORD *)&v66 + 1) = v59;
      *(_QWORD *)&v66 = v60;
      v45 = (__int64)sub_3406EB0(
                       v57,
                       (unsigned int)(v67 == 0) + 191,
                       (__int64)&v77,
                       (unsigned int)v79,
                       v80,
                       v62,
                       v66,
                       v61);
    }
    else
    {
      *((_QWORD *)&v63 + 1) = v10 | v75 & 0xFFFFFFFF00000000LL;
      *(_QWORD *)&v63 = v8;
      v45 = sub_340F900(
              v44,
              *(_DWORD *)(a2 + 24),
              (__int64)&v77,
              v79,
              v80,
              v43,
              v63,
              v72,
              *(_OWORD *)(*(_QWORD *)(a2 + 40) + 80LL));
    }
    v28 = (unsigned __int8 *)v45;
    goto LABEL_18;
  }
  if ( (unsigned int)(v18 - 88) > 7 )
    BUG();
  v21 = *(__int64 (**)())(*(_QWORD *)v17 + 656LL);
  if ( v21 == sub_2FE31B0 )
    goto LABEL_14;
  v69 = (unsigned int)v16;
  v50 = ((__int64 (__fastcall *)(__int64, __int64, __int64, __int64, _QWORD *, _QWORD))v21)(
          v17,
          v18,
          v79,
          v80,
          v16,
          (unsigned int)v16);
  v19 = v69;
  if ( v50 )
    goto LABEL_25;
  v17 = *a1;
  LODWORD(v18) = *(_DWORD *)(a2 + 24);
LABEL_14:
  v73 = (__int64)v8;
  v22 = (__int64)v8;
  v68 = v19;
  v23 = v10 | v75 & 0xFFFFFFFF00000000LL;
  v24 = sub_34696C0(v17, v18, (__int64)&v77, v22, v23, v19, a3, v72, a1[1]);
  v26 = v25;
  v27 = v24;
  if ( v24 )
  {
    if ( v70 )
    {
      v32 = *(_QWORD *)(a2 + 48);
      v74 = v25;
      v33 = (_QWORD *)a1[1];
      v34 = *(_WORD *)v32;
      v35 = *(_QWORD *)(v32 + 8);
      v81 = v34;
      v82 = v35;
      v36 = sub_32844A0(&v81, v26);
      *(_QWORD *)&v37 = v27;
      *((_QWORD *)&v37 + 1) = v74;
      v27 = sub_38139C0(v37, (__int64)&v77, v36, v67 & 1, v33, a3);
    }
    v28 = v27;
  }
  else
  {
    v46 = *(_QWORD *)(a2 + 48);
    v47 = *(_WORD *)v46;
    v48 = *(_QWORD *)(v46 + 8);
    v81 = v47;
    v82 = v48;
    v49 = sub_32844A0(&v81, v26);
    v28 = sub_3813E70(a2, v73, v23, v72, *((__int64 *)&v72 + 1), v68, a3, *a1, a1[1], v49);
  }
LABEL_18:
  if ( v77 )
    sub_B91220((__int64)&v77, v77);
  return v28;
}
