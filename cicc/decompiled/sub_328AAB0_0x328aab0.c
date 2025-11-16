// Function: sub_328AAB0
// Address: 0x328aab0
//
__int64 __fastcall sub_328AAB0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r9
  unsigned int v6; // r13d
  int v7; // eax
  unsigned __int16 *v9; // rax
  unsigned __int16 v10; // r10
  __int64 v11; // rdx
  __int64 *v12; // rsi
  __int64 v13; // r14
  __int64 v14; // r8
  __int64 v15; // r15
  int v16; // eax
  __int64 v17; // rdi
  __int64 v18; // rax
  __int64 (*v19)(); // rax
  __int64 v20; // rax
  unsigned __int64 v21; // rdx
  __int64 v22; // r9
  __int64 v23; // r8
  __int64 v24; // rsi
  __int64 v25; // rcx
  __int64 *v26; // rdi
  unsigned int v27; // ecx
  __int64 v28; // rdx
  __int64 v29; // rdx
  unsigned __int16 v30; // si
  __int64 v31; // rdx
  __int64 v32; // rsi
  __int64 v33; // rdx
  __int64 v34; // rdx
  char v35; // al
  __int64 v36; // rdx
  unsigned __int16 v37; // r10
  __int64 v38; // r9
  char v39; // al
  unsigned __int8 v40; // r11
  int v41; // eax
  __int64 v42; // rax
  __int64 v43; // rdx
  __int64 v44; // r15
  __int64 v45; // r14
  int v46; // r9d
  __int128 v47; // rax
  int v48; // r9d
  __int128 v49; // rax
  __int64 v50; // r15
  int v51; // r9d
  __int64 v52; // rax
  unsigned int v53; // edx
  __int64 v54; // rax
  unsigned __int16 v55; // si
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rax
  __int64 v59; // rax
  unsigned __int16 v60; // dx
  __int64 v61; // rax
  __int64 v62; // rax
  char v63; // al
  __int128 v64; // [rsp-10h] [rbp-B0h]
  __int128 v65; // [rsp-10h] [rbp-B0h]
  __int128 v66; // [rsp-10h] [rbp-B0h]
  __int128 v67; // [rsp-10h] [rbp-B0h]
  __int64 v68; // [rsp+0h] [rbp-A0h]
  __int64 v69; // [rsp+8h] [rbp-98h]
  unsigned __int16 v70; // [rsp+8h] [rbp-98h]
  unsigned __int8 v71; // [rsp+10h] [rbp-90h]
  __int64 v72; // [rsp+10h] [rbp-90h]
  unsigned int v73; // [rsp+18h] [rbp-88h]
  int v74; // [rsp+18h] [rbp-88h]
  __int64 v75; // [rsp+20h] [rbp-80h]
  __int64 v76; // [rsp+20h] [rbp-80h]
  unsigned __int16 v77; // [rsp+28h] [rbp-78h]
  unsigned int v78; // [rsp+2Ch] [rbp-74h]
  unsigned __int16 v79; // [rsp+2Ch] [rbp-74h]
  __int64 v80; // [rsp+30h] [rbp-70h]
  unsigned __int16 v81; // [rsp+30h] [rbp-70h]
  __int64 v82; // [rsp+30h] [rbp-70h]
  __int64 v83; // [rsp+38h] [rbp-68h]
  __int128 v84; // [rsp+40h] [rbp-60h]
  _WORD v85[4]; // [rsp+60h] [rbp-40h] BYREF
  __int64 v86; // [rsp+68h] [rbp-38h]

  v5 = *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL);
  v6 = **(unsigned __int16 **)(a2 + 48);
  v7 = *(_DWORD *)(a2 + 24);
  if ( v7 == 216 )
  {
    a2 = **(_QWORD **)(a2 + 40);
    v7 = *(_DWORD *)(a2 + 24);
  }
  if ( v7 != 189 )
    return 0;
  v9 = *(unsigned __int16 **)(a2 + 48);
  v10 = *v9;
  v83 = *((_QWORD *)v9 + 1);
  v11 = **(_QWORD **)(a2 + 40);
  if ( *(_DWORD *)(v11 + 24) != 57 )
    return 0;
  v12 = *(__int64 **)(v11 + 40);
  v13 = v12[5];
  *(_QWORD *)&v84 = *v12;
  v14 = *v12;
  v15 = *((unsigned int *)v12 + 12);
  *((_QWORD *)&v84 + 1) = *((unsigned int *)v12 + 2);
  v16 = *(_DWORD *)(*v12 + 24);
  if ( *(_DWORD *)(v13 + 24) == v16 && ((unsigned int)(v16 - 213) <= 1 || v16 == 222) )
  {
    v86 = 0;
    v26 = *(__int64 **)(v13 + 40);
    HIWORD(v27) = 0;
    v28 = *(_QWORD *)(v14 + 40);
    if ( v16 == 222 )
    {
      v59 = *(_QWORD *)(v28 + 40);
      v78 = 178;
      v60 = *(_WORD *)(v59 + 96);
      v86 = *(_QWORD *)(v59 + 104);
      v61 = v26[5];
      v81 = v60;
      v85[0] = v60;
      LOWORD(v27) = *(_WORD *)(v61 + 96);
      v34 = *(_QWORD *)(v61 + 104);
    }
    else
    {
      v29 = *(_QWORD *)(*(_QWORD *)v28 + 48LL) + 16LL * *(unsigned int *)(v28 + 8);
      v30 = *(_WORD *)v29;
      v31 = *(_QWORD *)(v29 + 8);
      v81 = v30;
      v85[0] = v30;
      v32 = *v26;
      v86 = v31;
      v33 = *(_QWORD *)(v32 + 48) + 16LL * *((unsigned int *)v26 + 2);
      LOWORD(v27) = *(_WORD *)v33;
      v34 = *(_QWORD *)(v33 + 8);
      v78 = (v16 == 214) + 178;
    }
    v77 = v10;
    v68 = v14;
    v69 = v5;
    v73 = v27;
    v75 = v34;
    v35 = sub_3280A00((__int64)v85, v27, v34);
    v36 = v75;
    v37 = v77;
    v38 = v69;
    if ( v35 )
    {
      HIWORD(v73) = v85[1];
      v75 = v86;
      if ( v81 )
        goto LABEL_23;
    }
    else
    {
      v55 = v73;
      if ( v81 != (_WORD)v73 )
        goto LABEL_34;
      if ( v81 )
        goto LABEL_25;
    }
    if ( v86 == v75 )
      goto LABEL_23;
    v55 = 0;
LABEL_34:
    v56 = *(_QWORD *)(v68 + 56);
    if ( !v56 || *(_QWORD *)(v56 + 32) )
      goto LABEL_35;
    v81 = v55;
LABEL_23:
    if ( v81 == (_WORD)v73 && (v81 || v75 == v36) || (v62 = *(_QWORD *)(v13 + 56)) != 0 && !*(_QWORD *)(v62 + 32) )
    {
LABEL_25:
      if ( !*((_BYTE *)a1 + 34)
        || (v71 = *((_BYTE *)a1 + 33), v39 = sub_328A020(a1[1], v78, v81, v75, v71),
                                       v38 = v69,
                                       v37 = v77,
                                       v40 = v71,
                                       v39) )
      {
        *((_QWORD *)&v65 + 1) = v15;
        HIWORD(v41) = HIWORD(v73);
        LOWORD(v41) = v81;
        *(_QWORD *)&v65 = v13;
        v82 = *a1;
        v72 = v38;
        v70 = v37;
        v74 = v41;
        v42 = sub_33FAF80(*a1, 216, a3, v41, v75, v38, v65);
        v44 = v43;
        v45 = v42;
        *(_QWORD *)&v47 = sub_33FAF80(*a1, 216, a3, v74, v75, v46, v84);
        *((_QWORD *)&v66 + 1) = v44;
        *(_QWORD *)&v66 = v45;
        *(_QWORD *)&v49 = sub_3406EB0(v82, v78, a3, (_WORD)v74, v75, v48, v47, v66);
        v50 = *((_QWORD *)&v49 + 1);
        v52 = sub_33FAF80(*a1, 214, a3, v70, v83, v51, v49);
        v22 = v72;
        v23 = v6;
        v24 = v52;
        v21 = v53 | v50 & 0xFFFFFFFF00000000LL;
        v25 = a3;
        return sub_33FB310(*a1, v24, v21, v25, v23, v22);
      }
LABEL_36:
      if ( !v40
        || ((v57 = a1[1], v58 = 1, v37 == 1) || v37 && (v58 = v37, *(_QWORD *)(v57 + 8LL * v37 + 112)))
        && !*(_BYTE *)(v78 + v57 + 500 * v58 + 6414) )
      {
        *((_QWORD *)&v67 + 1) = v15;
        *(_QWORD *)&v67 = v13;
        v80 = v38;
        v20 = sub_3406EB0(*a1, v78, a3, v37, v83, v38, v84, v67);
        goto LABEL_15;
      }
      return 0;
    }
LABEL_35:
    v40 = *((_BYTE *)a1 + 33);
    goto LABEL_36;
  }
  if ( (*(_BYTE *)(v11 + 28) & 2) == 0 )
    return 0;
  v17 = a1[1];
  if ( *((_BYTE *)a1 + 33) )
  {
    v18 = 1;
    if ( v10 != 1 )
    {
      if ( !v10 )
        return 0;
      v18 = v10;
      if ( !*(_QWORD *)(v17 + 8LL * v10 + 112) )
        return 0;
    }
    if ( *(_BYTE *)(v17 + 500 * v18 + 6592) )
      return 0;
  }
  else
  {
    v54 = 1;
    if ( v10 != 1 )
    {
      if ( !v10 )
        return 0;
      v54 = v10;
      if ( !*(_QWORD *)(v17 + 8LL * v10 + 112) )
        return 0;
    }
    if ( (*(_BYTE *)(v17 + 500 * v54 + 6592) & 0xFB) != 0 )
      return 0;
  }
  v19 = *(__int64 (**)())(*(_QWORD *)v17 + 464LL);
  if ( v19 != sub_2FE30A0 )
  {
    v76 = v5;
    v79 = v10;
    v63 = ((__int64 (__fastcall *)(__int64, _QWORD, __int64))v19)(v17, v10, v83);
    v10 = v79;
    v5 = v76;
    if ( !v63 )
      return 0;
  }
  *((_QWORD *)&v64 + 1) = v15;
  *(_QWORD *)&v64 = v13;
  v80 = v5;
  v20 = sub_3406EB0(*a1, 178, a3, v10, v83, v5, v84, v64);
LABEL_15:
  v22 = v80;
  v23 = v6;
  v24 = v20;
  v25 = a3;
  return sub_33FB310(*a1, v24, v21, v25, v23, v22);
}
