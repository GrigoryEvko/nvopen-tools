// Function: sub_10C1D20
// Address: 0x10c1d20
//
__int64 *__fastcall sub_10C1D20(
        __int64 *a1,
        __int64 *a2,
        __int64 *a3,
        __int64 *a4,
        __int64 *a5,
        __int64 a6,
        __int64 a7,
        int *a8,
        int *a9)
{
  __int64 v14; // r11
  __int64 v15; // r9
  __int64 v16; // r10
  char v17; // al
  __int64 *v18; // rcx
  __int64 v19; // r10
  __int64 v20; // r11
  __int64 v21; // r9
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rcx
  int v25; // edx
  __int64 v26; // r9
  __int64 v27; // rdi
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rdi
  unsigned int v31; // r14d
  __int64 v32; // rdi
  int v33; // esi
  __int64 v34; // rsi
  __int64 v35; // rdi
  __int64 v36; // rax
  __int64 v37; // r8
  _BYTE *v38; // rsi
  __int64 v39; // rsi
  char v40; // al
  __int64 v41; // rax
  __int64 v42; // rdi
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 *v45; // [rsp+0h] [rbp-C0h]
  __int64 *v46; // [rsp+0h] [rbp-C0h]
  __int64 v47; // [rsp+8h] [rbp-B8h]
  __int64 v48; // [rsp+10h] [rbp-B0h]
  __int64 v49; // [rsp+10h] [rbp-B0h]
  __int64 v50; // [rsp+10h] [rbp-B0h]
  __int64 v51; // [rsp+10h] [rbp-B0h]
  __int64 v52; // [rsp+18h] [rbp-A8h]
  __int64 v53; // [rsp+18h] [rbp-A8h]
  __int64 v54; // [rsp+18h] [rbp-A8h]
  __int64 v55; // [rsp+18h] [rbp-A8h]
  __int64 v56; // [rsp+20h] [rbp-A0h]
  __int64 v57; // [rsp+20h] [rbp-A0h]
  __int64 v58; // [rsp+20h] [rbp-A0h]
  __int64 v59; // [rsp+20h] [rbp-A0h]
  __int64 v60; // [rsp+28h] [rbp-98h]
  __int64 v61; // [rsp+28h] [rbp-98h]
  __int64 v62; // [rsp+28h] [rbp-98h]
  __int64 v65; // [rsp+50h] [rbp-70h] BYREF
  __int64 v66; // [rsp+58h] [rbp-68h] BYREF
  __int64 v67; // [rsp+60h] [rbp-60h] BYREF
  __int64 v68; // [rsp+68h] [rbp-58h] BYREF
  __int64 v69; // [rsp+70h] [rbp-50h] BYREF
  _BYTE *v70; // [rsp+78h] [rbp-48h] BYREF
  __int64 *v71; // [rsp+80h] [rbp-40h] BYREF
  __int64 *v72; // [rsp+88h] [rbp-38h]

  if ( (unsigned __int8)sub_10B8AD0(a6, a8, (__int64)&v65, &v66, &v67) )
  {
    v14 = 0;
    v15 = 0;
    v16 = 0;
  }
  else
  {
    if ( *(_BYTE *)a6 != 82 )
      goto LABEL_3;
    v24 = *(_QWORD *)(*(_QWORD *)(a6 - 64) + 8LL);
    v25 = *(unsigned __int8 *)(v24 + 8);
    if ( (unsigned int)(v25 - 17) <= 1 )
      LOBYTE(v25) = *(_BYTE *)(**(_QWORD **)(v24 + 16) + 8LL);
    if ( (_BYTE)v25 != 12 )
      goto LABEL_3;
    *a8 = *(_WORD *)(a6 + 2) & 0x3F;
    v16 = *(_QWORD *)(a6 - 64);
    v26 = *(_QWORD *)(a6 - 32);
    v67 = v26;
    if ( *(_BYTE *)v16 == 57 && *(_QWORD *)(v16 - 64) && (v65 = *(_QWORD *)(v16 - 64), *(_QWORD *)(v16 - 32)) )
    {
      v66 = *(_QWORD *)(v16 - 32);
    }
    else
    {
      v27 = *(_QWORD *)(v16 + 8);
      v65 = v16;
      v60 = v16;
      v28 = sub_AD62B0(v27);
      v26 = v67;
      v16 = v60;
      v66 = v28;
    }
    if ( *(_BYTE *)v26 == 57 && *(_QWORD *)(v26 - 64) && (v14 = *(_QWORD *)(v26 - 32)) != 0 )
    {
      v15 = *(_QWORD *)(v26 - 64);
    }
    else
    {
      v56 = v16;
      v61 = v26;
      v29 = sub_AD62B0(*(_QWORD *)(v26 + 8));
      v15 = v61;
      v16 = v56;
      v14 = v29;
    }
  }
  if ( (unsigned int)(*a8 - 32) > 1 )
  {
LABEL_3:
    LOBYTE(v72) = 0;
    return v71;
  }
  v47 = v15;
  v48 = v14;
  v52 = v16;
  v17 = sub_10B8AD0(a7, a9, (__int64)&v68, &v69, (__int64 *)&v70);
  v18 = &v69;
  v19 = v52;
  v20 = v48;
  v21 = v47;
  if ( !v17 )
  {
    if ( *(_BYTE *)a7 != 82 )
      goto LABEL_3;
    v32 = *(_QWORD *)(*(_QWORD *)(a7 - 64) + 8LL);
    v33 = *(unsigned __int8 *)(v32 + 8);
    if ( (unsigned int)(v33 - 17) <= 1 )
      LOBYTE(v33) = *(_BYTE *)(**(_QWORD **)(v32 + 16) + 8LL);
    if ( (_BYTE)v33 != 12 )
      goto LABEL_3;
    v71 = &v68;
    v72 = &v69;
    *a9 = *(_WORD *)(a7 + 2) & 0x3F;
    v34 = *(_QWORD *)(a7 - 64);
    v70 = *(_BYTE **)(a7 - 32);
    v62 = v34;
    if ( *(_BYTE *)v34 == 57 && (unsigned __int8)sub_10B8310(&v71, v34) )
    {
      v36 = v69;
    }
    else
    {
      v45 = v18;
      v35 = *(_QWORD *)(v34 + 8);
      v49 = v21;
      v53 = v20;
      v57 = v19;
      v68 = v34;
      v36 = sub_AD62B0(v35);
      v18 = v45;
      v69 = v36;
      v21 = v49;
      v20 = v53;
      v19 = v57;
    }
    v37 = v68;
    v38 = v70;
    if ( v68 == v65 || v68 == v66 || v21 == v68 || v20 == v68 )
    {
      *a1 = v68;
      *a4 = v36;
      *a5 = (__int64)v38;
    }
    else
    {
      if ( v21 != v36 && v66 != v36 && v65 != v36 && v20 != v36 )
        goto LABEL_46;
      *a1 = v36;
      *a4 = v37;
      *a5 = (__int64)v38;
    }
    v39 = *a1;
    v50 = v21;
    v54 = v20;
    v58 = v19;
    v46 = v18;
    v71 = 0;
    v40 = sub_995B10(&v71, v39);
    v19 = v58;
    v20 = v54;
    v21 = v50;
    if ( !v40 )
      goto LABEL_24;
    v38 = v70;
    v18 = v46;
LABEL_46:
    v71 = &v68;
    v41 = (__int64)v38;
    v72 = v18;
    if ( *v38 == 57 )
    {
      if ( (unsigned __int8)sub_10B8310(&v71, (__int64)v38) )
      {
        v43 = v69;
        goto LABEL_48;
      }
      v41 = (__int64)v70;
    }
    v42 = *(_QWORD *)(v41 + 8);
    v51 = v21;
    v55 = v20;
    v59 = v19;
    v68 = v41;
    v43 = sub_AD62B0(v42);
    v21 = v51;
    v20 = v55;
    v69 = v43;
    v19 = v59;
LABEL_48:
    v44 = v68;
    if ( v68 == v65 || v68 == v66 || v21 == v68 || v20 == v68 )
    {
      *a1 = v68;
      *a4 = v43;
      *a5 = v62;
    }
    else
    {
      if ( v21 != v43 && v66 != v43 && v65 != v43 && v20 != v43 )
        goto LABEL_3;
      *a1 = v43;
      *a4 = v44;
      *a5 = v62;
    }
    goto LABEL_24;
  }
  v22 = v68;
  v23 = v69;
  if ( v68 == v65 || v68 == v66 || v68 == v47 || v48 == v68 )
  {
    *a1 = v68;
    *a4 = v23;
  }
  else
  {
    if ( v47 != v69 && v66 != v69 && v65 != v69 && v48 != v69 )
      goto LABEL_3;
    *a1 = v69;
    *a4 = v22;
  }
  *a5 = (__int64)v70;
LABEL_24:
  if ( (unsigned int)(*a9 - 32) > 1 )
    goto LABEL_3;
  v30 = *a1;
  if ( *a1 == v65 )
  {
    v19 = v67;
    *a2 = v66;
    *a3 = v19;
    v30 = *a1;
  }
  else if ( v30 == v66 )
  {
    v19 = v67;
    *a2 = v65;
    *a3 = v19;
    v30 = *a1;
  }
  else if ( v21 == v30 )
  {
    *a2 = v20;
    *a3 = v19;
    v30 = *a1;
  }
  else if ( v20 == v30 )
  {
    *a2 = v21;
    *a3 = v19;
    v30 = *a1;
  }
  else
  {
    v19 = *a3;
  }
  v31 = sub_10B9C90(v30, *a2, v19, *a8);
  v71 = (__int64 *)__PAIR64__(sub_10B9C90(*a1, *a4, *a5, *a9), v31);
  LOBYTE(v72) = 1;
  return v71;
}
