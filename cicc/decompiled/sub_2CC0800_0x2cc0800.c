// Function: sub_2CC0800
// Address: 0x2cc0800
//
__int64 __fastcall sub_2CC0800(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned __int8 *a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  _QWORD *v9; // rdi
  unsigned __int64 v10; // rax
  __int64 v12; // r10
  __int16 v15; // di
  char v16; // al
  __int64 v17; // r9
  __int64 v18; // rax
  __int64 v19; // r11
  __int64 v20; // rdx
  __int64 v21; // rax
  __int64 v22; // rax
  unsigned __int64 v23; // rax
  __int64 v24; // r10
  __int16 v25; // di
  char v26; // al
  __int64 v27; // r9
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rdx
  unsigned __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rdx
  __int64 v35; // rdx
  unsigned __int64 v36; // rax
  __int64 result; // rax
  __int64 v38; // rdx
  __int64 v39; // rdx
  __int64 v40; // rdx
  char v41; // al
  __int64 v42; // r9
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rdx
  char v46; // al
  __int64 v47; // r9
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rdx
  __int64 v51; // rdx
  __int64 v52; // rcx
  __int64 v53; // rdx
  __int64 v54; // rdx
  __int64 v55; // rcx
  __int64 v56; // rdx
  __int64 v57; // [rsp+8h] [rbp-48h]
  __int64 v58; // [rsp+8h] [rbp-48h]
  __int64 v59; // [rsp+10h] [rbp-40h]
  __int64 v60; // [rsp+10h] [rbp-40h]

  v9 = (_QWORD *)(a1 + 48);
  v10 = *v9 & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_QWORD *)v10 == v9 )
    goto LABEL_78;
  if ( !v10 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v10 - 24) - 30 > 0xA )
LABEL_78:
    BUG();
  v12 = *(_QWORD *)(v10 - 120);
  v59 = v12;
  v57 = v12 + 24;
  v15 = *(_WORD *)(v12 + 2);
  if ( a6 == *(unsigned __int8 **)(v12 - 64) )
  {
    v46 = sub_B532B0(v15 & 0x3F);
    v48 = sub_2CC0510(a7, a6, v46, v57, 0, v47);
    v19 = a2;
    if ( *(_QWORD *)(v59 - 64) )
    {
      v49 = *(_QWORD *)(v59 - 56);
      **(_QWORD **)(v59 - 48) = v49;
      if ( v49 )
        *(_QWORD *)(v49 + 16) = *(_QWORD *)(v59 - 48);
    }
    *(_QWORD *)(v59 - 64) = v48;
    if ( v48 )
    {
      v50 = *(_QWORD *)(v48 + 16);
      *(_QWORD *)(v59 - 56) = v50;
      if ( v50 )
        *(_QWORD *)(v50 + 16) = v59 - 56;
      *(_QWORD *)(v59 - 48) = v48 + 16;
      *(_QWORD *)(v48 + 16) = v59 - 64;
    }
  }
  else
  {
    v16 = sub_B532B0(v15 & 0x3F);
    v18 = sub_2CC0510(a7, a6, v16, v57, 0, v17);
    v19 = a2;
    v20 = v18;
    if ( *(_QWORD *)(v59 - 32) )
    {
      v21 = *(_QWORD *)(v59 - 24);
      **(_QWORD **)(v59 - 16) = v21;
      if ( v21 )
        *(_QWORD *)(v21 + 16) = *(_QWORD *)(v59 - 16);
    }
    *(_QWORD *)(v59 - 32) = v20;
    if ( v20 )
    {
      v22 = *(_QWORD *)(v20 + 16);
      *(_QWORD *)(v59 - 24) = v22;
      if ( v22 )
        *(_QWORD *)(v22 + 16) = v59 - 24;
      *(_QWORD *)(v59 - 16) = v20 + 16;
      *(_QWORD *)(v20 + 16) = v59 - 32;
    }
  }
  v23 = *(_QWORD *)(v19 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v23 == v19 + 48 )
    goto LABEL_82;
  if ( !v23 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v23 - 24) - 30 > 0xA )
LABEL_82:
    BUG();
  v24 = *(_QWORD *)(v23 - 120);
  v60 = v24;
  v58 = v24 + 24;
  v25 = *(_WORD *)(v24 + 2);
  if ( a6 == *(unsigned __int8 **)(v24 - 64) )
  {
    v41 = sub_B532B0(v25 & 0x3F);
    v43 = sub_2CC0510(a7, a6, v41, v58, 0, v42);
    if ( *(_QWORD *)(v60 - 64) )
    {
      v44 = *(_QWORD *)(v60 - 56);
      **(_QWORD **)(v60 - 48) = v44;
      if ( v44 )
        *(_QWORD *)(v44 + 16) = *(_QWORD *)(v60 - 48);
    }
    *(_QWORD *)(v60 - 64) = v43;
    if ( v43 )
    {
      v45 = *(_QWORD *)(v43 + 16);
      *(_QWORD *)(v60 - 56) = v45;
      if ( v45 )
        *(_QWORD *)(v45 + 16) = v60 - 56;
      *(_QWORD *)(v60 - 48) = v43 + 16;
      *(_QWORD *)(v43 + 16) = v60 - 64;
    }
  }
  else
  {
    v26 = sub_B532B0(v25 & 0x3F);
    v28 = sub_2CC0510(a7, a6, v26, v58, 0, v27);
    if ( *(_QWORD *)(v60 - 32) )
    {
      v29 = *(_QWORD *)(v60 - 24);
      **(_QWORD **)(v60 - 16) = v29;
      if ( v29 )
        *(_QWORD *)(v29 + 16) = *(_QWORD *)(v60 - 16);
    }
    *(_QWORD *)(v60 - 32) = v28;
    if ( v28 )
    {
      v30 = *(_QWORD *)(v28 + 16);
      *(_QWORD *)(v60 - 24) = v30;
      if ( v30 )
        *(_QWORD *)(v30 + 16) = v60 - 24;
      *(_QWORD *)(v60 - 16) = v28 + 16;
      *(_QWORD *)(v28 + 16) = v60 - 32;
    }
  }
  v31 = *(_QWORD *)(a3 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v31 == a3 + 48 )
    goto LABEL_84;
  if ( !v31 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v31 - 24) - 30 > 0xA )
LABEL_84:
    BUG();
  v32 = *(_QWORD *)(v31 - 120);
  v33 = *(_QWORD *)(v32 - 64);
  if ( v33 )
  {
    if ( a5 != v33 )
      goto LABEL_28;
    v51 = *(_QWORD *)(v32 - 56);
    v52 = v32 - 64;
    **(_QWORD **)(v32 - 48) = v51;
    if ( v51 )
      *(_QWORD *)(v51 + 16) = *(_QWORD *)(v32 - 48);
  }
  else
  {
    v52 = v32 - 64;
    if ( a5 )
    {
LABEL_28:
      if ( *(_QWORD *)(v32 - 32) )
      {
        v34 = *(_QWORD *)(v32 - 24);
        **(_QWORD **)(v32 - 16) = v34;
        if ( v34 )
          *(_QWORD *)(v34 + 16) = *(_QWORD *)(v32 - 16);
      }
      *(_QWORD *)(v32 - 32) = a8;
      if ( a8 )
      {
        v35 = *(_QWORD *)(a8 + 16);
        *(_QWORD *)(v32 - 24) = v35;
        if ( v35 )
          *(_QWORD *)(v35 + 16) = v32 - 24;
        *(_QWORD *)(v32 - 16) = a8 + 16;
        *(_QWORD *)(a8 + 16) = v32 - 32;
      }
      goto LABEL_35;
    }
  }
  *(_QWORD *)(v32 - 64) = a8;
  if ( a8 )
  {
    v53 = *(_QWORD *)(a8 + 16);
    *(_QWORD *)(v32 - 56) = v53;
    if ( v53 )
      *(_QWORD *)(v53 + 16) = v32 - 56;
    *(_QWORD *)(v32 - 48) = a8 + 16;
    *(_QWORD *)(a8 + 16) = v52;
  }
LABEL_35:
  v36 = *(_QWORD *)(a4 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v36 == a4 + 48 )
    goto LABEL_80;
  if ( !v36 )
    BUG();
  if ( (unsigned int)*(unsigned __int8 *)(v36 - 24) - 30 > 0xA )
LABEL_80:
    BUG();
  result = *(_QWORD *)(v36 - 120);
  v38 = *(_QWORD *)(result - 64);
  if ( v38 )
  {
    if ( a5 != v38 )
      goto LABEL_40;
    v54 = *(_QWORD *)(result - 56);
    v55 = result - 64;
    **(_QWORD **)(result - 48) = v54;
    if ( v54 )
      *(_QWORD *)(v54 + 16) = *(_QWORD *)(result - 48);
  }
  else
  {
    v55 = result - 64;
    if ( a5 )
    {
LABEL_40:
      if ( *(_QWORD *)(result - 32) )
      {
        v39 = *(_QWORD *)(result - 24);
        **(_QWORD **)(result - 16) = v39;
        if ( v39 )
          *(_QWORD *)(v39 + 16) = *(_QWORD *)(result - 16);
      }
      *(_QWORD *)(result - 32) = a9;
      if ( a9 )
      {
        v40 = *(_QWORD *)(a9 + 16);
        *(_QWORD *)(result - 24) = v40;
        if ( v40 )
          *(_QWORD *)(v40 + 16) = result - 24;
        *(_QWORD *)(result - 16) = a9 + 16;
        result -= 32;
        *(_QWORD *)(a9 + 16) = result;
      }
      return result;
    }
  }
  *(_QWORD *)(result - 64) = a9;
  if ( a9 )
  {
    v56 = *(_QWORD *)(a9 + 16);
    *(_QWORD *)(result - 56) = v56;
    if ( v56 )
      *(_QWORD *)(v56 + 16) = result - 56;
    *(_QWORD *)(result - 48) = a9 + 16;
    *(_QWORD *)(a9 + 16) = v55;
  }
  return result;
}
