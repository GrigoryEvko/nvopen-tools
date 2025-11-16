// Function: sub_1C75B60
// Address: 0x1c75b60
//
__int64 __fastcall sub_1C75B60(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8,
        __int64 a9)
{
  int v13; // edi
  char v14; // al
  _QWORD *v15; // rax
  __int64 v16; // rcx
  unsigned __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // r15
  int v20; // edi
  char v21; // al
  _QWORD *v22; // rax
  __int64 v23; // rcx
  unsigned __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // r15
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rcx
  unsigned __int64 v31; // rdx
  __int64 v32; // rdx
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 result; // rax
  __int64 v36; // rdx
  __int64 v37; // rcx
  unsigned __int64 v38; // rdx
  __int64 v39; // rdx
  __int64 v40; // rdx
  char v41; // al
  _QWORD *v42; // rax
  __int64 v43; // rcx
  unsigned __int64 v44; // rdx
  __int64 v45; // rdx
  __int64 v46; // rdx
  __int64 v47; // r15
  char v48; // al
  _QWORD *v49; // rax
  __int64 v50; // rcx
  unsigned __int64 v51; // rdx
  __int64 v52; // rdx
  __int64 v53; // rcx
  __int64 v54; // rdi
  unsigned __int64 v55; // rdx
  __int64 v56; // rdx
  __int64 v57; // rcx
  __int64 v58; // rdi
  unsigned __int64 v59; // rdx
  __int64 v60; // rdx
  __int64 v61; // [rsp+0h] [rbp-40h]
  __int64 v62; // [rsp+8h] [rbp-38h]
  __int64 v63; // [rsp+8h] [rbp-38h]

  v62 = *(_QWORD *)(sub_157EBA0(a1) - 72);
  v61 = v62 + 24;
  v13 = *(unsigned __int16 *)(v62 + 18);
  if ( a6 == *(_QWORD *)(v62 - 48) )
  {
    v48 = sub_15FF7F0(v13 & 0xFFFF7FFF);
    v49 = sub_1C75090(a7, a6, v48, v61);
    if ( *(_QWORD *)(v62 - 48) )
    {
      v50 = *(_QWORD *)(v62 - 40);
      v51 = *(_QWORD *)(v62 - 32) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v51 = v50;
      if ( v50 )
        *(_QWORD *)(v50 + 16) = *(_QWORD *)(v50 + 16) & 3LL | v51;
    }
    *(_QWORD *)(v62 - 48) = v49;
    if ( v49 )
    {
      v52 = v49[1];
      *(_QWORD *)(v62 - 40) = v52;
      if ( v52 )
        *(_QWORD *)(v52 + 16) = (v62 - 40) | *(_QWORD *)(v52 + 16) & 3LL;
      *(_QWORD *)(v62 - 48 + 16) = (unsigned __int64)(v49 + 1) | *(_QWORD *)(v62 - 32) & 3LL;
      v49[1] = v62 - 48;
    }
  }
  else
  {
    v14 = sub_15FF7F0(v13 & 0xFFFF7FFF);
    v15 = sub_1C75090(a7, a6, v14, v61);
    if ( *(_QWORD *)(v62 - 24) )
    {
      v16 = *(_QWORD *)(v62 - 16);
      v17 = *(_QWORD *)(v62 - 8) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v17 = v16;
      if ( v16 )
        *(_QWORD *)(v16 + 16) = *(_QWORD *)(v16 + 16) & 3LL | v17;
    }
    *(_QWORD *)(v62 - 24) = v15;
    if ( v15 )
    {
      v18 = v15[1];
      *(_QWORD *)(v62 - 16) = v18;
      if ( v18 )
        *(_QWORD *)(v18 + 16) = (v62 - 16) | *(_QWORD *)(v18 + 16) & 3LL;
      *(_QWORD *)(v62 - 24 + 16) = (unsigned __int64)(v15 + 1) | *(_QWORD *)(v62 - 8) & 3LL;
      v15[1] = v62 - 24;
    }
  }
  v19 = *(_QWORD *)(sub_157EBA0(a2) - 72);
  v20 = *(unsigned __int16 *)(v19 + 18);
  v63 = v19 + 24;
  if ( a6 == *(_QWORD *)(v19 - 48) )
  {
    v41 = sub_15FF7F0(v20 & 0xFFFF7FFF);
    v42 = sub_1C75090(a7, a6, v41, v63);
    if ( *(_QWORD *)(v19 - 48) )
    {
      v43 = *(_QWORD *)(v19 - 40);
      v44 = *(_QWORD *)(v19 - 32) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v44 = v43;
      if ( v43 )
        *(_QWORD *)(v43 + 16) = *(_QWORD *)(v43 + 16) & 3LL | v44;
    }
    *(_QWORD *)(v19 - 48) = v42;
    if ( v42 )
    {
      v45 = v42[1];
      *(_QWORD *)(v19 - 40) = v45;
      if ( v45 )
        *(_QWORD *)(v45 + 16) = (v19 - 40) | *(_QWORD *)(v45 + 16) & 3LL;
      v46 = *(_QWORD *)(v19 - 32);
      v47 = v19 - 48;
      *(_QWORD *)(v47 + 16) = (unsigned __int64)(v42 + 1) | v46 & 3;
      v42[1] = v47;
    }
  }
  else
  {
    v21 = sub_15FF7F0(v20 & 0xFFFF7FFF);
    v22 = sub_1C75090(a7, a6, v21, v63);
    if ( *(_QWORD *)(v19 - 24) )
    {
      v23 = *(_QWORD *)(v19 - 16);
      v24 = *(_QWORD *)(v19 - 8) & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v24 = v23;
      if ( v23 )
        *(_QWORD *)(v23 + 16) = *(_QWORD *)(v23 + 16) & 3LL | v24;
    }
    *(_QWORD *)(v19 - 24) = v22;
    if ( v22 )
    {
      v25 = v22[1];
      *(_QWORD *)(v19 - 16) = v25;
      if ( v25 )
        *(_QWORD *)(v25 + 16) = (v19 - 16) | *(_QWORD *)(v25 + 16) & 3LL;
      v26 = *(_QWORD *)(v19 - 8);
      v27 = v19 - 24;
      *(_QWORD *)(v27 + 16) = (unsigned __int64)(v22 + 1) | v26 & 3;
      v22[1] = v27;
    }
  }
  v28 = *(_QWORD *)(sub_157EBA0(a3) - 72);
  v29 = *(_QWORD *)(v28 - 48);
  if ( v29 )
  {
    if ( a5 != v29 )
      goto LABEL_19;
    v53 = *(_QWORD *)(v28 - 40);
    v54 = v28 - 48;
    v55 = *(_QWORD *)(v28 - 32) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v55 = v53;
    if ( v53 )
      *(_QWORD *)(v53 + 16) = *(_QWORD *)(v53 + 16) & 3LL | v55;
  }
  else
  {
    if ( a5 )
    {
LABEL_19:
      if ( *(_QWORD *)(v28 - 24) )
      {
        v30 = *(_QWORD *)(v28 - 16);
        v31 = *(_QWORD *)(v28 - 8) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v31 = v30;
        if ( v30 )
          *(_QWORD *)(v30 + 16) = *(_QWORD *)(v30 + 16) & 3LL | v31;
      }
      *(_QWORD *)(v28 - 24) = a8;
      if ( a8 )
      {
        v32 = *(_QWORD *)(a8 + 8);
        *(_QWORD *)(v28 - 16) = v32;
        if ( v32 )
          *(_QWORD *)(v32 + 16) = (v28 - 16) | *(_QWORD *)(v32 + 16) & 3LL;
        v33 = *(_QWORD *)(v28 - 8);
        v34 = v28 - 24;
        *(_QWORD *)(v34 + 16) = (a8 + 8) | v33 & 3;
        *(_QWORD *)(a8 + 8) = v34;
      }
      goto LABEL_26;
    }
    v54 = v28 - 48;
  }
  *(_QWORD *)(v28 - 48) = a8;
  if ( a8 )
  {
    v56 = *(_QWORD *)(a8 + 8);
    *(_QWORD *)(v28 - 40) = v56;
    if ( v56 )
      *(_QWORD *)(v56 + 16) = (v28 - 40) | *(_QWORD *)(v56 + 16) & 3LL;
    *(_QWORD *)(v28 - 32) = (a8 + 8) | *(_QWORD *)(v28 - 32) & 3LL;
    *(_QWORD *)(a8 + 8) = v54;
  }
LABEL_26:
  result = *(_QWORD *)(sub_157EBA0(a4) - 72);
  v36 = *(_QWORD *)(result - 48);
  if ( v36 )
  {
    if ( a5 != v36 )
      goto LABEL_28;
    v57 = *(_QWORD *)(result - 40);
    v58 = result - 48;
    v59 = *(_QWORD *)(result - 32) & 0xFFFFFFFFFFFFFFFCLL;
    *(_QWORD *)v59 = v57;
    if ( v57 )
      *(_QWORD *)(v57 + 16) = *(_QWORD *)(v57 + 16) & 3LL | v59;
  }
  else
  {
    if ( a5 )
    {
LABEL_28:
      if ( *(_QWORD *)(result - 24) )
      {
        v37 = *(_QWORD *)(result - 16);
        v38 = *(_QWORD *)(result - 8) & 0xFFFFFFFFFFFFFFFCLL;
        *(_QWORD *)v38 = v37;
        if ( v37 )
          *(_QWORD *)(v37 + 16) = *(_QWORD *)(v37 + 16) & 3LL | v38;
      }
      *(_QWORD *)(result - 24) = a9;
      if ( a9 )
      {
        v39 = *(_QWORD *)(a9 + 8);
        *(_QWORD *)(result - 16) = v39;
        if ( v39 )
          *(_QWORD *)(v39 + 16) = (result - 16) | *(_QWORD *)(v39 + 16) & 3LL;
        v40 = *(_QWORD *)(result - 8);
        result -= 24;
        *(_QWORD *)(result + 16) = (a9 + 8) | v40 & 3;
        *(_QWORD *)(a9 + 8) = result;
      }
      return result;
    }
    v58 = result - 48;
  }
  *(_QWORD *)(result - 48) = a9;
  if ( a9 )
  {
    v60 = *(_QWORD *)(a9 + 8);
    *(_QWORD *)(result - 40) = v60;
    if ( v60 )
      *(_QWORD *)(v60 + 16) = (result - 40) | *(_QWORD *)(v60 + 16) & 3LL;
    *(_QWORD *)(result - 32) = (a9 + 8) | *(_QWORD *)(result - 32) & 3LL;
    *(_QWORD *)(a9 + 8) = v58;
    return a9;
  }
  return result;
}
