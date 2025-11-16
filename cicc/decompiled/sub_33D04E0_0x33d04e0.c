// Function: sub_33D04E0
// Address: 0x33d04e0
//
__int64 __fastcall sub_33D04E0(__int64 a1, __int64 a2, unsigned __int16 *a3, unsigned __int16 *a4, _BYTE *a5)
{
  unsigned __int16 v7; // ax
  __int64 v8; // rdx
  unsigned __int16 v9; // r14
  unsigned __int16 v10; // ax
  bool v11; // cf
  bool v12; // zf
  int v13; // eax
  unsigned int v14; // r9d
  char v15; // r11
  unsigned __int64 v16; // r10
  unsigned int v17; // r15d
  __int16 v18; // ax
  __int64 v19; // r8
  unsigned int v20; // edx
  __int64 v21; // r9
  char v22; // r11
  char v23; // r10
  __int16 v24; // bx
  int v25; // esi
  __int64 *v26; // r13
  __int16 v27; // ax
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // rcx
  __int64 v32; // rdx
  __int64 v33; // rax
  unsigned __int64 v34; // rax
  __int16 v35; // ax
  __int64 v36; // r8
  __int64 v37; // r9
  unsigned int v38; // edx
  char v39; // r11
  char v40; // r10
  __int64 v41; // r8
  __int64 v42; // r9
  __int16 v44; // ax
  __int16 v45; // ax
  __int64 v46; // rdx
  __int16 v47; // ax
  __int16 v48; // ax
  __int64 v49; // rdx
  __int64 v50; // rdx
  __int64 v51; // rdx
  unsigned int v52; // [rsp+0h] [rbp-80h]
  int v53; // [rsp+0h] [rbp-80h]
  char v54; // [rsp+6h] [rbp-7Ah]
  char v55; // [rsp+6h] [rbp-7Ah]
  char v56; // [rsp+7h] [rbp-79h]
  char v57; // [rsp+7h] [rbp-79h]
  char v58; // [rsp+7h] [rbp-79h]
  char v59; // [rsp+7h] [rbp-79h]
  __int64 *v60; // [rsp+8h] [rbp-78h]
  char v61; // [rsp+10h] [rbp-70h]
  char v62; // [rsp+10h] [rbp-70h]
  __int64 *v63; // [rsp+10h] [rbp-70h]
  char v64; // [rsp+10h] [rbp-70h]
  __int64 v65; // [rsp+18h] [rbp-68h]
  __int64 v67; // [rsp+28h] [rbp-58h]
  unsigned int v68; // [rsp+28h] [rbp-58h]
  __int64 v69; // [rsp+48h] [rbp-38h]

  v7 = *a3;
  if ( *a3 )
  {
    v65 = 0;
    v8 = v7 - 1;
    v9 = word_4456580[v8];
  }
  else
  {
    v9 = sub_3009970((__int64)a3, a2, (__int64)a3, (__int64)a4, (__int64)a5);
    v7 = *a3;
    v65 = v32;
    if ( !*a3 )
    {
      v33 = sub_3007240((__int64)a3);
      v14 = v33;
      v61 = BYTE4(v33);
      v13 = *a4;
      if ( (_WORD)v13 )
        goto LABEL_4;
      goto LABEL_15;
    }
    v8 = v7 - 1;
  }
  v10 = v7 - 176;
  v11 = v10 < 0x34u;
  v12 = v10 == 52;
  v13 = *a4;
  v14 = word_4456340[v8];
  v61 = v11 || v12;
  if ( (_WORD)v13 )
  {
LABEL_4:
    v15 = (unsigned __int16)(v13 - 176) <= 0x34u;
    LOBYTE(v16) = v15;
    v17 = word_4456340[v13 - 1];
    goto LABEL_5;
  }
LABEL_15:
  v68 = v14;
  v34 = sub_3007240((__int64)a4);
  v14 = v68;
  v17 = v34;
  v16 = HIDWORD(v34);
  v15 = BYTE4(v34);
LABEL_5:
  v60 = *(__int64 **)(a2 + 64);
  if ( v17 >= v14 )
  {
    LODWORD(v69) = v14;
    v57 = v15;
    BYTE4(v69) = v61;
    v62 = v16;
    if ( BYTE4(v69) )
    {
      v44 = sub_2D43AD0(v9, v14);
      v38 = v9;
      v40 = v62;
      v67 = 0;
      v39 = v57;
      v24 = v44;
      if ( v44 )
        goto LABEL_18;
    }
    else
    {
      v35 = sub_2D43050(v9, v14);
      v38 = v9;
      v39 = v57;
      v67 = 0;
      v40 = v62;
      v24 = v35;
      if ( v35 )
        goto LABEL_18;
    }
    v58 = v39;
    v64 = v40;
    v45 = sub_3009450(v60, v38, v65, v69, v36, v37);
    v39 = v58;
    v40 = v64;
    v67 = v46;
    v24 = v45;
LABEL_18:
    LODWORD(v69) = v17;
    BYTE4(v69) = v40;
    v63 = *(__int64 **)(a2 + 64);
    if ( v39 )
    {
      v27 = sub_2D43AD0(v9, v17);
      v30 = 0;
      if ( v27 )
      {
LABEL_20:
        *a5 = 1;
        goto LABEL_21;
      }
    }
    else
    {
      v27 = sub_2D43050(v9, v17);
      v30 = 0;
      if ( v27 )
        goto LABEL_20;
    }
    v27 = sub_3009450(v63, v9, v65, v69, v41, v42);
    v30 = v51;
    goto LABEL_20;
  }
  LODWORD(v69) = v17;
  BYTE4(v69) = v16;
  v52 = v14;
  v54 = v15;
  v56 = v16;
  if ( v15 )
  {
    v47 = sub_2D43AD0(v9, v17);
    v20 = v9;
    v23 = v56;
    v67 = 0;
    v22 = v54;
    v21 = v52;
    v24 = v47;
    if ( v47 )
      goto LABEL_8;
  }
  else
  {
    v18 = sub_2D43050(v9, v17);
    v20 = v9;
    v21 = v52;
    v67 = 0;
    v22 = 0;
    v23 = v56;
    v24 = v18;
    if ( v18 )
      goto LABEL_8;
  }
  v53 = v21;
  v55 = v22;
  v59 = v23;
  v48 = sub_3009450(v60, v20, v65, v69, v19, v21);
  LODWORD(v21) = v53;
  v22 = v55;
  v67 = v49;
  v23 = v59;
  v24 = v48;
LABEL_8:
  v25 = v21 - v17;
  if ( !v17 )
  {
    v22 = v61;
    v23 = v61;
  }
  LODWORD(v69) = v21 - v17;
  v26 = *(__int64 **)(a2 + 64);
  BYTE4(v69) = v23;
  if ( !v22 )
  {
    v27 = sub_2D43050(v9, v25);
    v30 = 0;
    if ( v27 )
      goto LABEL_12;
LABEL_27:
    v27 = sub_3009450(v26, v9, v65, v69, v28, v29);
    v30 = v50;
    goto LABEL_12;
  }
  v27 = sub_2D43AD0(v9, v25);
  v30 = 0;
  if ( !v27 )
    goto LABEL_27;
LABEL_12:
  *a5 = 0;
LABEL_21:
  *(_WORD *)a1 = v24;
  *(_WORD *)(a1 + 16) = v27;
  *(_QWORD *)(a1 + 8) = v67;
  *(_QWORD *)(a1 + 24) = v30;
  return a1;
}
