// Function: sub_7C8F90
// Address: 0x7c8f90
//
__int64 __fastcall sub_7C8F90(
        unsigned __int64 a1,
        unsigned int *a2,
        _DWORD *a3,
        _DWORD *a4,
        _DWORD *a5,
        __int64 a6,
        _QWORD *a7)
{
  _DWORD *v9; // rbx
  unsigned __int16 v10; // r14
  unsigned int v11; // r13d
  unsigned __int16 v13; // ax
  __int64 v14; // rdx
  _DWORD *v15; // rcx
  unsigned int v16; // ebx
  int v17; // eax
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  unsigned __int64 v22; // rdi
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  char v27; // al
  __int64 v28; // rdx
  __int64 v29; // rcx
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // r9
  __int64 v44; // rdx
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // r9
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // r9
  unsigned __int16 v52; // ax
  __int64 v53; // rdx
  __int64 v54; // rcx
  __int64 v55; // r8
  __int64 v56; // r9
  __int64 v57; // rdx
  __int64 v58; // rcx
  __int64 v59; // r8
  __int64 v60; // r9
  __int64 v61; // rdx
  __int64 v62; // rcx
  __int64 v63; // r8
  __int64 v64; // r9
  __int64 v65; // rdx
  __int64 v66; // rcx
  __int64 v67; // r8
  __int64 v68; // r9
  unsigned __int16 v69; // [rsp+6h] [rbp-1CAh]
  _DWORD *v70; // [rsp+8h] [rbp-1C8h]
  unsigned int v71; // [rsp+10h] [rbp-1C0h]
  int v72; // [rsp+14h] [rbp-1BCh]
  int v74; // [rsp+2Ch] [rbp-1A4h] BYREF
  unsigned int v75[88]; // [rsp+30h] [rbp-1A0h] BYREF
  int v76; // [rsp+190h] [rbp-40h]
  __int16 v77; // [rsp+194h] [rbp-3Ch]

  v9 = a5;
  v72 = dword_4F04D80;
  dword_4F04D80 = 1;
  if ( a4 )
    *a4 = 0;
  if ( a5 )
    *a5 = 0;
  if ( a3 )
    *a3 = 0;
  if ( a6 )
    *(_QWORD *)a6 = *(_QWORD *)&dword_4F077C8;
  if ( a7 )
    *a7 = *(_QWORD *)&dword_4F077C8;
  v10 = word_4F06418[0];
  if ( word_4F06418[0] == 73 || word_4F06418[0] == 163 )
  {
    a5 = 0;
    memset(v75, 0, sizeof(v75));
    v76 = 0;
    v77 = 0;
    if ( !a4 )
      goto LABEL_20;
  }
  else
  {
    if ( !(_DWORD)a2 || word_4F06418[0] != 55 )
    {
      if ( word_4F06418[0] == 56 && (unsigned int)sub_651030(&v74) )
      {
        if ( a6 )
          *(_QWORD *)a6 = *(_QWORD *)&dword_4F063F8;
        sub_7AE360(a1);
        sub_7B8B50(a1, a2, v28, v29, v30, v31);
        sub_7AE360(a1);
        sub_7B8B50(a1, a2, v32, v33, v34, v35);
        if ( word_4F06418[0] == 75 )
        {
          if ( a7 )
            *a7 = qword_4F063F0;
          v11 = 1;
          if ( v74 )
            sub_7AE360(a1);
          goto LABEL_17;
        }
        if ( a3 )
          *a3 = 1;
      }
      v11 = 0;
      goto LABEL_17;
    }
    v10 = 55;
    memset(v75, 0, sizeof(v75));
    v76 = 0;
    v77 = 0;
    if ( !a4 )
      goto LABEL_27;
  }
  *a4 = dword_4F06650[0];
  v10 = word_4F06418[0];
LABEL_20:
  v13 = v10;
  if ( v10 == 163 )
  {
    sub_7AE360(a1);
    sub_7B8B50(a1, a2, v65, v66, v67, v68);
    v13 = word_4F06418[0];
  }
  if ( v13 != 55 )
    goto LABEL_23;
LABEL_27:
  v14 = (unsigned int)BYTE1(v75[18]) + 1;
  if ( !dword_4D04428 )
  {
    a2 = v75;
    ++HIBYTE(v75[18]);
    ++BYTE1(v75[18]);
    sub_7C6880(a1, (__int64)v75, v14, (__int64)&dword_4D04428, (__int64)a5, a6);
    --BYTE1(v75[18]);
    --HIBYTE(v75[18]);
    if ( word_4F06418[0] != 73 )
      goto LABEL_24;
    goto LABEL_47;
  }
  v15 = &dword_4F077C8;
  ++HIBYTE(v75[18]);
  v70 = v9;
  v16 = 0;
  ++BYTE2(v75[18]);
  ++BYTE1(v75[18]);
  v71 = dword_4F077C8;
  v69 = unk_4F077CC;
  while ( 1 )
  {
    a2 = v75;
    v22 = a1;
    sub_7C6880(a1, (__int64)v75, v14, (__int64)v15, (__int64)a5, a6);
    v13 = word_4F06418[0];
    if ( word_4F06418[0] == 73 )
      break;
    if ( word_4F06418[0] != 74 )
    {
      v9 = v70;
      goto LABEL_69;
    }
    v17 = 0;
LABEL_31:
    if ( (_WORD)v16 != 67 )
    {
      v23 = v16;
      v9 = v70;
      goto LABEL_80;
    }
    sub_7B8B50(v22, a2, v23, v24, v25, v26);
    sub_7AE360(a1);
    sub_7B8B50(a1, a2, v18, v19, v20, v21);
  }
  v71 = dword_4F063F8;
  v69 = word_4F063FC[0];
  sub_7C6040(a1, 0, v23, v24, v25, v26);
  sub_7AE360(a1);
  a2 = 0;
  v22 = 0;
  v16 = sub_7BE840(0, 0);
  if ( word_4F06418[0] == 74 )
  {
    if ( (_WORD)v16 == 76 )
    {
      if ( !dword_4D04408 )
      {
        v9 = v70;
LABEL_38:
        v27 = HIBYTE(v75[18]) - 1;
        --BYTE2(v75[18]);
        --BYTE1(v75[18]);
        if ( a6 )
        {
          *(_DWORD *)a6 = v71;
          HIBYTE(v75[18]) = v27;
          *(_WORD *)(a6 + 4) = v69;
        }
        else
        {
          --HIBYTE(v75[18]);
        }
        goto LABEL_52;
      }
      sub_7B8B50(0, 0, v23, dword_4D04408, v25, v26);
      sub_7AE360(a1);
      a2 = 0;
      v22 = 0;
      v16 = sub_7BE840(0, 0);
      v17 = 0;
    }
    else
    {
      v17 = 1;
    }
    goto LABEL_31;
  }
  v23 = v16;
  v17 = 1;
  v9 = v70;
LABEL_80:
  if ( (_WORD)v23 != 73 && v17 )
    goto LABEL_38;
  v13 = word_4F06418[0];
LABEL_69:
  if ( v13 != 75 && v13 != 9 )
  {
    sub_7B8B50(v22, a2, v23, v24, v25, v26);
    v13 = word_4F06418[0];
  }
  --BYTE2(v75[18]);
  --BYTE1(v75[18]);
  --HIBYTE(v75[18]);
LABEL_23:
  if ( v13 != 73 )
  {
LABEL_24:
    v11 = 0;
    goto LABEL_25;
  }
LABEL_47:
  if ( a6 )
    *(_QWORD *)a6 = *(_QWORD *)&dword_4F063F8;
  sub_7B8190();
  dword_4F04D80 = 1;
  sub_7AE360(a1);
  sub_7B8B50(a1, a2, v36, v37, v38, v39);
  ++BYTE2(v75[18]);
  sub_7C6880(a1, (__int64)v75, v40, v41, v42, v43);
  if ( word_4F06418[0] == 74 )
    sub_7AE360(a1);
  sub_7B8260();
LABEL_52:
  if ( a7 )
    *a7 = qword_4F063F0;
  if ( v10 == 163 )
  {
    while ( (unsigned __int16)sub_7BE840(0, 0) == 150 )
    {
      sub_7B8B50(0, 0, v53, v54, v55, v56);
      sub_7AE360(a1);
      sub_7B8B50(a1, 0, v57, v58, v59, v60);
      ++BYTE1(v75[18]);
      ++HIBYTE(v75[18]);
      sub_7C6880(a1, (__int64)v75, v61, v62, v63, v64);
      --BYTE1(v75[18]);
      --HIBYTE(v75[18]);
      v52 = word_4F06418[0];
      if ( word_4F06418[0] == 73 )
      {
        sub_7B8190();
        dword_4F04D80 = 1;
        sub_7AE360(a1);
        sub_7B8B50(a1, v75, v44, v45, v46, v47);
        ++BYTE2(v75[18]);
        sub_7C6880(a1, (__int64)v75, v48, v49, v50, v51);
        if ( word_4F06418[0] == 74 )
          sub_7AE360(a1);
        sub_7B8260();
        v52 = word_4F06418[0];
      }
      if ( v52 != 74 )
        goto LABEL_56;
    }
  }
  v11 = 1;
  if ( word_4F06418[0] != 74 )
  {
LABEL_56:
    if ( a3 )
      *a3 = 1;
    v11 = 0;
  }
  if ( v9 )
    *v9 = dword_4F06650[0];
LABEL_25:
  sub_7AE210(a1);
LABEL_17:
  dword_4F04D80 = v72;
  return v11;
}
