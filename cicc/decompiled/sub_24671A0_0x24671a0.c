// Function: sub_24671A0
// Address: 0x24671a0
//
char *__fastcall sub_24671A0(__int64 a1, int a2, char a3, char a4, char a5)
{
  unsigned __int64 v10; // rsi
  char *v11; // rax
  char *v12; // r8
  char *v13; // rdi
  __int64 v14; // rcx
  __int64 v15; // rdx
  __int64 v16; // rax
  _DWORD *v17; // rdi
  __int64 v18; // rcx
  __int64 v19; // rdx
  unsigned __int64 v20; // rsi
  _QWORD *v21; // rax
  _QWORD *v22; // r15
  _QWORD *v23; // rdi
  __int64 v24; // rcx
  __int64 v25; // rdx
  __int64 v26; // rax
  _QWORD *v27; // rdi
  __int64 v28; // rcx
  __int64 v29; // rdx
  char v30; // r14
  _QWORD *v31; // r15
  _QWORD *v32; // r13
  unsigned __int64 v33; // rsi
  _QWORD *v34; // rax
  _QWORD *v35; // rdi
  __int64 v36; // rcx
  __int64 v37; // rdx
  __int64 v38; // rax
  _QWORD *v39; // rdi
  __int64 v40; // rcx
  __int64 v41; // rdx
  _QWORD *v42; // r14
  char *v43; // r13
  unsigned __int64 v44; // rsi
  char *v45; // rax
  char *v46; // rdi
  __int64 v47; // rcx
  __int64 v48; // rdx
  char *result; // rax
  char *v50; // rdi
  __int64 v51; // rcx
  __int64 v52; // rdx
  _QWORD *v53; // [rsp+8h] [rbp-38h]
  _QWORD *v54; // [rsp+8h] [rbp-38h]
  _QWORD *v55; // [rsp+8h] [rbp-38h]

  v53 = sub_C52410();
  v10 = sub_C959E0();
  v11 = (char *)v53[2];
  v12 = (char *)(v53 + 1);
  if ( v11 )
  {
    v13 = (char *)(v53 + 1);
    do
    {
      while ( 1 )
      {
        v14 = *((_QWORD *)v11 + 2);
        v15 = *((_QWORD *)v11 + 3);
        if ( v10 <= *((_QWORD *)v11 + 4) )
          break;
        v11 = (char *)*((_QWORD *)v11 + 3);
        if ( !v15 )
          goto LABEL_6;
      }
      v13 = v11;
      v11 = (char *)*((_QWORD *)v11 + 2);
    }
    while ( v14 );
LABEL_6:
    if ( v12 != v13 && v10 >= *((_QWORD *)v13 + 4) )
      v12 = v13;
  }
  v54 = v12;
  if ( v12 != (char *)sub_C52410() + 8 )
  {
    v16 = v54[7];
    if ( v16 )
    {
      v17 = v54 + 6;
      do
      {
        while ( 1 )
        {
          v18 = *(_QWORD *)(v16 + 16);
          v19 = *(_QWORD *)(v16 + 24);
          if ( *(_DWORD *)(v16 + 32) >= dword_4FE7FE8 )
            break;
          v16 = *(_QWORD *)(v16 + 24);
          if ( !v19 )
            goto LABEL_15;
        }
        v17 = (_DWORD *)v16;
        v16 = *(_QWORD *)(v16 + 16);
      }
      while ( v18 );
LABEL_15:
      if ( v54 + 6 != (_QWORD *)v17 && dword_4FE7FE8 >= v17[8] && (int)v17[9] > 0 )
        a4 = qword_4FE8068;
    }
  }
  *(_BYTE *)a1 = a4;
  if ( a4 )
    a2 = 2;
  v55 = sub_C52410();
  v20 = sub_C959E0();
  v21 = (_QWORD *)v55[2];
  v22 = v55 + 1;
  if ( v21 )
  {
    v23 = v55 + 1;
    do
    {
      while ( 1 )
      {
        v24 = v21[2];
        v25 = v21[3];
        if ( v20 <= v21[4] )
          break;
        v21 = (_QWORD *)v21[3];
        if ( !v25 )
          goto LABEL_24;
      }
      v23 = v21;
      v21 = (_QWORD *)v21[2];
    }
    while ( v24 );
LABEL_24:
    if ( v22 != v23 && v20 >= v23[4] )
      v22 = v23;
  }
  if ( v22 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v26 = v22[7];
    if ( v26 )
    {
      v27 = v22 + 6;
      do
      {
        while ( 1 )
        {
          v28 = *(_QWORD *)(v26 + 16);
          v29 = *(_QWORD *)(v26 + 24);
          if ( *(_DWORD *)(v26 + 32) >= dword_4FE8DE8 )
            break;
          v26 = *(_QWORD *)(v26 + 24);
          if ( !v29 )
            goto LABEL_33;
        }
        v27 = (_QWORD *)v26;
        v26 = *(_QWORD *)(v26 + 16);
      }
      while ( v28 );
LABEL_33:
      if ( v22 + 6 != v27 && dword_4FE8DE8 >= *((_DWORD *)v27 + 8) && *((int *)v27 + 9) > 0 )
        a2 = qword_4FE8E68;
    }
  }
  *(_DWORD *)(a1 + 4) = a2;
  v30 = *(_BYTE *)a1 | a3;
  v31 = sub_C52410();
  v32 = v31 + 1;
  v33 = sub_C959E0();
  v34 = (_QWORD *)v31[2];
  if ( v34 )
  {
    v35 = v31 + 1;
    do
    {
      while ( 1 )
      {
        v36 = v34[2];
        v37 = v34[3];
        if ( v33 <= v34[4] )
          break;
        v34 = (_QWORD *)v34[3];
        if ( !v37 )
          goto LABEL_40;
      }
      v35 = v34;
      v34 = (_QWORD *)v34[2];
    }
    while ( v36 );
LABEL_40:
    if ( v32 != v35 && v33 >= v35[4] )
      v32 = v35;
  }
  if ( v32 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v38 = v32[7];
    if ( v38 )
    {
      v39 = v32 + 6;
      do
      {
        while ( 1 )
        {
          v40 = *(_QWORD *)(v38 + 16);
          v41 = *(_QWORD *)(v38 + 24);
          if ( *(_DWORD *)(v38 + 32) >= dword_4FE8D08 )
            break;
          v38 = *(_QWORD *)(v38 + 24);
          if ( !v41 )
            goto LABEL_49;
        }
        v39 = (_QWORD *)v38;
        v38 = *(_QWORD *)(v38 + 16);
      }
      while ( v40 );
LABEL_49:
      if ( v32 + 6 != v39 && dword_4FE8D08 >= *((_DWORD *)v39 + 8) && *((int *)v39 + 9) > 0 )
        v30 = qword_4FE8D88;
    }
  }
  *(_BYTE *)(a1 + 8) = v30;
  v42 = sub_C52410();
  v43 = (char *)(v42 + 1);
  v44 = sub_C959E0();
  v45 = (char *)v42[2];
  if ( v45 )
  {
    v46 = (char *)(v42 + 1);
    do
    {
      while ( 1 )
      {
        v47 = *((_QWORD *)v45 + 2);
        v48 = *((_QWORD *)v45 + 3);
        if ( v44 <= *((_QWORD *)v45 + 4) )
          break;
        v45 = (char *)*((_QWORD *)v45 + 3);
        if ( !v48 )
          goto LABEL_56;
      }
      v46 = v45;
      v45 = (char *)*((_QWORD *)v45 + 2);
    }
    while ( v47 );
LABEL_56:
    if ( v43 != v46 && v44 >= *((_QWORD *)v46 + 4) )
      v43 = v46;
  }
  result = (char *)sub_C52410() + 8;
  if ( v43 != result )
  {
    result = (char *)*((_QWORD *)v43 + 7);
    if ( result )
    {
      v50 = v43 + 48;
      do
      {
        while ( 1 )
        {
          v51 = *((_QWORD *)result + 2);
          v52 = *((_QWORD *)result + 3);
          if ( *((_DWORD *)result + 8) >= dword_4FE8368 )
            break;
          result = (char *)*((_QWORD *)result + 3);
          if ( !v52 )
            goto LABEL_65;
        }
        v50 = result;
        result = (char *)*((_QWORD *)result + 2);
      }
      while ( v51 );
LABEL_65:
      if ( v43 + 48 != v50 && dword_4FE8368 >= *((_DWORD *)v50 + 8) && *((int *)v50 + 9) > 0 )
        a5 = byte_4FE83E8;
    }
  }
  *(_BYTE *)(a1 + 9) = a5;
  return result;
}
