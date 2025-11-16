// Function: sub_3850D70
// Address: 0x3850d70
//
__int64 __fastcall sub_3850D70(__int64 a1, int a2)
{
  unsigned __int64 v4; // rsi
  _QWORD *v5; // rax
  _DWORD *v6; // rdi
  __int64 v7; // rcx
  __int64 v8; // rdx
  bool v9; // zf
  unsigned __int64 v10; // rsi
  _QWORD *v11; // rax
  _DWORD *v12; // rdi
  __int64 v13; // rcx
  __int64 v14; // rdx
  unsigned __int64 v15; // rsi
  _QWORD *v16; // rax
  _DWORD *v17; // rdi
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rax
  _DWORD *v21; // r8
  _DWORD *v22; // rdi
  __int64 v23; // rcx
  __int64 v24; // rdx
  __int64 v26; // rax
  _DWORD *v27; // r8
  _DWORD *v28; // rdi
  __int64 v29; // rcx
  __int64 v30; // rdx
  __int64 v31; // rax
  _DWORD *v32; // r8
  _DWORD *v33; // rdi
  __int64 v34; // rcx
  __int64 v35; // rdx
  unsigned __int64 v36; // rsi
  _QWORD *v37; // rax
  _DWORD *v38; // rdi
  __int64 v39; // rcx
  __int64 v40; // rdx
  __int64 v41; // rax
  _DWORD *v42; // r8
  _DWORD *v43; // rdi
  __int64 v44; // rcx
  __int64 v45; // rdx

  *(_BYTE *)(a1 + 8) = 0;
  *(_BYTE *)(a1 + 16) = 0;
  *(_BYTE *)(a1 + 24) = 0;
  *(_BYTE *)(a1 + 32) = 0;
  *(_BYTE *)(a1 + 40) = 0;
  *(_BYTE *)(a1 + 48) = 0;
  *(_BYTE *)(a1 + 56) = 0;
  *(_BYTE *)(a1 + 61) = 0;
  v4 = sub_16D5D50();
  v5 = *(_QWORD **)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    v6 = dword_4FA0208;
    do
    {
      while ( 1 )
      {
        v7 = v5[2];
        v8 = v5[3];
        if ( v4 <= v5[4] )
          break;
        v5 = (_QWORD *)v5[3];
        if ( !v8 )
          goto LABEL_6;
      }
      v6 = v5;
      v5 = (_QWORD *)v5[2];
    }
    while ( v7 );
LABEL_6:
    if ( v6 != dword_4FA0208 && v4 >= *((_QWORD *)v6 + 4) )
    {
      v26 = *((_QWORD *)v6 + 7);
      v27 = v6 + 12;
      if ( v26 )
      {
        v28 = v6 + 12;
        do
        {
          while ( 1 )
          {
            v29 = *(_QWORD *)(v26 + 16);
            v30 = *(_QWORD *)(v26 + 24);
            if ( *(_DWORD *)(v26 + 32) >= dword_5051EC8 )
              break;
            v26 = *(_QWORD *)(v26 + 24);
            if ( !v30 )
              goto LABEL_46;
          }
          v28 = (_DWORD *)v26;
          v26 = *(_QWORD *)(v26 + 16);
        }
        while ( v29 );
LABEL_46:
        if ( v27 != v28 && dword_5051EC8 >= v28[8] && (int)v28[9] > 0 )
          a2 = dword_5051F60;
      }
    }
  }
  *(_DWORD *)a1 = a2;
  v9 = *(_BYTE *)(a1 + 8) == 0;
  *(_DWORD *)(a1 + 4) = dword_5051E80;
  if ( v9 )
    *(_BYTE *)(a1 + 8) = 1;
  v9 = *(_BYTE *)(a1 + 40) == 0;
  *(_DWORD *)(a1 + 36) = dword_5051BE0;
  if ( v9 )
    *(_BYTE *)(a1 + 40) = 1;
  v10 = sub_16D5D50();
  v11 = *(_QWORD **)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    v12 = dword_4FA0208;
    do
    {
      while ( 1 )
      {
        v13 = v11[2];
        v14 = v11[3];
        if ( v10 <= v11[4] )
          break;
        v11 = (_QWORD *)v11[3];
        if ( !v14 )
          goto LABEL_17;
      }
      v12 = v11;
      v11 = (_QWORD *)v11[2];
    }
    while ( v13 );
LABEL_17:
    if ( v12 != dword_4FA0208 && v10 >= *((_QWORD *)v12 + 4) )
    {
      v31 = *((_QWORD *)v12 + 7);
      v32 = v12 + 12;
      if ( v31 )
      {
        v33 = v12 + 12;
        do
        {
          while ( 1 )
          {
            v34 = *(_QWORD *)(v31 + 16);
            v35 = *(_QWORD *)(v31 + 24);
            if ( *(_DWORD *)(v31 + 32) >= dword_5051A68 )
              break;
            v31 = *(_QWORD *)(v31 + 24);
            if ( !v35 )
              goto LABEL_55;
          }
          v33 = (_DWORD *)v31;
          v31 = *(_QWORD *)(v31 + 16);
        }
        while ( v34 );
LABEL_55:
        if ( v32 != v33 && dword_5051A68 >= v33[8] && (int)v33[9] > 0 )
        {
          v9 = *(_BYTE *)(a1 + 48) == 0;
          *(_DWORD *)(a1 + 44) = dword_5051B00;
          if ( v9 )
            *(_BYTE *)(a1 + 48) = 1;
        }
      }
    }
  }
  v9 = *(_BYTE *)(a1 + 56) == 0;
  *(_DWORD *)(a1 + 52) = dword_5051DA0;
  if ( v9 )
    *(_BYTE *)(a1 + 56) = 1;
  v15 = sub_16D5D50();
  v16 = *(_QWORD **)&dword_4FA0208[2];
  if ( !*(_QWORD *)&dword_4FA0208[2] )
    goto LABEL_35;
  v17 = dword_4FA0208;
  do
  {
    while ( 1 )
    {
      v18 = v16[2];
      v19 = v16[3];
      if ( v15 <= v16[4] )
        break;
      v16 = (_QWORD *)v16[3];
      if ( !v19 )
        goto LABEL_26;
    }
    v17 = v16;
    v16 = (_QWORD *)v16[2];
  }
  while ( v18 );
LABEL_26:
  if ( v17 == dword_4FA0208 )
    goto LABEL_35;
  if ( v15 < *((_QWORD *)v17 + 4) )
    goto LABEL_35;
  v20 = *((_QWORD *)v17 + 7);
  v21 = v17 + 12;
  if ( !v20 )
    goto LABEL_35;
  v22 = v17 + 12;
  do
  {
    while ( 1 )
    {
      v23 = *(_QWORD *)(v20 + 16);
      v24 = *(_QWORD *)(v20 + 24);
      if ( *(_DWORD *)(v20 + 32) >= dword_5051EC8 )
        break;
      v20 = *(_QWORD *)(v20 + 24);
      if ( !v24 )
        goto LABEL_33;
    }
    v22 = (_DWORD *)v20;
    v20 = *(_QWORD *)(v20 + 16);
  }
  while ( v23 );
LABEL_33:
  if ( v21 != v22 && dword_5051EC8 >= v22[8] && v22[9] )
  {
    v36 = sub_16D5D50();
    v37 = *(_QWORD **)&dword_4FA0208[2];
    if ( !*(_QWORD *)&dword_4FA0208[2] )
      return a1;
    v38 = dword_4FA0208;
    do
    {
      while ( 1 )
      {
        v39 = v37[2];
        v40 = v37[3];
        if ( v36 <= v37[4] )
          break;
        v37 = (_QWORD *)v37[3];
        if ( !v40 )
          goto LABEL_66;
      }
      v38 = v37;
      v37 = (_QWORD *)v37[2];
    }
    while ( v39 );
LABEL_66:
    if ( v38 == dword_4FA0208 )
      return a1;
    if ( v36 < *((_QWORD *)v38 + 4) )
      return a1;
    v41 = *((_QWORD *)v38 + 7);
    v42 = v38 + 12;
    if ( !v41 )
      return a1;
    v43 = v38 + 12;
    do
    {
      while ( 1 )
      {
        v44 = *(_QWORD *)(v41 + 16);
        v45 = *(_QWORD *)(v41 + 24);
        if ( *(_DWORD *)(v41 + 32) >= dword_5051C28 )
          break;
        v41 = *(_QWORD *)(v41 + 24);
        if ( !v45 )
          goto LABEL_73;
      }
      v43 = (_DWORD *)v41;
      v41 = *(_QWORD *)(v41 + 16);
    }
    while ( v44 );
LABEL_73:
    if ( v42 == v43 || dword_5051C28 < v43[8] || (int)v43[9] <= 0 )
      return a1;
  }
  else
  {
LABEL_35:
    *(_DWORD *)(a1 + 28) = 5;
    if ( !*(_BYTE *)(a1 + 32) )
      *(_BYTE *)(a1 + 32) = 1;
    *(_DWORD *)(a1 + 20) = 50;
    if ( !*(_BYTE *)(a1 + 24) )
      *(_BYTE *)(a1 + 24) = 1;
  }
  v9 = *(_BYTE *)(a1 + 16) == 0;
  *(_DWORD *)(a1 + 12) = dword_5051CC0;
  if ( !v9 )
    return a1;
  *(_BYTE *)(a1 + 16) = 1;
  return a1;
}
