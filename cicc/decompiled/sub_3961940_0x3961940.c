// Function: sub_3961940
// Address: 0x3961940
//
unsigned __int64 __fastcall sub_3961940(__int64 a1)
{
  unsigned __int64 v2; // rsi
  _QWORD *v3; // rax
  _DWORD *v4; // rdi
  __int64 v5; // rcx
  __int64 v6; // rdx
  unsigned __int64 v7; // rsi
  _QWORD *v8; // rax
  _DWORD *v9; // rdi
  __int64 v10; // rcx
  __int64 v11; // rdx
  unsigned __int64 v12; // rsi
  _QWORD *v13; // rax
  _DWORD *v14; // rdi
  __int64 v15; // rcx
  __int64 v16; // rdx
  unsigned __int64 v17; // rsi
  unsigned __int64 result; // rax
  _DWORD *v19; // rdi
  __int64 v20; // rcx
  __int64 v21; // rdx
  _DWORD *v22; // r8
  _DWORD *v23; // rdi
  __int64 v24; // rcx
  __int64 v25; // rdx
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
  __int64 v36; // rax
  _DWORD *v37; // r8
  _DWORD *v38; // rdi
  __int64 v39; // rcx
  __int64 v40; // rdx
  _DWORD *v41; // rax
  int v42; // eax
  _DWORD *v43; // rax
  int v44; // eax
  int *v45; // rax
  int v46; // eax

  if ( BYTE4(qword_4FBB390[2]) )
  {
    v41 = sub_16D40F0((__int64)qword_4FBB390);
    v42 = v41 ? *v41 : LODWORD(qword_4FBB390[2]);
    if ( v42 >= 0 )
    {
      v43 = sub_16D40F0((__int64)qword_4FBB390);
      v44 = v43 ? *v43 : LODWORD(qword_4FBB390[2]);
      if ( v44 <= 10 )
      {
        v45 = (int *)sub_16D40F0((__int64)qword_4FBB390);
        if ( v45 )
          v46 = *v45;
        else
          v46 = qword_4FBB390[2];
        *(_DWORD *)(a1 + 64) = v46;
        if ( v46 <= 1 )
        {
          *(_BYTE *)(a1 + 68) = 1;
        }
        else if ( v46 != 2 )
        {
          goto LABEL_2;
        }
        *(_DWORD *)(a1 + 72) *= 2;
      }
    }
  }
LABEL_2:
  v2 = sub_16D5D50();
  v3 = *(_QWORD **)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    v4 = dword_4FA0208;
    do
    {
      while ( 1 )
      {
        v5 = v3[2];
        v6 = v3[3];
        if ( v2 <= v3[4] )
          break;
        v3 = (_QWORD *)v3[3];
        if ( !v6 )
          goto LABEL_7;
      }
      v4 = v3;
      v3 = (_QWORD *)v3[2];
    }
    while ( v5 );
LABEL_7:
    if ( v4 != dword_4FA0208 && v2 >= *((_QWORD *)v4 + 4) )
    {
      v26 = *((_QWORD *)v4 + 7);
      v27 = v4 + 12;
      if ( v26 )
      {
        v28 = v4 + 12;
        do
        {
          while ( 1 )
          {
            v29 = *(_QWORD *)(v26 + 16);
            v30 = *(_QWORD *)(v26 + 24);
            if ( *(_DWORD *)(v26 + 32) >= dword_5055EE8 )
              break;
            v26 = *(_QWORD *)(v26 + 24);
            if ( !v30 )
              goto LABEL_45;
          }
          v28 = (_DWORD *)v26;
          v26 = *(_QWORD *)(v26 + 16);
        }
        while ( v29 );
LABEL_45:
        if ( v27 != v28 && dword_5055EE8 >= v28[8] && (int)v28[9] > 0 )
          *(_BYTE *)(a1 + 68) = byte_5055F80;
      }
    }
  }
  v7 = sub_16D5D50();
  v8 = *(_QWORD **)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    v9 = dword_4FA0208;
    do
    {
      while ( 1 )
      {
        v10 = v8[2];
        v11 = v8[3];
        if ( v7 <= v8[4] )
          break;
        v8 = (_QWORD *)v8[3];
        if ( !v11 )
          goto LABEL_14;
      }
      v9 = v8;
      v8 = (_QWORD *)v8[2];
    }
    while ( v10 );
LABEL_14:
    if ( v9 != dword_4FA0208 && v7 >= *((_QWORD *)v9 + 4) )
    {
      v36 = *((_QWORD *)v9 + 7);
      v37 = v9 + 12;
      if ( v36 )
      {
        v38 = v9 + 12;
        do
        {
          while ( 1 )
          {
            v39 = *(_QWORD *)(v36 + 16);
            v40 = *(_QWORD *)(v36 + 24);
            if ( *(_DWORD *)(v36 + 32) >= dword_5055C48 )
              break;
            v36 = *(_QWORD *)(v36 + 24);
            if ( !v40 )
              goto LABEL_63;
          }
          v38 = (_DWORD *)v36;
          v36 = *(_QWORD *)(v36 + 16);
        }
        while ( v39 );
LABEL_63:
        if ( v37 != v38 && dword_5055C48 >= v38[8] && (int)v38[9] > 0 )
          *(_BYTE *)(a1 + 70) = byte_5055CE0;
      }
    }
  }
  v12 = sub_16D5D50();
  v13 = *(_QWORD **)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    v14 = dword_4FA0208;
    do
    {
      while ( 1 )
      {
        v15 = v13[2];
        v16 = v13[3];
        if ( v12 <= v13[4] )
          break;
        v13 = (_QWORD *)v13[3];
        if ( !v16 )
          goto LABEL_21;
      }
      v14 = v13;
      v13 = (_QWORD *)v13[2];
    }
    while ( v15 );
LABEL_21:
    if ( v14 != dword_4FA0208 && v12 >= *((_QWORD *)v14 + 4) )
    {
      v31 = *((_QWORD *)v14 + 7);
      v32 = v14 + 12;
      if ( v31 )
      {
        v33 = v14 + 12;
        do
        {
          while ( 1 )
          {
            v34 = *(_QWORD *)(v31 + 16);
            v35 = *(_QWORD *)(v31 + 24);
            if ( *(_DWORD *)(v31 + 32) >= dword_5055D28 )
              break;
            v31 = *(_QWORD *)(v31 + 24);
            if ( !v35 )
              goto LABEL_54;
          }
          v33 = (_DWORD *)v31;
          v31 = *(_QWORD *)(v31 + 16);
        }
        while ( v34 );
LABEL_54:
        if ( v32 != v33 && dword_5055D28 >= v33[8] && (int)v33[9] > 0 )
          *(_BYTE *)(a1 + 69) = byte_5055DC0;
      }
    }
  }
  v17 = sub_16D5D50();
  result = *(_QWORD *)&dword_4FA0208[2];
  if ( *(_QWORD *)&dword_4FA0208[2] )
  {
    v19 = dword_4FA0208;
    do
    {
      while ( 1 )
      {
        v20 = *(_QWORD *)(result + 16);
        v21 = *(_QWORD *)(result + 24);
        if ( v17 <= *(_QWORD *)(result + 32) )
          break;
        result = *(_QWORD *)(result + 24);
        if ( !v21 )
          goto LABEL_28;
      }
      v19 = (_DWORD *)result;
      result = *(_QWORD *)(result + 16);
    }
    while ( v20 );
LABEL_28:
    result = (unsigned __int64)dword_4FA0208;
    if ( v19 != dword_4FA0208 && v17 >= *((_QWORD *)v19 + 4) )
    {
      result = *((_QWORD *)v19 + 7);
      v22 = v19 + 12;
      if ( result )
      {
        v23 = v19 + 12;
        do
        {
          while ( 1 )
          {
            v24 = *(_QWORD *)(result + 16);
            v25 = *(_QWORD *)(result + 24);
            if ( *(_DWORD *)(result + 32) >= dword_5055E08 )
              break;
            result = *(_QWORD *)(result + 24);
            if ( !v25 )
              goto LABEL_35;
          }
          v23 = (_DWORD *)result;
          result = *(_QWORD *)(result + 16);
        }
        while ( v24 );
LABEL_35:
        if ( v22 != v23 && dword_5055E08 >= v23[8] && (int)v23[9] > 0 )
        {
          result = (unsigned int)dword_5055EA0;
          *(_DWORD *)(a1 + 72) = dword_5055EA0;
        }
      }
    }
  }
  return result;
}
