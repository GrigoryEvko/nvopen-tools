// Function: sub_2D04FA0
// Address: 0x2d04fa0
//
unsigned __int64 __fastcall sub_2D04FA0(__int64 a1)
{
  _QWORD *v2; // r13
  _QWORD *v3; // r12
  unsigned __int64 v4; // rsi
  _QWORD *v5; // rax
  _QWORD *v6; // rdi
  __int64 v7; // rcx
  __int64 v8; // rdx
  __int64 v9; // rax
  _QWORD *v10; // rdi
  __int64 v11; // rcx
  __int64 v12; // rdx
  _QWORD *v13; // r13
  _QWORD *v14; // r12
  unsigned __int64 v15; // rsi
  _QWORD *v16; // rax
  _QWORD *v17; // rdi
  __int64 v18; // rcx
  __int64 v19; // rdx
  __int64 v20; // rax
  _QWORD *v21; // rdi
  __int64 v22; // rcx
  __int64 v23; // rdx
  _QWORD *v24; // r13
  _QWORD *v25; // r12
  unsigned __int64 v26; // rsi
  _QWORD *v27; // rax
  _QWORD *v28; // rdi
  __int64 v29; // rcx
  __int64 v30; // rdx
  __int64 v31; // rax
  _QWORD *v32; // rdi
  __int64 v33; // rcx
  __int64 v34; // rdx
  _QWORD *v35; // r13
  char *v36; // r12
  unsigned __int64 v37; // rsi
  char *v38; // rax
  char *v39; // rdi
  __int64 v40; // rcx
  __int64 v41; // rdx
  unsigned __int64 result; // rax
  char *v43; // rdi
  __int64 v44; // rcx
  __int64 v45; // rdx
  _DWORD *v46; // rax
  int v47; // eax
  _DWORD *v48; // rax
  int v49; // eax
  int *v50; // rax
  int v51; // eax

  if ( BYTE4(qword_4F862F0[2]) )
  {
    v46 = sub_C94E20((__int64)qword_4F862F0);
    v47 = v46 ? *v46 : LODWORD(qword_4F862F0[2]);
    if ( v47 >= 0 )
    {
      v48 = sub_C94E20((__int64)qword_4F862F0);
      v49 = v48 ? *v48 : LODWORD(qword_4F862F0[2]);
      if ( v49 <= 10 )
      {
        v50 = (int *)sub_C94E20((__int64)qword_4F862F0);
        if ( v50 )
          v51 = *v50;
        else
          v51 = qword_4F862F0[2];
        *(_DWORD *)(a1 + 56) = v51;
        if ( v51 <= 1 )
        {
          *(_BYTE *)(a1 + 60) = 1;
        }
        else if ( v51 != 2 )
        {
          goto LABEL_2;
        }
        *(_DWORD *)(a1 + 64) *= 2;
      }
    }
  }
LABEL_2:
  v2 = sub_C52410();
  v3 = v2 + 1;
  v4 = sub_C959E0();
  v5 = (_QWORD *)v2[2];
  if ( v5 )
  {
    v6 = v2 + 1;
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
          goto LABEL_7;
      }
      v6 = v5;
      v5 = (_QWORD *)v5[2];
    }
    while ( v7 );
LABEL_7:
    if ( v3 != v6 && v4 >= v6[4] )
      v3 = v6;
  }
  if ( v3 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v9 = v3[7];
    if ( v9 )
    {
      v10 = v3 + 6;
      do
      {
        while ( 1 )
        {
          v11 = *(_QWORD *)(v9 + 16);
          v12 = *(_QWORD *)(v9 + 24);
          if ( *(_DWORD *)(v9 + 32) >= dword_5015C88 )
            break;
          v9 = *(_QWORD *)(v9 + 24);
          if ( !v12 )
            goto LABEL_16;
        }
        v10 = (_QWORD *)v9;
        v9 = *(_QWORD *)(v9 + 16);
      }
      while ( v11 );
LABEL_16:
      if ( v3 + 6 != v10 && dword_5015C88 >= *((_DWORD *)v10 + 8) && *((int *)v10 + 9) > 0 )
        *(_BYTE *)(a1 + 60) = qword_5015D08;
    }
  }
  v13 = sub_C52410();
  v14 = v13 + 1;
  v15 = sub_C959E0();
  v16 = (_QWORD *)v13[2];
  if ( v16 )
  {
    v17 = v13 + 1;
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
          goto LABEL_23;
      }
      v17 = v16;
      v16 = (_QWORD *)v16[2];
    }
    while ( v18 );
LABEL_23:
    if ( v14 != v17 && v15 >= v17[4] )
      v14 = v17;
  }
  if ( v14 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v20 = v14[7];
    if ( v20 )
    {
      v21 = v14 + 6;
      do
      {
        while ( 1 )
        {
          v22 = *(_QWORD *)(v20 + 16);
          v23 = *(_QWORD *)(v20 + 24);
          if ( *(_DWORD *)(v20 + 32) >= dword_50159E8 )
            break;
          v20 = *(_QWORD *)(v20 + 24);
          if ( !v23 )
            goto LABEL_32;
        }
        v21 = (_QWORD *)v20;
        v20 = *(_QWORD *)(v20 + 16);
      }
      while ( v22 );
LABEL_32:
      if ( v14 + 6 != v21 && dword_50159E8 >= *((_DWORD *)v21 + 8) && *((int *)v21 + 9) > 0 )
        *(_BYTE *)(a1 + 62) = byte_5015A68;
    }
  }
  v24 = sub_C52410();
  v25 = v24 + 1;
  v26 = sub_C959E0();
  v27 = (_QWORD *)v24[2];
  if ( v27 )
  {
    v28 = v24 + 1;
    do
    {
      while ( 1 )
      {
        v29 = v27[2];
        v30 = v27[3];
        if ( v26 <= v27[4] )
          break;
        v27 = (_QWORD *)v27[3];
        if ( !v30 )
          goto LABEL_39;
      }
      v28 = v27;
      v27 = (_QWORD *)v27[2];
    }
    while ( v29 );
LABEL_39:
    if ( v25 != v28 && v26 >= v28[4] )
      v25 = v28;
  }
  if ( v25 != (_QWORD *)((char *)sub_C52410() + 8) )
  {
    v31 = v25[7];
    if ( v31 )
    {
      v32 = v25 + 6;
      do
      {
        while ( 1 )
        {
          v33 = *(_QWORD *)(v31 + 16);
          v34 = *(_QWORD *)(v31 + 24);
          if ( *(_DWORD *)(v31 + 32) >= dword_5015AC8 )
            break;
          v31 = *(_QWORD *)(v31 + 24);
          if ( !v34 )
            goto LABEL_48;
        }
        v32 = (_QWORD *)v31;
        v31 = *(_QWORD *)(v31 + 16);
      }
      while ( v33 );
LABEL_48:
      if ( v25 + 6 != v32 && dword_5015AC8 >= *((_DWORD *)v32 + 8) && *((int *)v32 + 9) > 0 )
        *(_BYTE *)(a1 + 61) = qword_5015B48;
    }
  }
  v35 = sub_C52410();
  v36 = (char *)(v35 + 1);
  v37 = sub_C959E0();
  v38 = (char *)v35[2];
  if ( v38 )
  {
    v39 = (char *)(v35 + 1);
    do
    {
      while ( 1 )
      {
        v40 = *((_QWORD *)v38 + 2);
        v41 = *((_QWORD *)v38 + 3);
        if ( v37 <= *((_QWORD *)v38 + 4) )
          break;
        v38 = (char *)*((_QWORD *)v38 + 3);
        if ( !v41 )
          goto LABEL_55;
      }
      v39 = v38;
      v38 = (char *)*((_QWORD *)v38 + 2);
    }
    while ( v40 );
LABEL_55:
    if ( v36 != v39 && v37 >= *((_QWORD *)v39 + 4) )
      v36 = v39;
  }
  result = (unsigned __int64)sub_C52410() + 8;
  if ( v36 != (char *)result )
  {
    result = *((_QWORD *)v36 + 7);
    if ( result )
    {
      v43 = v36 + 48;
      do
      {
        while ( 1 )
        {
          v44 = *(_QWORD *)(result + 16);
          v45 = *(_QWORD *)(result + 24);
          if ( *(_DWORD *)(result + 32) >= dword_5015BA8 )
            break;
          result = *(_QWORD *)(result + 24);
          if ( !v45 )
            goto LABEL_64;
        }
        v43 = (char *)result;
        result = *(_QWORD *)(result + 16);
      }
      while ( v44 );
LABEL_64:
      if ( v36 + 48 != v43 && dword_5015BA8 >= *((_DWORD *)v43 + 8) && *((int *)v43 + 9) > 0 )
      {
        result = (unsigned int)qword_5015C28;
        *(_DWORD *)(a1 + 64) = qword_5015C28;
      }
    }
  }
  return result;
}
