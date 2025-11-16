// Function: sub_813790
// Address: 0x813790
//
void __fastcall sub_813790(unsigned __int64 a1, char a2, unsigned int a3, __int64 *a4, __int64 a5)
{
  __int64 v8; // rax
  __int64 v9; // r15
  int v10; // r13d
  _BOOL4 v11; // edx
  char v12; // cl
  __int64 v13; // r15
  char *v14; // rdi
  char v15; // dl
  __int64 v16; // rax
  char *v17; // rdi
  _QWORD *v18; // rdi
  __int64 v19; // rax
  __int64 v20; // rsi
  unsigned int v21; // ecx
  __int64 v22; // r8
  __int64 k; // rax
  _QWORD *v24; // rdx
  int v25; // r10d
  __int64 v26; // r9
  unsigned int v27; // edi
  unsigned __int64 v28; // r8
  char v29; // r12
  unsigned int v30; // ecx
  unsigned __int64 *v31; // rdx
  unsigned __int64 *v32; // rax
  unsigned __int64 v33; // rbx
  __int64 v34; // rcx
  __int64 v35; // rax
  int v36; // eax
  __int64 v37; // r10
  __int64 v38; // r10
  unsigned __int64 v39; // rax
  int v40; // r8d
  __int64 v41; // rdi
  unsigned int v42; // eax
  __int64 v43; // rsi
  __int64 v44; // rdx
  unsigned __int64 v45; // rcx
  __int64 v46; // r15
  int v47; // eax
  unsigned int v48; // r13d
  _QWORD *v49; // rax
  unsigned int v50; // r11d
  _QWORD *v51; // r10
  _QWORD *v52; // rcx
  _QWORD *v53; // rdx
  __int64 *v54; // r8
  unsigned __int64 *v55; // rsi
  unsigned __int64 v56; // rdi
  unsigned __int64 j; // rdx
  unsigned int v58; // edx
  unsigned __int64 *v59; // rax
  unsigned __int64 *v60; // rdx
  __int64 v61; // rsi
  char v62; // al
  __int64 v63; // rax
  char v64; // si
  int v65; // eax
  __int64 v66; // r15
  int v67; // eax
  _QWORD *v68; // rax
  _QWORD *v69; // rdx
  unsigned __int64 *v70; // rsi
  unsigned __int64 v71; // rdi
  unsigned __int64 i; // rdx
  unsigned int v73; // edx
  unsigned __int64 *v74; // rax
  __int64 v75; // rax
  char *v76; // rdi
  __int64 v77; // r13
  __int64 v78; // [rsp+10h] [rbp-70h]
  __int64 v79; // [rsp+18h] [rbp-68h]
  _QWORD *v80; // [rsp+18h] [rbp-68h]
  _QWORD *v81; // [rsp+18h] [rbp-68h]
  __int64 v82; // [rsp+20h] [rbp-60h]
  unsigned __int8 v85; // [rsp+38h] [rbp-48h]
  _QWORD v86[7]; // [rsp+48h] [rbp-38h] BYREF

  if ( a2 != 6 )
    goto LABEL_2;
  v38 = qword_4F18BD0;
  v39 = a1 >> 3;
  v40 = *(_DWORD *)(qword_4F18BD0 + 8);
  v41 = *(_QWORD *)qword_4F18BD0;
  v42 = v40 & v39;
  v43 = v42;
  v44 = *(_QWORD *)qword_4F18BD0 + 16LL * v42;
  v45 = *(_QWORD *)v44;
  if ( *(_QWORD *)v44 )
  {
    do
    {
      if ( a1 == v45 )
      {
        v61 = v41 + 16 * v43;
        v62 = *(_BYTE *)(v61 + 8);
        *(_BYTE *)(v61 + 8) = 1;
        if ( v62 )
          return;
        goto LABEL_103;
      }
      v42 = v40 & (v42 + 1);
      v43 = v42;
      v60 = (unsigned __int64 *)(v41 + 16LL * v42);
      v45 = *v60;
    }
    while ( *v60 );
    *v60 = a1;
    if ( a1 )
      *((_BYTE *)v60 + 8) = 1;
    v66 = *(unsigned int *)(v38 + 8);
    v67 = *(_DWORD *)(v38 + 12) + 1;
    *(_DWORD *)(v38 + 12) = v67;
    if ( 2 * v67 > (unsigned int)v66 )
    {
      v81 = (_QWORD *)v38;
      v48 = 2 * v66 + 1;
      v68 = (_QWORD *)sub_823970(16LL * (unsigned int)(2 * v66 + 2));
      v50 = v66 + 1;
      v51 = v81;
      v52 = v68;
      if ( 2 * (_DWORD)v66 != -2 )
      {
        v69 = &v68[2 * v48 + 2];
        do
        {
          if ( v68 )
            *v68 = 0;
          v68 += 2;
        }
        while ( v69 != v68 );
      }
      v54 = (__int64 *)*v81;
      if ( (_DWORD)v66 != -1 )
      {
        v70 = (unsigned __int64 *)*v81;
        do
        {
          v71 = *v70;
          if ( *v70 )
          {
            for ( i = v71 >> 3; ; LODWORD(i) = v73 + 1 )
            {
              v73 = v48 & i;
              v74 = &v52[2 * v73];
              if ( !*v74 )
                break;
            }
            *v74 = v71;
            *((_BYTE *)v74 + 8) = *((_BYTE *)v70 + 8);
          }
          v70 += 2;
        }
        while ( &v54[2 * v66 + 2] != (__int64 *)v70 );
      }
      goto LABEL_156;
    }
  }
  else
  {
    *(_QWORD *)v44 = a1;
    if ( a1 )
      *(_BYTE *)(v44 + 8) = 1;
    v46 = *(unsigned int *)(v38 + 8);
    v47 = *(_DWORD *)(v38 + 12) + 1;
    *(_DWORD *)(v38 + 12) = v47;
    if ( 2 * v47 > (unsigned int)v46 )
    {
      v80 = (_QWORD *)v38;
      v48 = 2 * v46 + 1;
      v49 = (_QWORD *)sub_823970(16LL * (unsigned int)(2 * v46 + 2));
      v50 = v46 + 1;
      v51 = v80;
      v52 = v49;
      if ( 2 * (_DWORD)v46 != -2 )
      {
        v53 = &v49[2 * v48 + 2];
        do
        {
          if ( v49 )
            *v49 = 0;
          v49 += 2;
        }
        while ( v53 != v49 );
      }
      v54 = (__int64 *)*v80;
      if ( (_DWORD)v46 != -1 )
      {
        v55 = (unsigned __int64 *)*v80;
        do
        {
          v56 = *v55;
          if ( *v55 )
          {
            for ( j = v56 >> 3; ; LODWORD(j) = v58 + 1 )
            {
              v58 = v48 & j;
              v59 = &v52[2 * v58];
              if ( !*v59 )
                break;
            }
            *v59 = v56;
            *((_BYTE *)v59 + 8) = *((_BYTE *)v55 + 8);
          }
          v55 += 2;
        }
        while ( v55 != (unsigned __int64 *)&v54[2 * v46 + 2] );
      }
LABEL_156:
      *v51 = v52;
      *((_DWORD *)v51 + 2) = v48;
      sub_823A00(v54, 16LL * v50);
    }
  }
LABEL_103:
  if ( *(_BYTE *)(a1 + 140) != 9
    || (v63 = *(_QWORD *)(a1 + 168), (*(_BYTE *)(v63 + 109) & 0x20) == 0)
    || (*(_BYTE *)(v63 + 110) & 0x18) == 0 )
  {
LABEL_2:
    v8 = *(_QWORD *)(a1 + 40);
    v85 = *(_BYTE *)(a1 + 89) & 4;
    if ( v85 )
    {
      v9 = *(_QWORD *)(v8 + 32);
      v82 = v9;
      v79 = *(_QWORD *)(v9 + 168);
      v85 = 6;
      v10 = sub_80AFC0(v9, 6);
      goto LABEL_4;
    }
    if ( !v8 )
      goto LABEL_34;
    v15 = *(_BYTE *)(v8 + 28);
    if ( v15 == 16 )
    {
      v9 = *(_QWORD *)(v8 + 32);
      v82 = v9;
      v85 = 6;
      v79 = 0;
      v10 = sub_80AFC0(v9, 6);
LABEL_4:
      v11 = 0;
      if ( v10 )
        goto LABEL_10;
      goto LABEL_5;
    }
    if ( v15 != 3 )
    {
LABEL_34:
      if ( a4 )
      {
        v79 = 0;
        v9 = 0;
        v11 = 1;
        v82 = 0;
        goto LABEL_36;
      }
      v82 = 0;
      v10 = 0;
      v11 = 1;
LABEL_161:
      v9 = 0;
      if ( a2 != 6 )
        goto LABEL_13;
      v79 = 0;
      v9 = 0;
      goto LABEL_37;
    }
    v9 = *(_QWORD *)(*(_QWORD *)(v8 + 32) + 40LL);
    v82 = *(_QWORD *)(v8 + 32);
    if ( v9 )
    {
      if ( *(_BYTE *)(v9 + 28) == 3 )
      {
        v85 = 28;
        v10 = 1;
        v11 = 0;
        goto LABEL_161;
      }
      if ( a3 )
      {
        v85 = 28;
        v9 = 0;
        v10 = 1;
        v11 = 0;
        v79 = 0;
        goto LABEL_10;
      }
      if ( a4 )
      {
        v79 = 0;
        v11 = 0;
        v9 = 0;
        v85 = 28;
        goto LABEL_36;
      }
      v79 = 0;
      v9 = 0;
    }
    else
    {
      if ( a3 )
      {
        v79 = 0;
        v10 = 1;
        v11 = 0;
        v85 = 28;
        goto LABEL_10;
      }
      v79 = 0;
      if ( a4 )
      {
        v85 = 28;
        v11 = 0;
        goto LABEL_36;
      }
    }
    v85 = 28;
    v10 = 0;
    v11 = 0;
    goto LABEL_10;
  }
  v82 = *(_QWORD *)(v63 + 240);
  if ( (*(_BYTE *)(v63 + 110) & 0x10) != 0 )
  {
    v85 = 8;
    v64 = 8;
  }
  else
  {
    v85 = 7;
    v64 = 7;
    if ( (*(_BYTE *)(v82 + 170) & 2) != 0 )
      v82 = *(_QWORD *)(*(_QWORD *)(v82 + 128) + 16LL);
  }
  v9 = 0;
  v65 = sub_80AFC0(v82, v64);
  v79 = 0;
  v11 = 0;
  v10 = v65;
  if ( !v65 )
  {
LABEL_5:
    if ( a3 )
    {
      v10 = 1;
      goto LABEL_10;
    }
    if ( !a4 )
      goto LABEL_10;
    if ( v9 )
    {
      v11 = sub_80AA60(v9);
      if ( v11 )
      {
        v11 = 0;
        *a4 = 0;
        goto LABEL_10;
      }
    }
LABEL_36:
    v10 = 0;
    *a4 = v82;
    if ( a2 != 6 )
      goto LABEL_11;
    goto LABEL_37;
  }
LABEL_10:
  if ( a2 != 6 )
  {
LABEL_11:
    if ( (*(_BYTE *)(a1 + 89) & 4) != 0 )
    {
      if ( (unsigned int)sub_80C5A0(v9, 6, 0, 0, v86, (_QWORD *)a5) )
        goto LABEL_29;
      v35 = sub_809820(v9);
      if ( v35 )
      {
        v78 = v35;
        v36 = sub_80C5A0(v35, 59, 0, 0, v86, (_QWORD *)a5);
        v37 = v78;
        if ( v36 )
        {
          v86[0] = *(_QWORD *)(v79 + 168);
          sub_811CB0(v86, 0, 0, (_QWORD *)a5);
          goto LABEL_80;
        }
        if ( v10 )
        {
          sub_813790(v82, v85, a3, a4, a5);
          v37 = v78;
        }
        if ( !*(_QWORD *)(a5 + 40) )
          sub_80A250(v37, 59, 0, a5);
      }
      else if ( v10 )
      {
        sub_813790(v82, v85, a3, a4, a5);
      }
      if ( dword_4D0425C && unk_4D04250 <= 0x76BFu )
      {
        v77 = *(_QWORD *)(*(_QWORD *)(v9 + 168) + 168LL);
        sub_810560(v9, (_QWORD *)a5);
        if ( v77 )
        {
          v86[0] = v77;
          sub_811CB0(v86, 0, 0, (_QWORD *)a5);
        }
      }
      else
      {
        sub_813620(v9, (_QWORD *)a5);
      }
      goto LABEL_80;
    }
    v8 = *(_QWORD *)(a1 + 40);
LABEL_13:
    if ( !v8 )
      goto LABEL_16;
    v12 = *(_BYTE *)(v8 + 28);
    if ( v12 != 16 )
    {
      if ( v12 == 3 )
      {
        if ( !v11 )
        {
          v13 = *(_QWORD *)(v8 + 32);
LABEL_18:
          if ( a3 && (unsigned int)sub_80C5A0(v13, 28, 0, 1, v86, (_QWORD *)a5) && (*(_BYTE *)(v13 + 124) & 0x10) != 0 )
          {
            v75 = sub_80A110((_QWORD *)a1, a5);
            if ( *(_DWORD *)(a5 + 48) )
              goto LABEL_29;
            if ( (*(_BYTE *)(v75 + 89) & 8) != 0 )
              v76 = *(char **)(v75 + 24);
            else
              v76 = *(char **)(v75 + 8);
            sub_80BC40(v76, (_QWORD *)a5);
          }
          if ( !(unsigned int)sub_80C5A0(v13, 28, 0, 0, v86, (_QWORD *)a5) )
          {
            if ( v10 )
              sub_813790(v82, v85, a3, a4, a5);
            if ( (*(_BYTE *)(v13 + 89) & 8) != 0 )
              v14 = *(char **)(v13 + 24);
            else
              v14 = *(char **)(v13 + 8);
            if ( !v14 )
            {
              v14 = *(char **)(v13 + 8);
              if ( !v14 )
                v14 = (char *)sub_80B070(v13, a5);
            }
            sub_80BC40(v14, (_QWORD *)a5);
            if ( !*(_QWORD *)(a5 + 40) )
              sub_80A250(v13, 28, 0, a5);
          }
LABEL_29:
          if ( a2 != 6 )
            return;
          goto LABEL_51;
        }
LABEL_17:
        v13 = sub_80A110((_QWORD *)a1, a5);
        if ( *(_DWORD *)(a5 + 48) )
          goto LABEL_29;
        goto LABEL_18;
      }
LABEL_16:
      if ( !v11 )
        goto LABEL_29;
      goto LABEL_17;
    }
    if ( (unsigned int)sub_80C5A0(v9, 6, 0, 0, v86, (_QWORD *)a5) )
      goto LABEL_29;
    if ( v10 )
      sub_813790(v82, v85, a3, a4, a5);
    sub_810560(v9, (_QWORD *)a5);
LABEL_80:
    if ( !*(_QWORD *)(a5 + 40) )
      sub_80A250(v9, 6, 0, a5);
    goto LABEL_29;
  }
LABEL_37:
  if ( *(_BYTE *)(a1 + 140) != 9 )
    goto LABEL_11;
  v16 = *(_QWORD *)(a1 + 168);
  if ( (*(_BYTE *)(v16 + 109) & 0x20) == 0 || (*(_BYTE *)(v16 + 110) & 0x18) == 0 )
    goto LABEL_11;
  if ( !(unsigned int)sub_80C5A0(v82, v85, 0, 0, v86, (_QWORD *)a5) )
  {
    if ( v10 )
      sub_813790(v82, v85, a3, a4, a5);
    if ( (*(_BYTE *)(v82 + 89) & 8) != 0 )
      v17 = *(char **)(v82 + 24);
    else
      v17 = *(char **)(v82 + 8);
    sub_80BC40(v17, (_QWORD *)a5);
    if ( !*(_QWORD *)(a5 + 40) )
      sub_80A250(v82, v85, 0, a5);
    if ( v85 == 7 && (*(_BYTE *)(v82 + 170) & 0x10) != 0 && **(_QWORD **)(v82 + 216) )
    {
      v86[0] = **(_QWORD **)(v82 + 216);
      sub_811CB0(v86, 0, 0, (_QWORD *)a5);
    }
    else
    {
      v18 = (_QWORD *)qword_4F18BE0;
      ++*(_QWORD *)a5;
      v19 = v18[2];
      if ( (unsigned __int64)(v19 + 1) > v18[1] )
      {
        sub_823810(v18);
        v18 = (_QWORD *)qword_4F18BE0;
        v19 = *(_QWORD *)(qword_4F18BE0 + 16);
      }
      *(_BYTE *)(v18[4] + v19) = 77;
      ++v18[2];
    }
  }
LABEL_51:
  v20 = qword_4F18BD0;
  v21 = *(_DWORD *)(qword_4F18BD0 + 8);
  v22 = *(_QWORD *)qword_4F18BD0;
  for ( k = v21 & (unsigned int)(a1 >> 3); ; k = v21 & ((_DWORD)k + 1) )
  {
    v24 = (_QWORD *)(v22 + 16LL * (unsigned int)k);
    if ( a1 == *v24 )
      break;
  }
  *v24 = 0;
  if ( *(_QWORD *)(v22 + 16LL * (((_DWORD)k + 1) & v21)) )
  {
    v25 = *(_DWORD *)(v20 + 8);
    v26 = *(_QWORD *)v20;
    v27 = v25 & (k + 1);
    v28 = *(_QWORD *)(*(_QWORD *)v20 + 16LL * v27);
    while ( 1 )
    {
      v30 = v25 & (v28 >> 3);
      v31 = (unsigned __int64 *)(v26 + 16LL * (v25 & (v27 + 1)));
      if ( v30 <= (unsigned int)k && (v27 < v30 || v27 > (unsigned int)k) || v27 > (unsigned int)k && v27 < v30 )
      {
        v32 = (unsigned __int64 *)(v26 + 16 * k);
        v33 = *v32;
        v34 = v26 + 16LL * v27;
        if ( *v32 )
        {
          *v32 = v28;
          v29 = *((_BYTE *)v32 + 8);
          if ( v28 )
            *((_BYTE *)v32 + 8) = *(_BYTE *)(v34 + 8);
          *(_QWORD *)v34 = v33;
          *(_BYTE *)(v34 + 8) = v29;
        }
        else
        {
          *v32 = v28;
          if ( v28 )
            *((_BYTE *)v32 + 8) = *(_BYTE *)(v34 + 8);
          *(_QWORD *)v34 = 0;
        }
        v28 = *v31;
        if ( !*v31 )
          break;
        k = v27;
      }
      else
      {
        v28 = *v31;
        if ( !*v31 )
          break;
      }
      v27 = v25 & (v27 + 1);
    }
  }
  --*(_DWORD *)(v20 + 12);
}
