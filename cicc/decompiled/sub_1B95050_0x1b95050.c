// Function: sub_1B95050
// Address: 0x1b95050
//
__int64 __fastcall sub_1B95050(__int64 a1, unsigned int a2, int a3)
{
  __int64 *v4; // r15
  __int64 v5; // rdx
  __int64 *v6; // r14
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 *v9; // r13
  __int64 *v10; // rdx
  __int64 v11; // rcx
  _DWORD *v12; // rax
  _DWORD *v13; // rcx
  __int64 v14; // rcx
  __int64 *v15; // r8
  _DWORD *v16; // rax
  _DWORD *v17; // rcx
  __int64 v18; // rcx
  _DWORD *v19; // rax
  _DWORD *v20; // rcx
  __int64 v21; // rcx
  _DWORD *v22; // rax
  _DWORD *v23; // rcx
  __int64 v24; // rax
  __int64 *v25; // r12
  __int64 v26; // rdi
  _DWORD *v27; // rax
  _DWORD *v28; // rcx
  __int64 v29; // r15
  __int64 v30; // rbx
  __int64 *v31; // r15
  __int64 v32; // rdx
  __int64 v33; // r12
  __int64 v34; // r13
  __int64 v36; // rax
  __int64 v37; // r8
  __int64 v38; // rdi
  __int64 v39; // rsi
  __int64 v40; // rcx
  __int64 v41; // rax
  __int64 v42; // rdi
  __int64 v43; // rax
  __int64 v44; // r9
  __int64 v45; // rdi
  __int64 v46; // rsi
  __int64 v47; // rcx
  __int64 v48; // rax
  __int64 v49; // r9
  __int64 v50; // rdi
  __int64 v51; // rsi
  __int64 v52; // rcx
  __int64 v53; // rax
  __int64 v54; // r8
  __int64 v55; // rsi
  __int64 v56; // rcx
  __int64 v57; // rax
  __int64 v58; // rax
  signed __int64 v59; // [rsp+0h] [rbp-50h]
  unsigned int v61[13]; // [rsp+1Ch] [rbp-34h] BYREF

  *(_DWORD *)(a1 + 116) = a3;
  v4 = *(__int64 **)(a1 + 48);
  v5 = 8LL * *(unsigned int *)(a1 + 56);
  *(_DWORD *)(a1 + 112) = a2;
  v6 = &v4[(unsigned __int64)v5 / 8];
  v7 = v5 >> 3;
  v8 = v5 >> 5;
  if ( v8 )
  {
    v9 = v4;
    v10 = &v4[4 * v8];
    while ( 1 )
    {
      v11 = *v9;
      if ( *(_QWORD *)(*v9 + 72) )
      {
        v36 = *(_QWORD *)(v11 + 48);
        v37 = v11 + 40;
        if ( !v36 )
          goto LABEL_37;
        v38 = v11 + 40;
        do
        {
          while ( 1 )
          {
            v39 = *(_QWORD *)(v36 + 16);
            v40 = *(_QWORD *)(v36 + 24);
            if ( a2 <= *(_DWORD *)(v36 + 32) )
              break;
            v36 = *(_QWORD *)(v36 + 24);
            if ( !v40 )
              goto LABEL_66;
          }
          v38 = v36;
          v36 = *(_QWORD *)(v36 + 16);
        }
        while ( v39 );
LABEL_66:
        if ( v37 == v38 || a2 < *(_DWORD *)(v38 + 32) )
          goto LABEL_37;
        v14 = v9[1];
        v15 = v9 + 1;
        if ( *(_QWORD *)(v14 + 72) )
        {
LABEL_69:
          v41 = *(_QWORD *)(v14 + 48);
          if ( !v41 )
            goto LABEL_36;
          v42 = v14 + 40;
          do
          {
            if ( a2 > *(_DWORD *)(v41 + 32) )
            {
              v41 = *(_QWORD *)(v41 + 24);
            }
            else
            {
              v42 = v41;
              v41 = *(_QWORD *)(v41 + 16);
            }
          }
          while ( v41 );
          if ( v14 + 40 == v42 || a2 < *(_DWORD *)(v42 + 32) )
            goto LABEL_36;
          goto LABEL_15;
        }
      }
      else
      {
        v12 = *(_DWORD **)(v11 + 8);
        v13 = &v12[*(unsigned int *)(v11 + 16)];
        if ( v12 == v13 )
          goto LABEL_37;
        while ( a2 != *v12 )
        {
          if ( v13 == ++v12 )
            goto LABEL_37;
        }
        if ( v13 == v12 )
          goto LABEL_37;
        v14 = v9[1];
        v15 = v9 + 1;
        if ( *(_QWORD *)(v14 + 72) )
          goto LABEL_69;
      }
      v16 = *(_DWORD **)(v14 + 8);
      v17 = &v16[*(unsigned int *)(v14 + 16)];
      if ( v16 == v17 )
        goto LABEL_36;
      while ( a2 != *v16 )
      {
        if ( v17 == ++v16 )
          goto LABEL_36;
      }
      if ( v17 == v16 )
      {
LABEL_36:
        v9 = v15;
        goto LABEL_37;
      }
LABEL_15:
      v18 = v9[2];
      v15 = v9 + 2;
      if ( *(_QWORD *)(v18 + 72) )
      {
        v43 = *(_QWORD *)(v18 + 48);
        v44 = v18 + 40;
        if ( !v43 )
          goto LABEL_36;
        v45 = v18 + 40;
        do
        {
          while ( 1 )
          {
            v46 = *(_QWORD *)(v43 + 16);
            v47 = *(_QWORD *)(v43 + 24);
            if ( a2 <= *(_DWORD *)(v43 + 32) )
              break;
            v43 = *(_QWORD *)(v43 + 24);
            if ( !v47 )
              goto LABEL_80;
          }
          v45 = v43;
          v43 = *(_QWORD *)(v43 + 16);
        }
        while ( v46 );
LABEL_80:
        if ( v44 == v45 || a2 < *(_DWORD *)(v45 + 32) )
          goto LABEL_36;
        v21 = v9[3];
        v15 = v9 + 3;
        if ( !*(_QWORD *)(v21 + 72) )
        {
LABEL_22:
          v22 = *(_DWORD **)(v21 + 8);
          v23 = &v22[*(unsigned int *)(v21 + 16)];
          if ( v22 == v23 )
            goto LABEL_36;
          while ( a2 != *v22 )
          {
            if ( v23 == ++v22 )
              goto LABEL_36;
          }
          if ( v23 == v22 )
            goto LABEL_36;
          goto LABEL_27;
        }
      }
      else
      {
        v19 = *(_DWORD **)(v18 + 8);
        v20 = &v19[*(unsigned int *)(v18 + 16)];
        if ( v19 == v20 )
          goto LABEL_36;
        while ( a2 != *v19 )
        {
          if ( v20 == ++v19 )
            goto LABEL_36;
        }
        if ( v20 == v19 )
          goto LABEL_36;
        v21 = v9[3];
        v15 = v9 + 3;
        if ( !*(_QWORD *)(v21 + 72) )
          goto LABEL_22;
      }
      v48 = *(_QWORD *)(v21 + 48);
      v49 = v21 + 40;
      if ( !v48 )
        goto LABEL_36;
      v50 = v21 + 40;
      do
      {
        while ( 1 )
        {
          v51 = *(_QWORD *)(v48 + 16);
          v52 = *(_QWORD *)(v48 + 24);
          if ( a2 <= *(_DWORD *)(v48 + 32) )
            break;
          v48 = *(_QWORD *)(v48 + 24);
          if ( !v52 )
            goto LABEL_88;
        }
        v50 = v48;
        v48 = *(_QWORD *)(v48 + 16);
      }
      while ( v51 );
LABEL_88:
      if ( v49 == v50 || a2 < *(_DWORD *)(v50 + 32) )
        goto LABEL_36;
LABEL_27:
      v9 += 4;
      if ( v10 == v9 )
      {
        v7 = v6 - v9;
        goto LABEL_29;
      }
    }
  }
  v9 = v4;
LABEL_29:
  if ( v7 == 2 )
  {
LABEL_102:
    v58 = *v9;
    v61[0] = a2;
    if ( sub_1B94FB0(v58 + 8, v61) )
    {
      ++v9;
LABEL_32:
      v24 = *v9;
      v61[0] = a2;
      if ( sub_1B94FB0(v24 + 8, v61) )
        goto LABEL_60;
      goto LABEL_37;
    }
    goto LABEL_37;
  }
  if ( v7 != 3 )
  {
    if ( v7 != 1 )
      goto LABEL_60;
    goto LABEL_32;
  }
  v57 = *v9;
  v61[0] = a2;
  if ( sub_1B94FB0(v57 + 8, v61) )
  {
    ++v9;
    goto LABEL_102;
  }
LABEL_37:
  if ( v6 == v9 )
    goto LABEL_60;
  v25 = v9 + 1;
  if ( v6 == v9 + 1 )
  {
    v6 = v9;
    goto LABEL_56;
  }
  do
  {
    v26 = *v25;
    if ( *(_QWORD *)(*v25 + 72) )
    {
      v53 = *(_QWORD *)(v26 + 48);
      if ( v53 )
      {
        v54 = v26 + 40;
        do
        {
          while ( 1 )
          {
            v55 = *(_QWORD *)(v53 + 16);
            v56 = *(_QWORD *)(v53 + 24);
            if ( a2 <= *(_DWORD *)(v53 + 32) )
              break;
            v53 = *(_QWORD *)(v53 + 24);
            if ( !v56 )
              goto LABEL_96;
          }
          v54 = v53;
          v53 = *(_QWORD *)(v53 + 16);
        }
        while ( v55 );
LABEL_96:
        if ( v26 + 40 != v54 && a2 >= *(_DWORD *)(v54 + 32) )
        {
LABEL_45:
          *v25 = 0;
          v29 = *v9;
          *v9 = v26;
          if ( v29 )
          {
            sub_1B949D0(v29);
            j_j___libc_free_0(v29, 472);
          }
          ++v9;
        }
      }
    }
    else
    {
      v27 = *(_DWORD **)(v26 + 8);
      v28 = &v27[*(unsigned int *)(v26 + 16)];
      if ( v27 != v28 )
      {
        while ( a2 != *v27 )
        {
          if ( v28 == ++v27 )
            goto LABEL_48;
        }
        if ( v27 != v28 )
          goto LABEL_45;
      }
    }
LABEL_48:
    ++v25;
  }
  while ( v6 != v25 );
  v4 = *(__int64 **)(a1 + 48);
  v25 = &v4[*(unsigned int *)(a1 + 56)];
  v59 = (char *)v25 - (char *)v6;
  v30 = v25 - v6;
  if ( (char *)v25 - (char *)v6 <= 0 )
  {
    v6 = v9;
  }
  else
  {
    v31 = v9;
    do
    {
      v32 = *v6;
      *v6 = 0;
      v33 = *v31;
      *v31 = v32;
      if ( v33 )
      {
        sub_1B949D0(v33);
        j_j___libc_free_0(v33, 472);
      }
      ++v6;
      ++v31;
      --v30;
    }
    while ( v30 );
    v6 = (__int64 *)((char *)v9 + v59);
    v4 = *(__int64 **)(a1 + 48);
    v25 = &v4[*(unsigned int *)(a1 + 56)];
  }
  if ( v6 != v25 )
  {
    do
    {
LABEL_56:
      v34 = *--v25;
      if ( v34 )
      {
        sub_1B949D0(v34);
        j_j___libc_free_0(v34, 472);
      }
    }
    while ( v25 != v6 );
    v4 = *(__int64 **)(a1 + 48);
  }
LABEL_60:
  *(_DWORD *)(a1 + 56) = v6 - v4;
  return a1;
}
