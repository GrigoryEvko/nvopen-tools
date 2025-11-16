// Function: sub_1B04FC0
// Address: 0x1b04fc0
//
__int64 __fastcall sub_1B04FC0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r13
  __int64 v9; // rax
  __int64 *v10; // r15
  __int64 *v11; // rbx
  __int64 *v12; // r9
  __int64 *v13; // r8
  __int64 v14; // rsi
  __int64 *v15; // rdi
  unsigned int v16; // r10d
  __int64 *v17; // rax
  __int64 *v18; // rcx
  __int64 *v19; // rcx
  __int64 v20; // rax
  __int64 v21; // r15
  __int64 *v22; // r13
  __int64 v23; // rbx
  _QWORD *v24; // r14
  _QWORD *v25; // rax
  __int64 v26; // rax
  _QWORD *v27; // rdx
  __int64 v28; // r12
  __int64 *v29; // rax
  __int64 *v30; // rax
  __int64 v31; // rdx
  __int64 *v32; // rcx
  __int64 v33; // rdi
  __int64 *v34; // rbx
  __int64 v36; // rdx
  __int64 v37; // rsi
  __int64 *v38; // rax
  __int64 *v39; // r13
  __int64 *v40; // rax
  unsigned __int64 v41; // r14
  unsigned int v42; // r15d
  _QWORD *v43; // rax
  _QWORD *v44; // r9
  __int64 v45; // rax
  __int64 v46; // rax
  _QWORD *v47; // rdx
  __int64 v48; // rsi
  __int64 v49; // rdx
  __int64 v50; // rsi
  __int64 *v51; // rdi
  unsigned int v52; // r10d
  __int64 *v53; // rsi
  __int64 *v54; // rsi
  _QWORD *v55; // rdx
  _QWORD *v56; // rdx
  __int64 v57; // rax
  __int64 v60; // [rsp+10h] [rbp-50h]
  __int64 v61; // [rsp+18h] [rbp-48h]
  int v62; // [rsp+18h] [rbp-48h]
  __int64 v63; // [rsp+20h] [rbp-40h]
  _QWORD *v64; // [rsp+20h] [rbp-40h]
  __int64 *v66; // [rsp+28h] [rbp-38h]

  v7 = a2;
  v9 = sub_13FCB50(a2);
  v10 = *(__int64 **)(a2 + 32);
  v11 = *(__int64 **)(a2 + 40);
  v61 = v9;
  if ( v10 != v11 )
  {
    v12 = *(__int64 **)(a4 + 16);
    v13 = *(__int64 **)(a4 + 8);
    do
    {
LABEL_5:
      v14 = *v10;
      if ( v13 != v12 )
        goto LABEL_3;
      v15 = &v13[*(unsigned int *)(a4 + 28)];
      v16 = *(_DWORD *)(a4 + 28);
      if ( v15 != v13 )
      {
        v17 = v13;
        v18 = 0;
        while ( v14 != *v17 )
        {
          if ( *v17 == -2 )
            v18 = v17;
          if ( v15 == ++v17 )
          {
            if ( !v18 )
              goto LABEL_93;
            ++v10;
            *v18 = v14;
            v12 = *(__int64 **)(a4 + 16);
            --*(_DWORD *)(a4 + 32);
            v13 = *(__int64 **)(a4 + 8);
            ++*(_QWORD *)a4;
            if ( v11 != v10 )
              goto LABEL_5;
            goto LABEL_14;
          }
        }
        goto LABEL_4;
      }
LABEL_93:
      if ( v16 < *(_DWORD *)(a4 + 24) )
      {
        *(_DWORD *)(a4 + 28) = v16 + 1;
        *v15 = v14;
        v13 = *(__int64 **)(a4 + 8);
        ++*(_QWORD *)a4;
        v12 = *(__int64 **)(a4 + 16);
      }
      else
      {
LABEL_3:
        sub_16CCBA0(a4, v14);
        v12 = *(__int64 **)(a4 + 16);
        v13 = *(__int64 **)(a4 + 8);
      }
LABEL_4:
      ++v10;
    }
    while ( v11 != v10 );
  }
LABEL_14:
  v19 = *(__int64 **)(a1 + 32);
  v66 = *(__int64 **)(a1 + 40);
  if ( v19 != v66 )
  {
    v20 = v7;
    v21 = v7 + 56;
    v63 = a3;
    v22 = v19;
    v23 = v20;
    while ( 1 )
    {
      while ( 1 )
      {
        v27 = *(_QWORD **)(v23 + 72);
        v25 = *(_QWORD **)(v23 + 64);
        v28 = *v22;
        if ( v27 == v25 )
        {
          v24 = &v25[*(unsigned int *)(v23 + 84)];
          if ( v25 == v24 )
          {
            v56 = *(_QWORD **)(v23 + 64);
          }
          else
          {
            do
            {
              if ( v28 == *v25 )
                break;
              ++v25;
            }
            while ( v24 != v25 );
            v56 = v24;
          }
LABEL_29:
          while ( v56 != v25 )
          {
            if ( *v25 < 0xFFFFFFFFFFFFFFFELL )
              goto LABEL_19;
            ++v25;
          }
          if ( v24 != v25 )
            goto LABEL_20;
        }
        else
        {
          v24 = &v27[*(unsigned int *)(v23 + 80)];
          v25 = sub_16CC9F0(v21, *v22);
          if ( v28 == *v25 )
          {
            v36 = *(_QWORD *)(v23 + 72);
            if ( v36 == *(_QWORD *)(v23 + 64) )
              v37 = *(unsigned int *)(v23 + 84);
            else
              v37 = *(unsigned int *)(v23 + 80);
            v56 = (_QWORD *)(v36 + 8 * v37);
            goto LABEL_29;
          }
          v26 = *(_QWORD *)(v23 + 72);
          if ( v26 == *(_QWORD *)(v23 + 64) )
          {
            v25 = (_QWORD *)(v26 + 8LL * *(unsigned int *)(v23 + 84));
            v56 = v25;
            goto LABEL_29;
          }
          v25 = (_QWORD *)(v26 + 8LL * *(unsigned int *)(v23 + 80));
LABEL_19:
          if ( v24 != v25 )
            goto LABEL_20;
        }
        if ( sub_15CC8F0(a6, v61, v28) )
          break;
        v38 = *(__int64 **)(v63 + 8);
        if ( *(__int64 **)(v63 + 16) == v38 )
        {
          v51 = &v38[*(unsigned int *)(v63 + 28)];
          v52 = *(_DWORD *)(v63 + 28);
          if ( v38 != v51 )
          {
            v53 = 0;
            while ( v28 != *v38 )
            {
              if ( *v38 == -2 )
                v53 = v38;
              if ( v51 == ++v38 )
              {
                if ( !v53 )
                  goto LABEL_98;
                *v53 = v28;
                --*(_DWORD *)(v63 + 32);
                ++*(_QWORD *)v63;
                goto LABEL_20;
              }
            }
            goto LABEL_20;
          }
LABEL_98:
          v57 = v63;
          if ( v52 < *(_DWORD *)(v63 + 24) )
            goto LABEL_97;
        }
        sub_16CCBA0(v63, v28);
LABEL_20:
        if ( v66 == ++v22 )
          goto LABEL_34;
      }
      v29 = *(__int64 **)(a5 + 8);
      if ( *(__int64 **)(a5 + 16) == v29 )
      {
        v51 = &v29[*(unsigned int *)(a5 + 28)];
        v52 = *(_DWORD *)(a5 + 28);
        if ( v29 != v51 )
        {
          v54 = 0;
          while ( v28 != *v29 )
          {
            if ( *v29 == -2 )
              v54 = v29;
            if ( v51 == ++v29 )
            {
              if ( !v54 )
                goto LABEL_96;
              *v54 = v28;
              --*(_DWORD *)(a5 + 32);
              ++*(_QWORD *)a5;
              goto LABEL_20;
            }
          }
          goto LABEL_20;
        }
LABEL_96:
        v57 = a5;
        if ( v52 >= *(_DWORD *)(a5 + 24) )
          goto LABEL_33;
LABEL_97:
        *(_DWORD *)(v57 + 28) = v52 + 1;
        *v51 = v28;
        ++*(_QWORD *)v57;
        goto LABEL_20;
      }
LABEL_33:
      ++v22;
      sub_16CCBA0(a5, v28);
      if ( v66 == v22 )
      {
LABEL_34:
        a3 = v63;
        v7 = v23;
        break;
      }
    }
  }
  v60 = sub_13FC520(v7);
  v30 = *(__int64 **)(a3 + 16);
  if ( v30 == *(__int64 **)(a3 + 8) )
    v31 = *(unsigned int *)(a3 + 28);
  else
    v31 = *(unsigned int *)(a3 + 24);
  v32 = &v30[v31];
  if ( v30 == v32 )
    return 1;
  while ( 1 )
  {
    v33 = *v30;
    v34 = v30;
    if ( (unsigned __int64)*v30 < 0xFFFFFFFFFFFFFFFELL )
      break;
    if ( v32 == ++v30 )
      return 1;
  }
  if ( v32 == v30 )
    return 1;
  v39 = v32;
  if ( v60 != v33 )
    goto LABEL_55;
  do
  {
    do
    {
LABEL_49:
      v40 = v34 + 1;
      if ( v34 + 1 == v39 )
        return 1;
      v33 = *v40;
      for ( ++v34; (unsigned __int64)*v40 >= 0xFFFFFFFFFFFFFFFELL; v34 = v40 )
      {
        if ( v39 == ++v40 )
          return 1;
        v33 = *v40;
      }
      if ( v39 == v34 )
        return 1;
    }
    while ( v60 == v33 );
LABEL_55:
    v41 = sub_157EBA0(v33);
    v62 = sub_15F4D60(v41);
  }
  while ( !v62 );
  v42 = 0;
  while ( 1 )
  {
    v46 = sub_15F4DF0(v41, v42);
    v47 = *(_QWORD **)(a3 + 16);
    v48 = v46;
    v43 = *(_QWORD **)(a3 + 8);
    if ( v47 != v43 )
      break;
    v44 = &v43[*(unsigned int *)(a3 + 28)];
    if ( v43 == v44 )
    {
      v55 = *(_QWORD **)(a3 + 8);
    }
    else
    {
      do
      {
        if ( v48 == *v43 )
          break;
        ++v43;
      }
      while ( v44 != v43 );
      v55 = v44;
    }
LABEL_70:
    while ( v55 != v43 )
    {
      if ( *v43 < 0xFFFFFFFFFFFFFFFELL )
        goto LABEL_60;
      ++v43;
    }
    if ( v44 == v43 )
      return 0;
LABEL_61:
    if ( v62 == ++v42 )
      goto LABEL_49;
  }
  v64 = &v47[*(unsigned int *)(a3 + 24)];
  v43 = sub_16CC9F0(a3, v48);
  v44 = v64;
  if ( v48 == *v43 )
  {
    v49 = *(_QWORD *)(a3 + 16);
    if ( v49 == *(_QWORD *)(a3 + 8) )
      v50 = *(unsigned int *)(a3 + 28);
    else
      v50 = *(unsigned int *)(a3 + 24);
    v55 = (_QWORD *)(v49 + 8 * v50);
    goto LABEL_70;
  }
  v45 = *(_QWORD *)(a3 + 16);
  if ( v45 == *(_QWORD *)(a3 + 8) )
  {
    v43 = (_QWORD *)(v45 + 8LL * *(unsigned int *)(a3 + 28));
    v55 = v43;
    goto LABEL_70;
  }
  v43 = (_QWORD *)(v45 + 8LL * *(unsigned int *)(a3 + 24));
LABEL_60:
  if ( v44 != v43 )
    goto LABEL_61;
  return 0;
}
