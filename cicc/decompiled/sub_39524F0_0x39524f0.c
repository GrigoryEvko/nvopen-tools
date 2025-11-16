// Function: sub_39524F0
// Address: 0x39524f0
//
unsigned __int64 __fastcall sub_39524F0(unsigned int *a1)
{
  __int64 v2; // rcx
  _BYTE *v3; // r14
  char *v4; // rax
  char *v5; // r12
  unsigned __int64 v6; // rdi
  __int64 v7; // r12
  __int64 v8; // r15
  __int64 v9; // rax
  char *v10; // r13
  signed __int64 v11; // rdx
  __int64 v12; // rdx
  _BYTE *v13; // rsi
  _BYTE *v14; // rdx
  unsigned __int64 result; // rax
  __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // r12
  _QWORD *v19; // r14
  __int64 v20; // rax
  __int64 v21; // r13
  _BYTE *v22; // rsi
  _QWORD *v23; // rdx
  __int64 v24; // r13
  int v25; // r14d
  unsigned int v26; // edx
  unsigned __int64 v27; // r15
  __int64 v28; // rax
  unsigned int v29; // edx
  void *v30; // r12
  void *v31; // r8
  void *v32; // rax
  unsigned int v33; // esi
  unsigned __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // rdi
  _QWORD *v37; // rcx
  __int64 v38; // rdx
  unsigned int v39; // r11d
  unsigned int v40; // r11d
  __int64 v41; // r10
  __int64 v42; // rcx
  int v43; // edx
  _QWORD *v44; // rax
  __int64 v45; // rsi
  int v46; // r9d
  _QWORD *v47; // rdi
  unsigned int v48; // ecx
  unsigned int v49; // eax
  unsigned int v50; // r15d
  int v51; // r9d
  __int64 v52; // r10
  __int64 v53; // rcx
  __int64 v54; // rsi
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // [rsp+8h] [rbp-68h]
  __int64 v58; // [rsp+10h] [rbp-60h]
  __int64 v59; // [rsp+18h] [rbp-58h]
  unsigned __int64 v60; // [rsp+20h] [rbp-50h]
  int v61; // [rsp+20h] [rbp-50h]
  unsigned __int64 v62; // [rsp+20h] [rbp-50h]
  void *v63; // [rsp+20h] [rbp-50h]
  unsigned int v64; // [rsp+20h] [rbp-50h]
  unsigned int v65; // [rsp+2Ch] [rbp-44h]
  __int64 v66[7]; // [rsp+38h] [rbp-38h] BYREF

  v2 = *(_QWORD *)a1;
  v3 = (_BYTE *)*((_QWORD *)a1 + 5);
  v58 = (__int64)(a1 + 10);
  v4 = (char *)*((_QWORD *)a1 + 7);
  v5 = (char *)*((_QWORD *)a1 + 6);
  v6 = *(unsigned int *)(*(_QWORD *)a1 + 40LL);
  if ( v6 > (v4 - v3) >> 3 )
  {
    v7 = v5 - v3;
    v8 = 8 * v6;
    if ( *(_DWORD *)(v2 + 40) )
    {
      v9 = sub_22077B0(8 * v6);
      v3 = (_BYTE *)*((_QWORD *)a1 + 5);
      v10 = (char *)v9;
      v11 = *((_QWORD *)a1 + 6) - (_QWORD)v3;
      if ( v11 <= 0 )
        goto LABEL_4;
    }
    else
    {
      v11 = v7;
      v10 = 0;
      if ( v7 <= 0 )
      {
LABEL_4:
        if ( !v3 )
        {
LABEL_5:
          v5 = &v10[v7];
          v4 = &v10[v8];
          *((_QWORD *)a1 + 5) = v10;
          v2 = *(_QWORD *)a1;
          *((_QWORD *)a1 + 6) = v5;
          *((_QWORD *)a1 + 7) = &v10[v8];
          goto LABEL_6;
        }
LABEL_48:
        j_j___libc_free_0((unsigned __int64)v3);
        goto LABEL_5;
      }
    }
    memmove(v10, v3, v11);
    goto LABEL_48;
  }
LABEL_6:
  v12 = *(_QWORD *)(v2 + 56);
  v66[0] = v12;
  if ( v5 == v4 )
  {
    sub_39514B0(v58, v5, v66);
    v13 = (_BYTE *)*((_QWORD *)a1 + 6);
  }
  else
  {
    if ( v5 )
    {
      *(_QWORD *)v5 = v12;
      v5 = (char *)*((_QWORD *)a1 + 6);
    }
    v13 = v5 + 8;
    *((_QWORD *)a1 + 6) = v5 + 8;
  }
  v14 = (_BYTE *)*((_QWORD *)a1 + 5);
  result = (unsigned __int64)(a1 + 2);
  v65 = 0;
  v16 = 0;
  v57 = (__int64)(a1 + 2);
  if ( v13 != v14 )
  {
    while ( 1 )
    {
      v17 = *(_QWORD *)&v14[8 * v16];
      v18 = 8 * v16;
      v19 = *(_QWORD **)(v17 + 24);
      v20 = *(_QWORD *)(v17 + 32) - (_QWORD)v19;
      v21 = v20 >> 3;
      if ( v20 > 0 )
      {
        do
        {
          while ( 1 )
          {
            v22 = (_BYTE *)*((_QWORD *)a1 + 6);
            if ( v22 != *((_BYTE **)a1 + 7) )
              break;
            v23 = v19++;
            sub_3952360(v58, v22, v23);
            if ( !--v21 )
              goto LABEL_18;
          }
          if ( v22 )
          {
            *(_QWORD *)v22 = *v19;
            v22 = (_BYTE *)*((_QWORD *)a1 + 6);
          }
          ++v19;
          *((_QWORD *)a1 + 6) = v22 + 8;
          --v21;
        }
        while ( v21 );
LABEL_18:
        v14 = (_BYTE *)*((_QWORD *)a1 + 5);
      }
      v24 = **(_QWORD **)&v14[v18];
      v25 = *(_DWORD *)(*(_QWORD *)a1 + 40LL);
      v26 = (unsigned int)(v25 + 63) >> 6;
      v27 = 8LL * v26;
      v59 = v26;
      v28 = malloc(v27);
      v29 = (unsigned int)(v25 + 63) >> 6;
      v30 = (void *)v28;
      if ( !v28 )
      {
        if ( v27 || (v56 = malloc(1u), v29 = (unsigned int)(v25 + 63) >> 6, !v56) )
        {
          v64 = v29;
          sub_16BD1C0("Allocation failed", 1u);
          v29 = v64;
        }
        else
        {
          v30 = (void *)v56;
        }
      }
      if ( v29 )
        memset(v30, 0, v27);
      if ( v25 )
      {
        v31 = (void *)malloc(v27);
        if ( !v31 )
        {
          if ( v27 || (v55 = malloc(1u), v31 = 0, !v55) )
          {
            v63 = v31;
            sub_16BD1C0("Allocation failed", 1u);
            v31 = v63;
          }
          else
          {
            v31 = (void *)v55;
          }
        }
        v32 = memcpy(v31, v30, v27);
        v33 = a1[8];
        v34 = (unsigned __int64)v32;
        if ( !v33 )
        {
LABEL_29:
          ++*((_QWORD *)a1 + 1);
          goto LABEL_30;
        }
      }
      else
      {
        v33 = a1[8];
        v59 = 0;
        v34 = 0;
        if ( !v33 )
          goto LABEL_29;
      }
      v35 = *((_QWORD *)a1 + 2);
      v36 = (v33 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v37 = (_QWORD *)(v35 + 40 * v36);
      v38 = *v37;
      if ( v24 != *v37 )
        break;
LABEL_26:
      _libc_free(v34);
      _libc_free((unsigned __int64)v30);
      v14 = (_BYTE *)*((_QWORD *)a1 + 5);
      v16 = ++v65;
      result = (__int64)(*((_QWORD *)a1 + 6) - (_QWORD)v14) >> 3;
      if ( v65 >= result )
        return result;
    }
    v61 = 1;
    v44 = 0;
    while ( v38 != -8 )
    {
      if ( v44 || v38 != -16 )
        v37 = v44;
      LODWORD(v36) = (v33 - 1) & (v61 + v36);
      v38 = *(_QWORD *)(v35 + 40LL * (unsigned int)v36);
      if ( v24 == v38 )
        goto LABEL_26;
      ++v61;
      v44 = v37;
      v37 = (_QWORD *)(v35 + 40LL * (unsigned int)v36);
    }
    if ( !v44 )
      v44 = v37;
    v48 = a1[6];
    ++*((_QWORD *)a1 + 1);
    v43 = v48 + 1;
    if ( 4 * (v48 + 1) >= 3 * v33 )
    {
LABEL_30:
      v60 = v34;
      sub_3951640(v57, 2 * v33);
      v39 = a1[8];
      if ( !v39 )
        goto LABEL_77;
      v40 = v39 - 1;
      v41 = *((_QWORD *)a1 + 2);
      v34 = v60;
      LODWORD(v42) = v40 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v43 = a1[6] + 1;
      v44 = (_QWORD *)(v41 + 40LL * (unsigned int)v42);
      v45 = *v44;
      if ( v24 == *v44 )
        goto LABEL_43;
      v46 = 1;
      v47 = 0;
      while ( v45 != -8 )
      {
        if ( !v47 && v45 == -16 )
          v47 = v44;
        v42 = v40 & ((_DWORD)v42 + v46);
        v44 = (_QWORD *)(v41 + 40 * v42);
        v45 = *v44;
        if ( v24 == *v44 )
          goto LABEL_43;
        ++v46;
      }
    }
    else
    {
      if ( v33 - a1[7] - v43 > v33 >> 3 )
        goto LABEL_43;
      v62 = v34;
      sub_3951640(v57, v33);
      v49 = a1[8];
      if ( !v49 )
      {
LABEL_77:
        ++a1[6];
        BUG();
      }
      v50 = v49 - 1;
      v51 = 1;
      v52 = *((_QWORD *)a1 + 2);
      v34 = v62;
      v43 = a1[6] + 1;
      v47 = 0;
      LODWORD(v53) = (v49 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
      v44 = (_QWORD *)(v52 + 40LL * (unsigned int)v53);
      v54 = *v44;
      if ( v24 == *v44 )
        goto LABEL_43;
      while ( v54 != -8 )
      {
        if ( !v47 && v54 == -16 )
          v47 = v44;
        v53 = v50 & ((_DWORD)v53 + v51);
        v44 = (_QWORD *)(v52 + 40 * v53);
        v54 = *v44;
        if ( v24 == *v44 )
          goto LABEL_43;
        ++v51;
      }
    }
    if ( v47 )
      v44 = v47;
LABEL_43:
    a1[6] = v43;
    if ( *v44 != -8 )
      --a1[7];
    v44[2] = v34;
    v34 = 0;
    *v44 = v24;
    *((_DWORD *)v44 + 2) = v65;
    *((_DWORD *)v44 + 8) = v25;
    v44[3] = v59;
    goto LABEL_26;
  }
  return result;
}
