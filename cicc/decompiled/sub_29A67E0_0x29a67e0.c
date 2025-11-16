// Function: sub_29A67E0
// Address: 0x29a67e0
//
__int64 __fastcall sub_29A67E0(unsigned int **a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 v6; // rbx
  unsigned __int64 v7; // rax
  __int64 v8; // rdx
  _QWORD *v9; // rax
  _QWORD *i; // rdx
  __int64 v11; // r9
  __int64 result; // rax
  __int64 v13; // rbx
  _QWORD *v14; // r13
  __int64 v15; // r8
  __int64 v16; // rdi
  __int64 v17; // rsi
  __int64 v18; // r12
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rsi
  __int64 v22; // rdi
  int *v23; // rax
  unsigned __int64 v24; // rdi
  int *v25; // r8
  __int64 v26; // rsi
  __int64 v27; // rcx
  __int64 *v28; // r10
  _QWORD *v29; // r9
  unsigned int v30; // ecx
  __int64 v31; // rsi
  __int64 v32; // rax
  _QWORD *v33; // r8
  unsigned __int64 v34; // rdx
  int *v35; // rbx
  int *v36; // rax
  bool v37; // al
  __int64 v38; // rbx
  __int64 v39; // rax
  unsigned int v40; // ecx
  __int64 v41; // r9
  __int64 v42; // rax
  __int64 v43; // rdx
  unsigned __int64 v44; // r9
  __int64 *v45; // r10
  __int64 v46; // rbx
  char v47; // di
  int *v48; // rcx
  __int64 v49; // rdi
  __int64 v50; // rsi
  __int64 v51; // rcx
  int *v52; // rdi
  int *v53; // rax
  unsigned __int64 v54; // r14
  int *v55; // r15
  unsigned __int64 v56; // r13
  _QWORD *v57; // rdi
  unsigned __int64 v58; // rdi
  _QWORD *v59; // rax
  __int64 v60; // rax
  unsigned int v61; // [rsp+Ch] [rbp-84h]
  int *v62; // [rsp+10h] [rbp-80h]
  unsigned __int64 v63; // [rsp+10h] [rbp-80h]
  __int64 *v64; // [rsp+18h] [rbp-78h]
  int *v65; // [rsp+18h] [rbp-78h]
  __int64 v66; // [rsp+20h] [rbp-70h]
  int *v69; // [rsp+38h] [rbp-58h]
  unsigned int v70; // [rsp+38h] [rbp-58h]
  __int64 v71; // [rsp+38h] [rbp-58h]
  __int64 *v72; // [rsp+38h] [rbp-58h]
  __int64 v73; // [rsp+40h] [rbp-50h]
  int *v74; // [rsp+48h] [rbp-48h]
  __int64 v75[7]; // [rsp+58h] [rbp-38h] BYREF

  v6 = **a1;
  v7 = *(unsigned int *)(a2 + 32);
  if ( v6 != v7 )
  {
    if ( v6 < v7 )
    {
      *(_DWORD *)(a2 + 32) = v6;
    }
    else
    {
      if ( v6 > *(unsigned int *)(a2 + 36) )
      {
        sub_C8D5F0(a2 + 24, (const void *)(a2 + 40), v6, 8u, a5, a6);
        v7 = *(unsigned int *)(a2 + 32);
      }
      v8 = *(_QWORD *)(a2 + 24);
      v9 = (_QWORD *)(v8 + 8 * v7);
      for ( i = (_QWORD *)(v8 + 8 * v6); i != v9; ++v9 )
      {
        if ( v9 )
          *v9 = 0;
      }
      *(_DWORD *)(a2 + 32) = v6;
    }
  }
  v11 = *(_QWORD *)a1[1];
  result = a2;
  v13 = *(_QWORD *)(a2 + 184);
  v14 = (_QWORD *)(a2 + 176);
  if ( v13 )
  {
    v15 = a2 + 176;
    result = *(_QWORD *)(a2 + 184);
    do
    {
      while ( 1 )
      {
        v16 = *(_QWORD *)(result + 16);
        v17 = *(_QWORD *)(result + 24);
        if ( (unsigned int)v11 <= *(_DWORD *)(result + 32) )
          break;
        result = *(_QWORD *)(result + 24);
        if ( !v17 )
          goto LABEL_15;
      }
      v15 = result;
      result = *(_QWORD *)(result + 16);
    }
    while ( v16 );
LABEL_15:
    if ( v14 != (_QWORD *)v15 && (unsigned int)v11 >= *(_DWORD *)(v15 + 32) )
    {
      v18 = a2 + 176;
      v19 = *(_QWORD *)(a2 + 184);
      do
      {
        while ( 1 )
        {
          v20 = *(_QWORD *)(v19 + 16);
          v21 = *(_QWORD *)(v19 + 24);
          if ( (unsigned int)v11 <= *(_DWORD *)(v19 + 32) )
            break;
          v19 = *(_QWORD *)(v19 + 24);
          if ( !v21 )
            goto LABEL_21;
        }
        v18 = v19;
        v19 = *(_QWORD *)(v19 + 16);
      }
      while ( v20 );
LABEL_21:
      if ( v14 != (_QWORD *)v18 && *(_DWORD *)(v18 + 32) > (unsigned int)v11 )
        v18 = a2 + 176;
      v22 = *(_QWORD *)(v18 + 64);
      v73 = 0;
      v74 = (int *)(v18 + 48);
      if ( v22 != v18 + 48 )
      {
        do
        {
          v73 += **(_QWORD **)(v22 + 64);
          v22 = sub_220EEE0(v22);
        }
        while ( v74 != (int *)v22 );
      }
      v23 = *(int **)(v18 + 56);
      if ( !v23 )
      {
        v66 = 0;
        goto LABEL_50;
      }
      v24 = *(_QWORD *)a1[2];
      v25 = (int *)(v18 + 48);
      do
      {
        while ( 1 )
        {
          v26 = *((_QWORD *)v23 + 2);
          v27 = *((_QWORD *)v23 + 3);
          if ( *((_QWORD *)v23 + 4) >= v24 )
            break;
          v23 = (int *)*((_QWORD *)v23 + 3);
          if ( !v27 )
            goto LABEL_31;
        }
        v25 = v23;
        v23 = (int *)*((_QWORD *)v23 + 2);
      }
      while ( v26 );
LABEL_31:
      v66 = 0;
      if ( v74 == v25 || v24 < *((_QWORD *)v25 + 4) )
        goto LABEL_50;
      v28 = (__int64 *)(v25 + 10);
      v29 = v14;
      v66 = **((_QWORD **)v25 + 8);
      v30 = *a1[3];
      do
      {
        while ( 1 )
        {
          v31 = *(_QWORD *)(v13 + 16);
          v32 = *(_QWORD *)(v13 + 24);
          if ( v30 <= *(_DWORD *)(v13 + 32) )
            break;
          v13 = *(_QWORD *)(v13 + 24);
          if ( !v32 )
            goto LABEL_37;
        }
        v29 = (_QWORD *)v13;
        v13 = *(_QWORD *)(v13 + 16);
      }
      while ( v31 );
LABEL_37:
      if ( v14 == v29 || v30 < *((_DWORD *)v29 + 8) )
      {
        v38 = (__int64)v29;
        v62 = v25;
        v64 = (__int64 *)(v25 + 10);
        v70 = *a1[3];
        v39 = sub_22077B0(0x58u);
        v40 = v70;
        v41 = v39;
        *(_DWORD *)(v39 + 48) = 0;
        v39 += 48;
        *(_DWORD *)(v39 - 16) = v70;
        *(_QWORD *)(v39 + 8) = 0;
        *(_QWORD *)(v41 + 64) = v39;
        *(_QWORD *)(v41 + 72) = v39;
        *(_QWORD *)(v41 + 80) = 0;
        v71 = v41;
        v61 = v40;
        v42 = sub_29A66E0((_QWORD *)(a2 + 168), v38, (unsigned int *)(v41 + 32));
        v44 = v71;
        v45 = v64;
        v46 = v42;
        if ( v43 )
        {
          v47 = v14 == (_QWORD *)v43 || v42 || v61 < *(_DWORD *)(v43 + 32);
          sub_220F040(v47, v71, (_QWORD *)v43, v14);
          v29 = (_QWORD *)v71;
          v28 = v64;
          v25 = v62;
          ++*(_QWORD *)(a2 + 208);
        }
        else
        {
          v65 = v62;
          v72 = v45;
          v63 = v44;
          sub_29A3980(0);
          j_j___libc_free_0(v63);
          v25 = v65;
          v28 = v72;
          v29 = (_QWORD *)v46;
        }
      }
      v75[0] = *((_QWORD *)v25 + 7);
      sub_29A62B0(v29 + 5, v75, v28);
      v33 = *(_QWORD **)(v18 + 56);
      if ( v33 )
      {
        v34 = *(_QWORD *)a1[2];
        v35 = *(int **)(v18 + 56);
        v69 = (int *)(v18 + 48);
        while ( 1 )
        {
          while ( *((_QWORD *)v35 + 4) < v34 )
          {
            v35 = (int *)*((_QWORD *)v35 + 3);
            if ( !v35 )
              goto LABEL_45;
          }
          v36 = (int *)*((_QWORD *)v35 + 2);
          if ( *((_QWORD *)v35 + 4) <= v34 )
            break;
          v69 = v35;
          v35 = (int *)*((_QWORD *)v35 + 2);
          if ( !v36 )
          {
LABEL_45:
            v37 = v74 == v69;
            goto LABEL_46;
          }
        }
        v48 = (int *)*((_QWORD *)v35 + 3);
        if ( v48 )
        {
          do
          {
            while ( 1 )
            {
              v49 = *((_QWORD *)v48 + 2);
              v50 = *((_QWORD *)v48 + 3);
              if ( v34 < *((_QWORD *)v48 + 4) )
                break;
              v48 = (int *)*((_QWORD *)v48 + 3);
              if ( !v50 )
                goto LABEL_64;
            }
            v69 = v48;
            v48 = (int *)*((_QWORD *)v48 + 2);
          }
          while ( v49 );
        }
LABEL_64:
        while ( v36 )
        {
          while ( 1 )
          {
            v51 = *((_QWORD *)v36 + 3);
            if ( v34 <= *((_QWORD *)v36 + 4) )
              break;
            v36 = (int *)*((_QWORD *)v36 + 3);
            if ( !v51 )
              goto LABEL_67;
          }
          v35 = v36;
          v36 = (int *)*((_QWORD *)v36 + 2);
        }
LABEL_67:
        if ( *(int **)(v18 + 64) != v35 || v74 != v69 )
        {
          if ( v69 != v35 )
          {
            do
            {
              v52 = v35;
              v35 = (int *)sub_220EF30((__int64)v35);
              v53 = sub_220F330(v52, v74);
              v54 = *((_QWORD *)v53 + 28);
              v55 = v53;
              while ( v54 )
              {
                v56 = v54;
                sub_29A3730(*(_QWORD **)(v54 + 24));
                v57 = *(_QWORD **)(v54 + 56);
                v54 = *(_QWORD *)(v54 + 16);
                sub_29A3980(v57);
                j_j___libc_free_0(v56);
              }
              v58 = *((_QWORD *)v55 + 8);
              if ( (int *)v58 != v55 + 20 )
                _libc_free(v58);
              v59 = (_QWORD *)*((_QWORD *)v55 + 6);
              if ( v59 )
                *v59 = *((_QWORD *)v55 + 5);
              v60 = *((_QWORD *)v55 + 5);
              if ( v60 )
                *(_QWORD *)(v60 + 8) = *((_QWORD *)v55 + 6);
              j_j___libc_free_0((unsigned __int64)v55);
              --*(_QWORD *)(v18 + 80);
            }
            while ( v35 != v69 );
            v73 -= v66;
            goto LABEL_50;
          }
          goto LABEL_49;
        }
      }
      else
      {
        v69 = (int *)(v18 + 48);
        v37 = 1;
LABEL_46:
        if ( *(int **)(v18 + 64) != v69 || !v37 )
          goto LABEL_49;
      }
      sub_29A3980(v33);
      *(_QWORD *)(v18 + 56) = 0;
      *(_QWORD *)(v18 + 80) = 0;
      *(_QWORD *)(v18 + 64) = v74;
      *(_QWORD *)(v18 + 72) = v74;
LABEL_49:
      v73 -= v66;
LABEL_50:
      *(_QWORD *)(*(_QWORD *)(a2 + 24) + 8LL * *a1[4]) = v66;
      result = *(_QWORD *)(a2 + 24);
      *(_QWORD *)(result + 8LL * *a1[5]) = v73;
    }
  }
  return result;
}
