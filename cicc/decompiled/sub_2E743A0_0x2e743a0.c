// Function: sub_2E743A0
// Address: 0x2e743a0
//
__int64 __fastcall sub_2E743A0(__int64 *a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 *v5; // rbx
  __int64 v6; // rsi
  bool v7; // zf
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // r11
  __int64 v11; // rdi
  __int64 *v12; // r15
  __int64 v13; // r12
  __int64 *v14; // rsi
  __int64 v15; // r9
  __int64 v16; // r10
  int v17; // ebx
  int v18; // r13d
  __int64 v19; // r14
  __int64 *v20; // rdi
  unsigned int v21; // edx
  __int64 *v22; // rax
  __int64 v23; // rcx
  unsigned int v24; // edx
  __int64 v25; // rax
  __int64 *v26; // rcx
  unsigned int v27; // eax
  __int64 v28; // rcx
  __int64 v29; // rax
  __int64 *v30; // rdx
  unsigned int v31; // r9d
  unsigned int v32; // edx
  __int64 *v33; // rax
  __int64 v34; // r8
  int v35; // eax
  unsigned int v36; // r8d
  int v37; // edx
  unsigned int v38; // r8d
  int v39; // ecx
  int v40; // eax
  int v41; // r9d
  __int64 v42; // rbx
  __int64 v43; // r12
  __int64 *v44; // rbx
  __int64 v45; // rcx
  char *v46; // r14
  __int64 *v47; // rbx
  __int64 v48; // rax
  int v49; // r8d
  __int64 v50; // rax
  int v51; // [rsp+Ch] [rbp-84h]
  int v52; // [rsp+Ch] [rbp-84h]
  __int64 *v53; // [rsp+10h] [rbp-80h]
  __int64 v54; // [rsp+20h] [rbp-70h]
  char *v55; // [rsp+28h] [rbp-68h]
  __int64 *v58; // [rsp+40h] [rbp-50h]
  __int64 *v59; // [rsp+48h] [rbp-48h]
  __int64 *v60[7]; // [rsp+58h] [rbp-38h] BYREF

  result = a2 - (char *)a1;
  v55 = a2;
  v54 = a3;
  if ( a2 - (char *)a1 <= 128 )
    return result;
  if ( !a3 )
  {
    v58 = (__int64 *)a2;
    goto LABEL_44;
  }
  v53 = a1 + 2;
  while ( 2 )
  {
    --v54;
    v5 = &a1[result >> 4];
    v6 = a1[1];
    v60[0] = (__int64 *)a4;
    v7 = !sub_2E6E900(v60, v6, *v5);
    v8 = *((_QWORD *)v55 - 1);
    if ( v7 )
    {
      if ( sub_2E6E900(v60, a1[1], v8) )
      {
        v10 = a1[1];
        v11 = *a1;
        *a1 = v10;
        a1[1] = v11;
        goto LABEL_7;
      }
      v46 = v55;
      if ( !sub_2E6E900(v60, *v5, *((_QWORD *)v55 - 1)) )
      {
        v50 = *a1;
        *a1 = *v5;
        *v5 = v50;
        v10 = *a1;
        v11 = a1[1];
        goto LABEL_7;
      }
      v47 = a1;
LABEL_50:
      v48 = *v47;
      *v47 = *((_QWORD *)v46 - 1);
      *((_QWORD *)v46 - 1) = v48;
      v10 = *v47;
      v11 = v47[1];
      goto LABEL_7;
    }
    if ( !sub_2E6E900(v60, *v5, v8) )
    {
      v46 = v55;
      v47 = a1;
      if ( !sub_2E6E900(v60, a1[1], *((_QWORD *)v55 - 1)) )
      {
        v10 = a1[1];
        v11 = *a1;
        *a1 = v10;
        a1[1] = v11;
        goto LABEL_7;
      }
      goto LABEL_50;
    }
    v9 = *a1;
    *a1 = *v5;
    *v5 = v9;
    v10 = *a1;
    v11 = a1[1];
LABEL_7:
    v12 = v53;
    v13 = v11;
    v14 = (__int64 *)v55;
    v15 = *(unsigned int *)(a4 + 24);
    v16 = *(_QWORD *)(a4 + 8);
    v17 = v15;
    v18 = v15 - 1;
    while ( 1 )
    {
      v58 = v12 - 1;
      v59 = (__int64 *)(v16 + 16 * v15);
      v19 = v18 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v20 = (__int64 *)(v16 + 16 * v19);
      if ( !v17 )
        break;
      v21 = v18 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v22 = (__int64 *)(v16 + 16LL * v21);
      v23 = *v22;
      if ( *v22 != v13 )
      {
        v40 = 1;
        while ( v23 != -4096 )
        {
          v49 = v40 + 1;
          v21 = v18 & (v40 + v21);
          v22 = (__int64 *)(v16 + 16LL * v21);
          v23 = *v22;
          if ( *v22 == v13 )
            goto LABEL_11;
          v40 = v49;
        }
        v22 = (__int64 *)(v16 + 16 * v15);
      }
LABEL_11:
      v24 = *((_DWORD *)v22 + 2);
      v25 = *v20;
      v26 = (__int64 *)(v16 + 16 * v19);
      if ( *v20 == v10 )
      {
LABEL_12:
        v27 = *((_DWORD *)v26 + 2);
      }
      else
      {
        v38 = v18 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
        v39 = 1;
        while ( v25 != -4096 )
        {
          v38 = v18 & (v39 + v38);
          v52 = v39 + 1;
          v26 = (__int64 *)(v16 + 16LL * v38);
          v25 = *v26;
          if ( *v26 == v10 )
            goto LABEL_12;
          v39 = v52;
        }
        v27 = *((_DWORD *)v59 + 2);
      }
      if ( v24 >= v27 )
      {
        v28 = *--v14;
        goto LABEL_15;
      }
LABEL_8:
      v13 = *v12++;
    }
    do
    {
      v28 = *--v14;
      if ( !v17 )
        break;
LABEL_15:
      v29 = *v20;
      v30 = (__int64 *)(v16 + 16 * v19);
      if ( *v20 == v10 )
      {
LABEL_16:
        v31 = *((_DWORD *)v30 + 2);
      }
      else
      {
        v36 = v18 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
        v37 = 1;
        while ( v29 != -4096 )
        {
          v41 = v37 + 1;
          v36 = v18 & (v37 + v36);
          v30 = (__int64 *)(v16 + 16LL * v36);
          v29 = *v30;
          if ( *v30 == v10 )
            goto LABEL_16;
          v37 = v41;
        }
        v31 = *((_DWORD *)v59 + 2);
      }
      v32 = v18 & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
      v33 = (__int64 *)(v16 + 16LL * v32);
      v34 = *v33;
      if ( v28 != *v33 )
      {
        v35 = 1;
        while ( v34 != -4096 )
        {
          v32 = v18 & (v35 + v32);
          v51 = v35 + 1;
          v33 = (__int64 *)(v16 + 16LL * v32);
          v34 = *v33;
          if ( v28 == *v33 )
            goto LABEL_18;
          v35 = v51;
        }
        v33 = v59;
      }
LABEL_18:
      ;
    }
    while ( v31 < *((_DWORD *)v33 + 2) );
    if ( v14 > v58 )
    {
      *(v12 - 1) = v28;
      *v14 = v13;
      v10 = *a1;
      v15 = *(unsigned int *)(a4 + 24);
      v16 = *(_QWORD *)(a4 + 8);
      v17 = v15;
      v18 = v15 - 1;
      goto LABEL_8;
    }
    sub_2E743A0(v58, v55, v54, a4);
    result = (char *)v58 - (char *)a1;
    if ( (char *)v58 - (char *)a1 > 128 )
    {
      if ( v54 )
      {
        v55 = (char *)(v12 - 1);
        continue;
      }
LABEL_44:
      v42 = result >> 3;
      v43 = ((result >> 3) - 2) >> 1;
      sub_2E74160((__int64)a1, v43, result >> 3, a1[v43], (__int64 *)a4);
      do
      {
        --v43;
        sub_2E74160((__int64)a1, v43, v42, a1[v43], (__int64 *)a4);
      }
      while ( v43 );
      v44 = v58;
      do
      {
        v45 = *--v44;
        *v44 = *a1;
        result = sub_2E74160((__int64)a1, 0, v44 - a1, v45, (__int64 *)a4);
      }
      while ( (char *)v44 - (char *)a1 > 8 );
    }
    return result;
  }
}
