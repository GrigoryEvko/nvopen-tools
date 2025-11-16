// Function: sub_19E2E90
// Address: 0x19e2e90
//
void __fastcall sub_19E2E90(__int64 *a1, char *a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rsi
  __int64 *v5; // r15
  __int64 v6; // r13
  __int64 *v7; // r13
  __int64 v8; // r12
  bool v9; // al
  __int64 v10; // r11
  __int64 v11; // rbx
  __int64 v12; // r11
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // r11
  int v17; // r9d
  __int64 v18; // r14
  char *v19; // rbx
  char *v20; // rcx
  int v21; // r13d
  unsigned int v22; // edx
  __int64 *v23; // rax
  __int64 v24; // r8
  unsigned int v25; // edi
  __int64 v26; // rax
  __int64 *v27; // rdx
  __int64 v28; // r10
  _QWORD *v29; // rsi
  __int64 v30; // rax
  _QWORD *v31; // rdx
  unsigned int v32; // r8d
  __int64 v33; // rdi
  unsigned int v34; // edx
  __int64 *v35; // rax
  __int64 v36; // r10
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // rax
  int v40; // eax
  unsigned int v41; // edi
  int v42; // edx
  unsigned int v43; // r8d
  int v44; // edx
  int v45; // r12d
  int v46; // eax
  int v47; // r8d
  __int64 v48; // rax
  __int64 v49; // r13
  __int64 v50; // rbx
  __int64 v51; // r9
  __int64 *v52; // rbx
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // rdx
  __int64 v56; // rax
  int v57; // r12d
  int v59; // [rsp+Ch] [rbp-74h]
  char *v60; // [rsp+10h] [rbp-70h]
  __int64 v61; // [rsp+20h] [rbp-60h]
  char *v62; // [rsp+28h] [rbp-58h]
  __int64 *v64; // [rsp+38h] [rbp-48h]
  _QWORD v65[7]; // [rsp+48h] [rbp-38h] BYREF

  v62 = a2;
  v4 = a2 - (char *)a1;
  v61 = a3;
  if ( v4 <= 256 )
    return;
  v5 = a1;
  v6 = v4;
  if ( !a3 )
  {
    v64 = (__int64 *)v62;
LABEL_45:
    v49 = v6 >> 4;
    v50 = (v49 - 2) >> 1;
    sub_19E2230((__int64)v5, v50, v49, v5[2 * v50], v5[2 * v50 + 1], a4);
    do
    {
      --v50;
      sub_19E2230((__int64)v5, v50, v49, v5[2 * v50], v5[2 * v50 + 1], v51);
    }
    while ( v50 );
    v52 = v64;
    do
    {
      v52 -= 2;
      v53 = *v52;
      v54 = v52[1];
      *v52 = *v5;
      v52[1] = v5[1];
      sub_19E2230((__int64)v5, 0, ((char *)v52 - (char *)v5) >> 4, v53, v54, v51);
    }
    while ( (char *)v52 - (char *)v5 > 16 );
    return;
  }
  v60 = (char *)(a1 + 2);
  while ( 2 )
  {
    v65[0] = a4;
    --v61;
    v7 = &a1[2 * (v6 >> 5)];
    v8 = (__int64)(v62 - 16);
    v9 = sub_19E2DB0((__int64)v65, a1[3], (__int64)v7);
    v11 = *a1;
    if ( !v9 )
    {
      if ( !sub_19E2DB0((__int64)v65, v10, v8) )
      {
        if ( !sub_19E2DB0((__int64)v65, v7[1], v8) )
          goto LABEL_6;
LABEL_51:
        *a1 = *((_QWORD *)v62 - 2);
        v55 = *((_QWORD *)v62 - 1);
        *((_QWORD *)v62 - 2) = v11;
        v56 = a1[1];
        a1[1] = v55;
        *((_QWORD *)v62 - 1) = v56;
        v15 = a1[3];
        v16 = a1[1];
        goto LABEL_7;
      }
LABEL_43:
      v48 = a1[2];
      v15 = a1[1];
      a1[2] = v11;
      a1[1] = v16;
      *a1 = v48;
      a1[3] = v15;
      goto LABEL_7;
    }
    if ( !sub_19E2DB0((__int64)v65, v7[1], v8) )
    {
      if ( sub_19E2DB0((__int64)v65, v12, v8) )
        goto LABEL_51;
      goto LABEL_43;
    }
LABEL_6:
    *a1 = *v7;
    v13 = v7[1];
    *v7 = v11;
    v14 = a1[1];
    a1[1] = v13;
    v7[1] = v14;
    v15 = a1[3];
    v16 = a1[1];
LABEL_7:
    v17 = *(_DWORD *)(a4 + 2384);
    v18 = *(_QWORD *)(a4 + 2368);
    v19 = v60;
    v20 = v62;
    v21 = v17 - 1;
    while ( 1 )
    {
      v64 = (__int64 *)v19;
      v28 = v21 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v29 = (_QWORD *)(v18 + 16 * v28);
      if ( !v17 )
        break;
      v22 = v21 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v23 = (__int64 *)(v18 + 16LL * v22);
      v24 = *v23;
      if ( *v23 == v15 )
      {
LABEL_9:
        v25 = *((_DWORD *)v23 + 2);
      }
      else
      {
        v46 = 1;
        while ( v24 != -8 )
        {
          v57 = v46 + 1;
          v22 = v21 & (v46 + v22);
          v23 = (__int64 *)(v18 + 16LL * v22);
          v24 = *v23;
          if ( *v23 == v15 )
            goto LABEL_9;
          v46 = v57;
        }
        v25 = 0;
      }
      v26 = *v29;
      v27 = (__int64 *)(v18 + 16 * v28);
      if ( *v29 != v16 )
      {
        v43 = v21 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v44 = 1;
        while ( v26 != -8 )
        {
          v45 = v44 + 1;
          v43 = v21 & (v44 + v43);
          v27 = (__int64 *)(v18 + 16LL * v43);
          v26 = *v27;
          if ( *v27 == v16 )
            goto LABEL_11;
          v44 = v45;
        }
        break;
      }
LABEL_11:
      if ( *((_DWORD *)v27 + 2) <= v25 )
        break;
LABEL_12:
      v15 = *((_QWORD *)v19 + 3);
      v19 += 16;
    }
    v20 -= 16;
    while ( v17 )
    {
      v30 = *v29;
      v31 = v29;
      if ( *v29 == v16 )
      {
LABEL_16:
        v32 = *((_DWORD *)v31 + 2);
        v33 = *((_QWORD *)v20 + 1);
      }
      else
      {
        v41 = v21 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v42 = 1;
        while ( v30 != -8 )
        {
          v47 = v42 + 1;
          v41 = v21 & (v42 + v41);
          v31 = (_QWORD *)(v18 + 16LL * v41);
          v30 = *v31;
          if ( *v31 == v16 )
            goto LABEL_16;
          v42 = v47;
        }
        v33 = *((_QWORD *)v20 + 1);
        v32 = 0;
      }
      v34 = v21 & (((unsigned int)v33 >> 9) ^ ((unsigned int)v33 >> 4));
      v35 = (__int64 *)(v18 + 16LL * v34);
      v36 = *v35;
      if ( *v35 != v33 )
      {
        v40 = 1;
        while ( v36 != -8 )
        {
          v34 = v21 & (v40 + v34);
          v59 = v40 + 1;
          v35 = (__int64 *)(v18 + 16LL * v34);
          v36 = *v35;
          if ( *v35 == v33 )
            goto LABEL_18;
          v40 = v59;
        }
        break;
      }
LABEL_18:
      if ( *((_DWORD *)v35 + 2) <= v32 )
        break;
      v20 -= 16;
    }
    if ( v19 < v20 )
    {
      v37 = *(_QWORD *)v19;
      *(_QWORD *)v19 = *(_QWORD *)v20;
      v38 = *((_QWORD *)v20 + 1);
      *(_QWORD *)v20 = v37;
      v39 = *((_QWORD *)v19 + 1);
      *((_QWORD *)v19 + 1) = v38;
      *((_QWORD *)v20 + 1) = v39;
      v17 = *(_DWORD *)(a4 + 2384);
      v18 = *(_QWORD *)(a4 + 2368);
      v16 = a1[1];
      v21 = v17 - 1;
      goto LABEL_12;
    }
    sub_19E2E90(v19, v62, v61, a4);
    v6 = v19 - (char *)a1;
    if ( v19 - (char *)a1 > 256 )
    {
      if ( v61 )
      {
        v62 = v19;
        continue;
      }
      v5 = a1;
      goto LABEL_45;
    }
    break;
  }
}
