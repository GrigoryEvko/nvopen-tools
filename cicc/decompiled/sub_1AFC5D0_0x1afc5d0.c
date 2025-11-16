// Function: sub_1AFC5D0
// Address: 0x1afc5d0
//
void __fastcall sub_1AFC5D0(__int64 *a1, char *a2, __int64 a3, __int64 *a4)
{
  __int64 v4; // rbx
  __int64 v5; // rsi
  __int64 *v6; // rbx
  __int64 v7; // rax
  __int64 v8; // r11
  __int64 *v9; // r14
  char *v10; // r15
  __int64 v11; // r12
  unsigned int v12; // eax
  unsigned int v13; // r13d
  unsigned int v14; // ecx
  __int64 *v15; // rdx
  __int64 v16; // rsi
  __int64 *v17; // rdi
  unsigned int v18; // ecx
  __int64 *v19; // r8
  __int64 *v20; // rdx
  char *v21; // rcx
  __int64 v22; // rsi
  __int64 *v23; // rax
  unsigned int v24; // r10d
  unsigned int v25; // edx
  __int64 *v26; // rax
  __int64 v27; // r9
  __int64 v28; // rbx
  unsigned int v29; // edx
  __int64 v30; // r10
  int i; // eax
  int v32; // r9d
  int v33; // eax
  unsigned int v34; // r10d
  __int64 v35; // r9
  int v36; // edx
  int v37; // esi
  int v38; // edx
  int v39; // edi
  __int64 v40; // rax
  __int64 v41; // rbx
  __int64 v42; // r8
  __int64 j; // r12
  __int64 v44; // r8
  __int64 *v45; // rbx
  __int64 v46; // rcx
  __int64 v47; // r12
  char *v48; // r14
  __int64 *v49; // rdi
  __int64 v50; // rdi
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 *v53; // [rsp+8h] [rbp-88h]
  __int64 v54; // [rsp+18h] [rbp-78h]
  char *v55; // [rsp+20h] [rbp-70h]
  int v58; // [rsp+38h] [rbp-58h]
  unsigned int v59; // [rsp+3Ch] [rbp-54h]
  char *v60; // [rsp+40h] [rbp-50h]
  __int64 v61; // [rsp+48h] [rbp-48h]
  __int64 *v62[7]; // [rsp+58h] [rbp-38h] BYREF

  v4 = a2 - (char *)a1;
  v54 = a3;
  v55 = a2;
  if ( a2 - (char *)a1 <= 128 )
    return;
  if ( !a3 )
  {
    v60 = a2;
    goto LABEL_45;
  }
  v53 = a1 + 2;
  while ( 2 )
  {
    --v54;
    v5 = a1[1];
    v6 = &a1[(__int64)(((v55 - (char *)a1) >> 3) + ((unsigned __int64)(v55 - (char *)a1) >> 63)) >> 1];
    v62[0] = a4;
    if ( !sub_1AFC4C0(v62, v5, *v6) )
    {
      if ( sub_1AFC4C0(v62, a1[1], *((_QWORD *)v55 - 1)) )
      {
        v8 = a1[1];
        v61 = *a1;
        v40 = *a1;
        *a1 = v8;
        a1[1] = v40;
        goto LABEL_7;
      }
      v48 = v55;
      v49 = a1;
      if ( !sub_1AFC4C0(v62, *v6, *((_QWORD *)v55 - 1)) )
      {
        v52 = *a1;
        *a1 = *v6;
        *v6 = v52;
        v8 = *a1;
        v61 = a1[1];
        goto LABEL_7;
      }
LABEL_54:
      v51 = *v49;
      *v49 = *((_QWORD *)v48 - 1);
      *((_QWORD *)v48 - 1) = v51;
      v8 = *v49;
      v61 = v49[1];
      goto LABEL_7;
    }
    if ( !sub_1AFC4C0(v62, *v6, *((_QWORD *)v55 - 1)) )
    {
      v48 = v55;
      v49 = a1;
      if ( !sub_1AFC4C0(v62, a1[1], *((_QWORD *)v55 - 1)) )
      {
        v8 = a1[1];
        v61 = *a1;
        v50 = *a1;
        *a1 = v8;
        a1[1] = v50;
        goto LABEL_7;
      }
      goto LABEL_54;
    }
    v7 = *a1;
    *a1 = *v6;
    *v6 = v7;
    v8 = *a1;
    v61 = a1[1];
LABEL_7:
    v9 = v53;
    v10 = v55;
    v11 = *(_QWORD *)(*a4 + 32);
    v12 = *(_DWORD *)(*a4 + 48);
    while ( 1 )
    {
      v60 = (char *)(v9 - 1);
      if ( !v12 )
        goto LABEL_59;
      v13 = v12 - 1;
      v14 = (v12 - 1) & (((unsigned int)v61 >> 9) ^ ((unsigned int)v61 >> 4));
      v15 = (__int64 *)(v11 + 16LL * v14);
      v16 = *v15;
      if ( *v15 != v61 )
      {
        v38 = 1;
        while ( v16 != -8 )
        {
          v39 = v38 + 1;
          v14 = v13 & (v38 + v14);
          v15 = (__int64 *)(v11 + 16LL * v14);
          v16 = *v15;
          if ( *v15 == v61 )
            goto LABEL_10;
          v38 = v39;
        }
LABEL_59:
        BUG();
      }
LABEL_10:
      v17 = (__int64 *)(v11 + 16LL * v12);
      if ( v15 == v17 )
        BUG();
      v18 = *(_DWORD *)(v15[1] + 16);
      v59 = v13 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v19 = (__int64 *)(v11 + 16LL * v59);
      v20 = v19;
      if ( v8 != *v19 )
      {
        v34 = v13 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v35 = *v19;
        v36 = 1;
        while ( v35 != -8 )
        {
          v37 = v36 + 1;
          v34 = v13 & (v36 + v34);
          v20 = (__int64 *)(v11 + 16LL * v34);
          v35 = *v20;
          if ( *v20 == v8 )
            goto LABEL_12;
          v36 = v37;
        }
LABEL_60:
        BUG();
      }
LABEL_12:
      if ( v20 == v17 )
        goto LABEL_60;
      if ( v18 <= *(_DWORD *)(v20[1] + 16) )
        break;
LABEL_22:
      v28 = *v9++;
      v61 = v28;
    }
    v21 = v10 - 8;
    do
    {
      v22 = *(_QWORD *)v21;
      v10 = v21;
      v23 = (__int64 *)(v11 + 16LL * v59);
      if ( v8 != *v19 )
      {
        v29 = v13 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v30 = *v19;
        for ( i = 1; ; i = v32 )
        {
          if ( v30 == -8 )
            BUG();
          v32 = i + 1;
          v29 = v13 & (i + v29);
          v23 = (__int64 *)(v11 + 16LL * v29);
          v30 = *v23;
          if ( v8 == *v23 )
            break;
        }
      }
      if ( v17 == v23 )
        BUG();
      v24 = *(_DWORD *)(v23[1] + 16);
      v25 = v13 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v26 = (__int64 *)(v11 + 16LL * v25);
      v27 = *v26;
      if ( v22 != *v26 )
      {
        v33 = 1;
        while ( v27 != -8 )
        {
          v25 = v13 & (v33 + v25);
          v58 = v33 + 1;
          v26 = (__int64 *)(v11 + 16LL * v25);
          v27 = *v26;
          if ( v22 == *v26 )
            goto LABEL_18;
          v33 = v58;
        }
LABEL_58:
        BUG();
      }
LABEL_18:
      if ( v26 == v17 )
        goto LABEL_58;
      v21 -= 8;
    }
    while ( v24 > *(_DWORD *)(v26[1] + 16) );
    if ( v60 < v10 )
    {
      *(v9 - 1) = v22;
      *(_QWORD *)v10 = v61;
      v8 = *a1;
      v11 = *(_QWORD *)(*a4 + 32);
      v12 = *(_DWORD *)(*a4 + 48);
      goto LABEL_22;
    }
    sub_1AFC5D0(v60, v55, v54, a4);
    v4 = v60 - (char *)a1;
    if ( v60 - (char *)a1 > 128 )
    {
      if ( v54 )
      {
        v55 = (char *)(v9 - 1);
        continue;
      }
LABEL_45:
      v41 = v4 >> 3;
      v42 = (__int64)a4;
      for ( j = (v41 - 2) >> 1; ; --j )
      {
        sub_1AFBEB0((__int64)a1, j, v41, a1[j], v42);
        if ( !j )
          break;
      }
      v44 = (__int64)a4;
      v45 = (__int64 *)(v60 - 8);
      do
      {
        v46 = *v45;
        v47 = (char *)v45-- - (char *)a1;
        v45[1] = *a1;
        sub_1AFBEB0((__int64)a1, 0, v47 >> 3, v46, v44);
      }
      while ( v47 > 8 );
    }
    break;
  }
}
