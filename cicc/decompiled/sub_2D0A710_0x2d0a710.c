// Function: sub_2D0A710
// Address: 0x2d0a710
//
__int64 __fastcall sub_2D0A710(_QWORD *a1, __int64 a2, _BYTE *a3)
{
  __int64 v5; // r12
  _QWORD *v6; // r14
  unsigned int v7; // esi
  __int64 v8; // rcx
  __int64 v9; // r10
  unsigned int v10; // edx
  int *v11; // rax
  __int64 v12; // r8
  int v13; // esi
  __int64 v14; // rax
  _QWORD *v15; // rdi
  __int64 v16; // rcx
  __int64 v17; // rdx
  _QWORD *v18; // r12
  int v19; // esi
  __int64 v20; // r8
  __int64 v21; // rax
  __int64 v22; // rcx
  __int64 v23; // rdx
  int *v24; // rax
  __int64 v26; // r12
  __int64 v27; // rsi
  __int64 v28; // rcx
  int v29; // r9d
  unsigned int v30; // edx
  int *v31; // rax
  __int64 v32; // r8
  __int64 v33; // rax
  __int64 v34; // rdi
  int v35; // esi
  __int64 v36; // rax
  __int64 v37; // rcx
  __int64 v38; // rdx
  int *v39; // rdi
  int v40; // eax
  int v41; // edx
  __int64 v42; // r11
  unsigned int v43; // r14d
  int i; // r10d
  int v45; // r13d
  __int64 *v46; // r10
  int v47; // r11d
  int *v48; // r10
  int v49; // eax
  int v50; // edx
  int *v51; // rax
  __int64 v52; // rdx
  int v53; // [rsp+10h] [rbp-60h]
  __int64 v54[2]; // [rsp+18h] [rbp-58h] BYREF
  int v55; // [rsp+2Ch] [rbp-44h] BYREF
  __int64 v56; // [rsp+30h] [rbp-40h] BYREF
  int *v57[7]; // [rsp+38h] [rbp-38h] BYREF

  v5 = a1[2];
  v6 = (_QWORD *)*a1;
  v54[0] = a2;
  v7 = *(_DWORD *)(v5 + 272);
  if ( !v7 )
  {
    v57[0] = 0;
    ++*(_QWORD *)(v5 + 248);
    goto LABEL_44;
  }
  v8 = v54[0];
  v9 = *(_QWORD *)(v5 + 256);
  v10 = (v7 - 1) & ((LODWORD(v54[0]) >> 9) ^ (LODWORD(v54[0]) >> 4));
  v11 = (int *)(v9 + 16LL * v10);
  v12 = *(_QWORD *)v11;
  if ( v54[0] != *(_QWORD *)v11 )
  {
    v53 = 1;
    v39 = 0;
    while ( v12 != -4096 )
    {
      if ( v12 == -8192 && !v39 )
        v39 = v11;
      v10 = (v7 - 1) & (v53 + v10);
      v11 = (int *)(v9 + 16LL * v10);
      v12 = *(_QWORD *)v11;
      if ( v54[0] == *(_QWORD *)v11 )
        goto LABEL_3;
      ++v53;
    }
    if ( !v39 )
      v39 = v11;
    v57[0] = v39;
    v40 = *(_DWORD *)(v5 + 264);
    ++*(_QWORD *)(v5 + 248);
    v41 = v40 + 1;
    if ( 4 * (v40 + 1) < 3 * v7 )
    {
      if ( v7 - *(_DWORD *)(v5 + 268) - v41 > v7 >> 3 )
      {
LABEL_40:
        *(_DWORD *)(v5 + 264) = v41;
        if ( *(_QWORD *)v39 != -4096 )
          --*(_DWORD *)(v5 + 268);
        *(_QWORD *)v39 = v8;
        v13 = 0;
        v39[2] = 0;
        goto LABEL_4;
      }
LABEL_45:
      sub_B23080(v5 + 248, v7);
      sub_B1C700(v5 + 248, v54, v57);
      v8 = v54[0];
      v39 = v57[0];
      v41 = *(_DWORD *)(v5 + 264) + 1;
      goto LABEL_40;
    }
LABEL_44:
    v7 *= 2;
    goto LABEL_45;
  }
LABEL_3:
  v13 = v11[2];
LABEL_4:
  v14 = v6[2];
  if ( v14 )
  {
    v15 = v6 + 1;
    do
    {
      while ( 1 )
      {
        v16 = *(_QWORD *)(v14 + 16);
        v17 = *(_QWORD *)(v14 + 24);
        if ( *(_DWORD *)(v14 + 32) >= v13 )
          break;
        v14 = *(_QWORD *)(v14 + 24);
        if ( !v17 )
          goto LABEL_9;
      }
      v15 = (_QWORD *)v14;
      v14 = *(_QWORD *)(v14 + 16);
    }
    while ( v16 );
LABEL_9:
    if ( v6 + 1 != v15 && *((_DWORD *)v15 + 8) <= v13 )
    {
      *a3 = 1;
      v18 = (_QWORD *)*a1;
      v19 = *(_DWORD *)sub_2D0A0A0(a1[2] + 248LL, v54);
      v20 = (__int64)(v18 + 1);
      LODWORD(v56) = v19;
      v21 = v18[2];
      if ( v21 )
      {
        do
        {
          while ( 1 )
          {
            v22 = *(_QWORD *)(v21 + 16);
            v23 = *(_QWORD *)(v21 + 24);
            if ( v19 <= *(_DWORD *)(v21 + 32) )
              break;
            v21 = *(_QWORD *)(v21 + 24);
            if ( !v23 )
              goto LABEL_16;
          }
          v20 = v21;
          v21 = *(_QWORD *)(v21 + 16);
        }
        while ( v22 );
LABEL_16:
        if ( v18 + 1 != (_QWORD *)v20 && v19 >= *(_DWORD *)(v20 + 32) )
          return *(_QWORD *)(v20 + 40);
      }
      v24 = (int *)&v56;
LABEL_19:
      v57[0] = v24;
      v20 = sub_2D07F60(v18, v20, v57);
      return *(_QWORD *)(v20 + 40);
    }
  }
  v26 = a1[1];
  v27 = *(unsigned int *)(v26 + 24);
  v28 = *(_QWORD *)(v26 + 8);
  if ( (_DWORD)v27 )
  {
    v29 = v27 - 1;
    v30 = (v27 - 1) & ((LODWORD(v54[0]) >> 9) ^ (LODWORD(v54[0]) >> 4));
    v31 = (int *)(v28 + 16LL * v30);
    v32 = *(_QWORD *)v31;
    if ( v54[0] == *(_QWORD *)v31 )
    {
      if ( v31 != (int *)(v28 + 16 * v27) )
      {
LABEL_24:
        v33 = *((_QWORD *)v31 + 1);
LABEL_25:
        v34 = a1[2];
        v18 = (_QWORD *)*a1;
        v56 = v33;
        v35 = *(_DWORD *)sub_2D0A0A0(v34 + 248, &v56);
        v36 = v18[2];
        v20 = (__int64)(v18 + 1);
        v55 = v35;
        if ( v36 )
        {
          do
          {
            while ( 1 )
            {
              v37 = *(_QWORD *)(v36 + 16);
              v38 = *(_QWORD *)(v36 + 24);
              if ( v35 <= *(_DWORD *)(v36 + 32) )
                break;
              v36 = *(_QWORD *)(v36 + 24);
              if ( !v38 )
                goto LABEL_30;
            }
            v20 = v36;
            v36 = *(_QWORD *)(v36 + 16);
          }
          while ( v37 );
LABEL_30:
          if ( (_QWORD *)v20 != v18 + 1 && v35 >= *(_DWORD *)(v20 + 32) )
            return *(_QWORD *)(v20 + 40);
        }
        v24 = &v55;
        goto LABEL_19;
      }
      return 0;
    }
    v42 = *(_QWORD *)v31;
    v43 = (v27 - 1) & ((LODWORD(v54[0]) >> 9) ^ (LODWORD(v54[0]) >> 4));
    for ( i = 1; ; i = v45 )
    {
      if ( v42 == -4096 )
        return 0;
      v45 = i + 1;
      v43 = v29 & (i + v43);
      v46 = (__int64 *)(v28 + 16LL * v43);
      v42 = *v46;
      if ( v54[0] == *v46 )
        break;
    }
    if ( v46 == (__int64 *)(v28 + 16LL * (unsigned int)v27) )
      return 0;
    v47 = 1;
    v48 = 0;
    while ( v32 != -4096 )
    {
      if ( !v48 && v32 == -8192 )
        v48 = v31;
      v30 = v29 & (v47 + v30);
      v31 = (int *)(v28 + 16LL * v30);
      v32 = *(_QWORD *)v31;
      if ( v54[0] == *(_QWORD *)v31 )
        goto LABEL_24;
      ++v47;
    }
    if ( !v48 )
      v48 = v31;
    v57[0] = v48;
    v49 = *(_DWORD *)(v26 + 16);
    ++*(_QWORD *)v26;
    v50 = v49 + 1;
    if ( 4 * (v49 + 1) >= (unsigned int)(3 * v27) )
    {
      LODWORD(v27) = 2 * v27;
    }
    else if ( (int)v27 - *(_DWORD *)(v26 + 20) - v50 > (unsigned int)v27 >> 3 )
    {
LABEL_56:
      *(_DWORD *)(v26 + 16) = v50;
      v51 = v57[0];
      if ( *(_QWORD *)v57[0] != -4096 )
        --*(_DWORD *)(v26 + 20);
      v52 = v54[0];
      *((_QWORD *)v51 + 1) = 0;
      *(_QWORD *)v51 = v52;
      v33 = 0;
      goto LABEL_25;
    }
    sub_22E02D0(v26, v27);
    sub_27EFA30(v26, v54, v57);
    v50 = *(_DWORD *)(v26 + 16) + 1;
    goto LABEL_56;
  }
  return 0;
}
