// Function: sub_24AE100
// Address: 0x24ae100
//
char __fastcall sub_24AE100(__int64 a1)
{
  _QWORD *v2; // r15
  _QWORD *v3; // r12
  char v4; // si
  __int64 v5; // rax
  _QWORD *v6; // rdi
  __int64 v7; // r8
  unsigned int v8; // ecx
  _QWORD *v9; // rdx
  _QWORD *v10; // r10
  __int64 v11; // r13
  int v12; // ecx
  __int64 *v13; // rax
  __int64 v14; // rsi
  __int64 *j; // rcx
  __int64 v16; // rdx
  bool v17; // zf
  __int64 *v18; // rax
  unsigned __int64 v19; // rsi
  __int64 *k; // rcx
  __int64 v21; // rdx
  __int64 v22; // rdx
  __int64 *v23; // rax
  __int64 v24; // rdi
  __int64 *m; // rsi
  __int64 v26; // rdx
  _QWORD *v27; // rdi
  __int64 v28; // rax
  __int64 v29; // r14
  __int64 v30; // r15
  __int64 v31; // rax
  __int64 v32; // r12
  __int64 v33; // r13
  __int64 v34; // rsi
  int v35; // eax
  __int64 *v36; // rax
  unsigned __int64 v37; // rsi
  __int64 *i; // rcx
  __int64 v39; // rdx
  __int64 v40; // rdx
  int v41; // edx
  int v42; // r11d
  int v43; // eax
  __int64 v44; // rsi
  unsigned int v45; // r8d
  __int64 v46; // r11
  __int64 v47; // rcx
  __int64 v48; // rdx
  unsigned int v49; // esi
  __int64 *v50; // rax
  __int64 v51; // r9
  unsigned __int64 v52; // r13
  unsigned __int64 v53; // r12
  unsigned int v54; // r15d
  __int64 v55; // rdx
  unsigned int v56; // esi
  __int64 *v57; // rax
  __int64 v58; // rdi
  __int64 v59; // rax
  unsigned __int64 v60; // rax
  char result; // al
  int v62; // eax
  int v63; // eax
  int v64; // r10d
  int v65; // [rsp+4h] [rbp-3Ch]
  __int64 v66; // [rsp+8h] [rbp-38h]

  do
  {
    v2 = (_QWORD *)(*(_QWORD *)a1 + 72LL);
    v3 = (_QWORD *)(*v2 & 0xFFFFFFFFFFFFFFF8LL);
    if ( v3 == v2 )
      break;
    v4 = 0;
    do
    {
      while ( 1 )
      {
        v5 = *(unsigned int *)(a1 + 296);
        v6 = v3 - 3;
        v7 = *(_QWORD *)(a1 + 280);
        if ( !v3 )
          v6 = 0;
        if ( (_DWORD)v5 )
        {
          v8 = (v5 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
          v9 = (_QWORD *)(v7 + 16LL * v8);
          v10 = (_QWORD *)*v9;
          if ( v6 != (_QWORD *)*v9 )
          {
            v41 = 1;
            while ( v10 != (_QWORD *)-4096LL )
            {
              v42 = v41 + 1;
              v8 = (v5 - 1) & (v41 + v8);
              v9 = (_QWORD *)(v7 + 16LL * v8);
              v10 = (_QWORD *)*v9;
              if ( v6 == (_QWORD *)*v9 )
                goto LABEL_10;
              v41 = v42;
            }
            goto LABEL_5;
          }
LABEL_10:
          if ( v9 != (_QWORD *)(v7 + 16 * v5) )
          {
            v11 = v9[1];
            if ( v11 )
              break;
          }
        }
LABEL_5:
        v3 = (_QWORD *)(*v3 & 0xFFFFFFFFFFFFFFF8LL);
        if ( v2 == v3 )
          goto LABEL_36;
      }
      v12 = *(_DWORD *)(v11 + 36);
      if ( *(_BYTE *)(v11 + 24) )
      {
        if ( v12 == 1 )
        {
LABEL_47:
          v36 = *(__int64 **)(v11 + 72);
          v37 = 0;
          for ( i = &v36[*(unsigned int *)(v11 + 80)]; i != v36; ++v36 )
          {
            v39 = *v36;
            if ( !*(_BYTE *)(*v36 + 25) && *(_BYTE *)(v39 + 40) )
              v37 += *(_QWORD *)(v39 + 32);
          }
          v40 = *(_QWORD *)(v11 + 16) - v37;
          if ( *(_QWORD *)(v11 + 16) <= v37 )
            v40 = 0;
          sub_24A5F10(a1, v11 + 72, v40);
          v4 = 1;
        }
        if ( *(_DWORD *)(v11 + 32) == 1 )
          goto LABEL_20;
        goto LABEL_5;
      }
      if ( !v12 )
      {
        v13 = *(__int64 **)(v11 + 72);
        v14 = 0;
        for ( j = &v13[*(unsigned int *)(v11 + 80)]; j != v13; ++v13 )
        {
          v16 = *v13;
          if ( !*(_BYTE *)(*v13 + 25) && *(_BYTE *)(v16 + 40) )
            v14 += *(_QWORD *)(v16 + 32);
        }
        v17 = *(_DWORD *)(v11 + 32) == 1;
        *(_QWORD *)(v11 + 16) = v14;
        v4 = 1;
        *(_BYTE *)(v11 + 24) = 1;
        if ( v17 )
        {
LABEL_20:
          v18 = *(__int64 **)(v11 + 40);
          v19 = 0;
          for ( k = &v18[*(unsigned int *)(v11 + 48)]; k != v18; ++v18 )
          {
            v21 = *v18;
            if ( !*(_BYTE *)(*v18 + 25) && *(_BYTE *)(v21 + 40) )
              v19 += *(_QWORD *)(v21 + 32);
          }
          v22 = *(_QWORD *)(v11 + 16) - v19;
          if ( *(_QWORD *)(v11 + 16) <= v19 )
            v22 = 0;
          sub_24A5F10(a1, v11 + 40, v22);
          goto LABEL_35;
        }
        goto LABEL_5;
      }
      if ( *(_DWORD *)(v11 + 32) )
        goto LABEL_5;
      v23 = *(__int64 **)(v11 + 40);
      v24 = 0;
      for ( m = &v23[*(unsigned int *)(v11 + 48)]; m != v23; ++v23 )
      {
        v26 = *v23;
        if ( !*(_BYTE *)(*v23 + 25) && *(_BYTE *)(v26 + 40) )
          v24 += *(_QWORD *)(v26 + 32);
      }
      *(_QWORD *)(v11 + 16) = v24;
      *(_BYTE *)(v11 + 24) = 1;
      if ( v12 == 1 )
        goto LABEL_47;
LABEL_35:
      v4 = 1;
      v3 = (_QWORD *)(*v3 & 0xFFFFFFFFFFFFFFF8LL);
    }
    while ( v2 != v3 );
LABEL_36:
    ;
  }
  while ( v4 );
  *(_DWORD *)(a1 + 108) = 2;
  v27 = (_QWORD *)(a1 + 96);
  *(_QWORD *)(a1 + 112) = a1 + 440;
  v28 = *(_QWORD *)(a1 + 96);
  *(_QWORD *)(a1 + 144) = a1;
  v29 = *(_QWORD *)(v28 + 80);
  v30 = v28 + 72;
LABEL_38:
  while ( v30 != v29 )
  {
    v31 = v29;
    v29 = *(_QWORD *)(v29 + 8);
    v32 = *(_QWORD *)(v31 + 32);
    v33 = v31 + 24;
    if ( v31 + 24 != v32 )
    {
      while ( 1 )
      {
        v34 = v32;
        v32 = *(_QWORD *)(v32 + 8);
        v35 = *(unsigned __int8 *)(v34 - 24);
        if ( v35 == 86 )
        {
          if ( (_BYTE)qword_4FEBC88
            && !(_BYTE)qword_4FEB5C8
            && !*(_BYTE *)(a1 + 152)
            && (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(v34 - 120) + 8LL) + 8LL) - 17 > 1 )
          {
            v43 = *(_DWORD *)(a1 + 108);
            v44 = v34 - 24;
            if ( v43 == 1 )
            {
              sub_24AAB30((__int64)v27, v44);
            }
            else if ( v43 == 2 )
            {
              sub_24ADFC0(v27, v44);
            }
            else
            {
              if ( v43 )
LABEL_102:
                BUG();
              ++*(_DWORD *)(a1 + 104);
            }
          }
        }
        else if ( (unsigned int)(v35 - 29) <= 0x39 )
        {
          if ( (unsigned int)(v35 - 30) > 0x37 )
            goto LABEL_102;
        }
        else if ( (unsigned int)(v35 - 87) > 9 )
        {
          goto LABEL_102;
        }
        if ( v33 == v32 )
          goto LABEL_38;
      }
    }
  }
  v45 = *(_DWORD *)(a1 + 296);
  v46 = *(_QWORD *)(a1 + 280);
  v47 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  v66 = *(_QWORD *)a1;
  v48 = v47 - 24;
  if ( !v47 )
    v48 = 0;
  if ( v45 )
  {
    v49 = (v45 - 1) & (((unsigned int)v48 >> 9) ^ ((unsigned int)v48 >> 4));
    v50 = (__int64 *)(v46 + 16LL * v49);
    v51 = *v50;
    if ( v48 == *v50 )
      goto LABEL_72;
    v63 = 1;
    while ( v51 != -4096 )
    {
      v64 = v63 + 1;
      v49 = (v45 - 1) & (v63 + v49);
      v50 = (__int64 *)(v46 + 16LL * v49);
      v51 = *v50;
      if ( v48 == *v50 )
        goto LABEL_72;
      v63 = v64;
    }
  }
  v50 = (__int64 *)(v46 + 16LL * v45);
LABEL_72:
  v52 = *(_QWORD *)(v50[1] + 16);
  if ( v66 + 72 == v47 )
  {
    v53 = *(_QWORD *)(v50[1] + 16);
  }
  else
  {
    v53 = *(_QWORD *)(v50[1] + 16);
    v54 = v45 - 1;
    do
    {
      v55 = v47 - 24;
      if ( !v47 )
        v55 = 0;
      if ( v45 )
      {
        v56 = v54 & (((unsigned int)v55 >> 9) ^ ((unsigned int)v55 >> 4));
        v57 = (__int64 *)(v46 + 16LL * v56);
        v58 = *v57;
        if ( v55 == *v57 )
        {
LABEL_78:
          if ( (__int64 *)(v46 + 16LL * v45) != v57 )
          {
            v59 = v57[1];
            if ( v59 )
            {
              v60 = *(_QWORD *)(v59 + 16);
              if ( v53 < v60 )
                v53 = v60;
            }
          }
        }
        else
        {
          v62 = 1;
          while ( v58 != -4096 )
          {
            v56 = v54 & (v62 + v56);
            v65 = v62 + 1;
            v57 = (__int64 *)(v46 + 16LL * v56);
            v58 = *v57;
            if ( v55 == *v57 )
              goto LABEL_78;
            v62 = v65;
          }
        }
      }
      v47 = *(_QWORD *)(v47 + 8);
    }
    while ( v66 + 72 != v47 );
  }
  if ( v53 && !v52 )
    v52 = 1;
  sub_B2F4C0(v66, v52, 0, 0);
  result = sub_D84440(*(_QWORD *)(a1 + 24), v52);
  if ( result )
  {
    *(_DWORD *)(a1 + 504) = 2;
  }
  else
  {
    result = sub_D84450(*(_QWORD *)(a1 + 24), v53);
    if ( result )
      *(_DWORD *)(a1 + 504) = 1;
  }
  return result;
}
