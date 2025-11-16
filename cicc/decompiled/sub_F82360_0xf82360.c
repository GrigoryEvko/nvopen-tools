// Function: sub_F82360
// Address: 0xf82360
//
__int64 __fastcall sub_F82360(__int64 a1, __int64 a2)
{
  int v3; // r15d
  _QWORD *v4; // rbx
  unsigned int v5; // eax
  __int64 v6; // r14
  _QWORD *v7; // r13
  __int64 v8; // rax
  unsigned int v9; // eax
  __int64 v10; // rdx
  int v11; // eax
  unsigned int v12; // eax
  __int64 v13; // rdx
  __int64 v14; // rbx
  __int64 v15; // r13
  __int64 v16; // rax
  char v17; // al
  __int64 v18; // rdx
  int v19; // r15d
  _QWORD *v20; // rbx
  unsigned int v21; // eax
  __int64 v22; // r13
  _QWORD *v23; // r14
  __int64 v24; // r13
  __int64 v25; // rax
  _QWORD *v26; // r13
  __int64 result; // rax
  _QWORD *v28; // rbx
  __int64 v29; // rax
  int v30; // edx
  __int64 v31; // rbx
  unsigned int v32; // r15d
  unsigned int v33; // eax
  _QWORD *v34; // rdi
  unsigned __int64 v35; // rdx
  unsigned __int64 v36; // rax
  _QWORD *v37; // rax
  __int64 v38; // rdx
  _QWORD *i; // rdx
  __int64 v40; // rax
  __int64 v41; // rbx
  unsigned int v42; // r15d
  unsigned int v43; // eax
  unsigned __int64 *v44; // r15
  unsigned __int64 v45; // rdx
  unsigned __int64 v46; // rax
  __int64 v47; // rax
  unsigned __int64 *v48; // rbx
  __int64 v49; // rax
  unsigned __int64 *v50; // r13
  __int64 v51; // rax
  unsigned __int64 *v52; // rbx
  __int64 v53; // rax
  bool v54; // zf
  _QWORD *v55; // rax
  _QWORD v56[4]; // [rsp+0h] [rbp-80h] BYREF
  unsigned __int64 v57; // [rsp+20h] [rbp-60h] BYREF
  __int64 v58; // [rsp+28h] [rbp-58h] BYREF
  __int64 v59; // [rsp+30h] [rbp-50h]
  __int64 v60; // [rsp+38h] [rbp-48h]
  char v61; // [rsp+40h] [rbp-40h]

  v3 = *(_DWORD *)(a1 + 48);
  ++*(_QWORD *)(a1 + 32);
  if ( !v3 )
  {
    a2 = *(unsigned int *)(a1 + 52);
    if ( !(_DWORD)a2 )
      goto LABEL_18;
  }
  v4 = *(_QWORD **)(a1 + 40);
  v5 = 4 * v3;
  v6 = 40LL * *(unsigned int *)(a1 + 56);
  if ( (unsigned int)(4 * v3) < 0x40 )
    v5 = 64;
  v7 = &v4[(unsigned __int64)v6 / 8];
  if ( *(_DWORD *)(a1 + 56) <= v5 )
  {
    while ( 1 )
    {
      if ( v4 == v7 )
        goto LABEL_17;
      if ( *v4 != -4096 )
        break;
      if ( v4[1] != -4096 )
        goto LABEL_8;
LABEL_12:
      v4 += 5;
    }
    if ( *v4 != -8192 || v4[1] != -8192 )
    {
LABEL_8:
      v8 = v4[4];
      if ( v8 != 0 && v8 != -4096 && v8 != -8192 )
        sub_BD60C0(v4 + 2);
    }
    *v4 = -4096;
    v4[1] = -4096;
    goto LABEL_12;
  }
  do
  {
    if ( *v4 == -4096 )
    {
      if ( v4[1] == -4096 )
        goto LABEL_72;
    }
    else if ( *v4 == -8192 && v4[1] == -8192 )
    {
      goto LABEL_72;
    }
    v29 = v4[4];
    if ( v29 != -4096 && v29 != 0 && v29 != -8192 )
      sub_BD60C0(v4 + 2);
LABEL_72:
    v4 += 5;
  }
  while ( v4 != v7 );
  v30 = *(_DWORD *)(a1 + 56);
  if ( !v3 )
  {
    if ( v30 )
    {
      a2 = v6;
      sub_C7D6A0(*(_QWORD *)(a1 + 40), v6, 8);
      *(_QWORD *)(a1 + 40) = 0;
      *(_QWORD *)(a1 + 48) = 0;
      *(_DWORD *)(a1 + 56) = 0;
      goto LABEL_18;
    }
LABEL_17:
    *(_QWORD *)(a1 + 48) = 0;
    goto LABEL_18;
  }
  v31 = 64;
  v32 = v3 - 1;
  if ( v32 )
  {
    _BitScanReverse(&v33, v32);
    v31 = (unsigned int)(1 << (33 - (v33 ^ 0x1F)));
    if ( (int)v31 < 64 )
      v31 = 64;
  }
  v34 = *(_QWORD **)(a1 + 40);
  if ( (_DWORD)v31 == v30 )
  {
    *(_QWORD *)(a1 + 48) = 0;
    v55 = &v34[5 * v31];
    do
    {
      if ( v34 )
      {
        *v34 = -4096;
        v34[1] = -4096;
      }
      v34 += 5;
    }
    while ( v55 != v34 );
  }
  else
  {
    sub_C7D6A0((__int64)v34, v6, 8);
    a2 = 8;
    v35 = ((((((((4 * (int)v31 / 3u + 1) | ((unsigned __int64)(4 * (int)v31 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v31 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v31 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v31 / 3u + 1) | ((unsigned __int64)(4 * (int)v31 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v31 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v31 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v31 / 3u + 1) | ((unsigned __int64)(4 * (int)v31 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v31 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v31 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v31 / 3u + 1) | ((unsigned __int64)(4 * (int)v31 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v31 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v31 / 3u + 1) >> 1)) >> 16;
    v36 = (v35
         | (((((((4 * (int)v31 / 3u + 1) | ((unsigned __int64)(4 * (int)v31 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v31 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v31 / 3u + 1) >> 1)) >> 4)
           | (((4 * (int)v31 / 3u + 1) | ((unsigned __int64)(4 * (int)v31 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v31 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v31 / 3u + 1) >> 1)) >> 8)
         | (((((4 * (int)v31 / 3u + 1) | ((unsigned __int64)(4 * (int)v31 / 3u + 1) >> 1)) >> 2)
           | (4 * (int)v31 / 3u + 1)
           | ((unsigned __int64)(4 * (int)v31 / 3u + 1) >> 1)) >> 4)
         | (((4 * (int)v31 / 3u + 1) | ((unsigned __int64)(4 * (int)v31 / 3u + 1) >> 1)) >> 2)
         | (4 * (int)v31 / 3u + 1)
         | ((unsigned __int64)(4 * (int)v31 / 3u + 1) >> 1))
        + 1;
    *(_DWORD *)(a1 + 56) = v36;
    v37 = (_QWORD *)sub_C7D670(40 * v36, 8);
    v38 = *(unsigned int *)(a1 + 56);
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 40) = v37;
    for ( i = &v37[5 * v38]; i != v37; v37 += 5 )
    {
      if ( v37 )
      {
        *v37 = -4096;
        v37[1] = -4096;
      }
    }
  }
LABEL_18:
  sub_F817B0(a1 + 64);
  sub_F817B0(a1 + 96);
  ++*(_QWORD *)(a1 + 128);
  if ( *(_BYTE *)(a1 + 156) )
  {
LABEL_23:
    *(_QWORD *)(a1 + 148) = 0;
  }
  else
  {
    v9 = 4 * (*(_DWORD *)(a1 + 148) - *(_DWORD *)(a1 + 152));
    v10 = *(unsigned int *)(a1 + 144);
    if ( v9 < 0x20 )
      v9 = 32;
    if ( (unsigned int)v10 <= v9 )
    {
      memset(*(void **)(a1 + 136), -1, 8 * v10);
      goto LABEL_23;
    }
    sub_C8C990(a1 + 128, a2);
  }
  v11 = *(_DWORD *)(a1 + 304);
  ++*(_QWORD *)(a1 + 288);
  if ( !v11 && !*(_DWORD *)(a1 + 308) )
    goto LABEL_44;
  v12 = 4 * v11;
  v13 = *(unsigned int *)(a1 + 312);
  if ( v12 < 0x40 )
    v12 = 64;
  if ( (unsigned int)v13 > v12 )
  {
    sub_F81E90(a1 + 288);
    goto LABEL_44;
  }
  v14 = *(_QWORD *)(a1 + 296);
  v58 = 2;
  v59 = 0;
  v15 = v14 + 48 * v13;
  v61 = 0;
  v57 = (unsigned __int64)&unk_49E51C0;
  v60 = -4096;
  if ( v14 == v15 )
  {
    *(_QWORD *)(a1 + 304) = 0;
    goto LABEL_44;
  }
  while ( 2 )
  {
    if ( *(_BYTE *)(v14 + 32) )
    {
      *(_QWORD *)(v14 + 24) = 0;
      v16 = v60;
      if ( v60 )
        goto LABEL_32;
    }
    else
    {
      v18 = *(_QWORD *)(v14 + 24);
      v16 = v60;
      if ( v18 != v60 )
      {
        if ( v18 != 0 && v18 != -4096 && v18 != -8192 )
        {
          sub_BD60C0((_QWORD *)(v14 + 8));
          v16 = v60;
        }
LABEL_32:
        *(_QWORD *)(v14 + 24) = v16;
        if ( v16 != -4096 && v16 != 0 && v16 != -8192 )
          sub_BD6050((unsigned __int64 *)(v14 + 8), v58 & 0xFFFFFFFFFFFFFFF8LL);
      }
    }
    v17 = v61;
    v14 += 48;
    *(_BYTE *)(v14 - 16) = v61;
    if ( v15 != v14 )
      continue;
    break;
  }
  *(_QWORD *)(a1 + 304) = 0;
  if ( !v17 )
  {
    v57 = (unsigned __int64)&unk_49DB368;
    if ( v60 != -4096 && v60 != -8192 )
    {
      if ( v60 )
        sub_BD60C0(&v58);
    }
  }
LABEL_44:
  v19 = *(_DWORD *)(a1 + 496);
  ++*(_QWORD *)(a1 + 480);
  if ( v19 || *(_DWORD *)(a1 + 500) )
  {
    v20 = *(_QWORD **)(a1 + 488);
    v21 = 4 * v19;
    v22 = 24LL * *(unsigned int *)(a1 + 504);
    if ( (unsigned int)(4 * v19) < 0x40 )
      v21 = 64;
    v23 = &v20[(unsigned __int64)v22 / 8];
    if ( *(_DWORD *)(a1 + 504) > v21 )
    {
      v56[0] = 0;
      v56[1] = 0;
      v56[2] = -4096;
      v57 = 0;
      v58 = 0;
      v59 = -8192;
      do
      {
        v40 = v20[2];
        if ( v40 != -4096 && v40 != 0 && v40 != -8192 )
          sub_BD60C0(v20);
        v20 += 3;
      }
      while ( v20 != v23 );
      sub_D68D70(&v57);
      sub_D68D70(v56);
      if ( v19 )
      {
        v41 = 64;
        v42 = v19 - 1;
        if ( v42 )
        {
          _BitScanReverse(&v43, v42);
          v41 = (unsigned int)(1 << (33 - (v43 ^ 0x1F)));
          if ( (int)v41 < 64 )
            v41 = 64;
        }
        v44 = *(unsigned __int64 **)(a1 + 488);
        if ( *(_DWORD *)(a1 + 504) == (_DWORD)v41 )
        {
          v57 = 0;
          v52 = &v44[3 * v41];
          v58 = 0;
          *(_QWORD *)(a1 + 496) = 0;
          v59 = -4096;
          if ( v52 == v44 )
            goto LABEL_62;
          do
          {
            if ( v44 )
            {
              *v44 = 0;
              v44[1] = 0;
              v53 = v59;
              v54 = v59 == -4096;
              v44[2] = v59;
              if ( v53 != 0 && !v54 && v53 != -8192 )
                sub_BD6050(v44, v57 & 0xFFFFFFFFFFFFFFF8LL);
            }
            v44 += 3;
          }
          while ( v52 != v44 );
          v51 = v59;
          if ( v59 == 0 || v59 == -4096 )
            goto LABEL_62;
        }
        else
        {
          sub_C7D6A0(*(_QWORD *)(a1 + 488), v22, 8);
          v45 = ((((((((4 * (int)v41 / 3u + 1) | ((unsigned __int64)(4 * (int)v41 / 3u + 1) >> 1)) >> 2)
                   | (4 * (int)v41 / 3u + 1)
                   | ((unsigned __int64)(4 * (int)v41 / 3u + 1) >> 1)) >> 4)
                 | (((4 * (int)v41 / 3u + 1) | ((unsigned __int64)(4 * (int)v41 / 3u + 1) >> 1)) >> 2)
                 | (4 * (int)v41 / 3u + 1)
                 | ((unsigned __int64)(4 * (int)v41 / 3u + 1) >> 1)) >> 8)
               | (((((4 * (int)v41 / 3u + 1) | ((unsigned __int64)(4 * (int)v41 / 3u + 1) >> 1)) >> 2)
                 | (4 * (int)v41 / 3u + 1)
                 | ((unsigned __int64)(4 * (int)v41 / 3u + 1) >> 1)) >> 4)
               | (((4 * (int)v41 / 3u + 1) | ((unsigned __int64)(4 * (int)v41 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v41 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v41 / 3u + 1) >> 1)) >> 16;
          v46 = (v45
               | (((((((4 * (int)v41 / 3u + 1) | ((unsigned __int64)(4 * (int)v41 / 3u + 1) >> 1)) >> 2)
                   | (4 * (int)v41 / 3u + 1)
                   | ((unsigned __int64)(4 * (int)v41 / 3u + 1) >> 1)) >> 4)
                 | (((4 * (int)v41 / 3u + 1) | ((unsigned __int64)(4 * (int)v41 / 3u + 1) >> 1)) >> 2)
                 | (4 * (int)v41 / 3u + 1)
                 | ((unsigned __int64)(4 * (int)v41 / 3u + 1) >> 1)) >> 8)
               | (((((4 * (int)v41 / 3u + 1) | ((unsigned __int64)(4 * (int)v41 / 3u + 1) >> 1)) >> 2)
                 | (4 * (int)v41 / 3u + 1)
                 | ((unsigned __int64)(4 * (int)v41 / 3u + 1) >> 1)) >> 4)
               | (((4 * (int)v41 / 3u + 1) | ((unsigned __int64)(4 * (int)v41 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v41 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v41 / 3u + 1) >> 1))
              + 1;
          *(_DWORD *)(a1 + 504) = v46;
          v47 = sub_C7D670(24 * v46, 8);
          v57 = 0;
          v48 = (unsigned __int64 *)v47;
          *(_QWORD *)(a1 + 488) = v47;
          v49 = *(unsigned int *)(a1 + 504);
          *(_QWORD *)(a1 + 496) = 0;
          v58 = 0;
          v50 = &v48[3 * v49];
          v59 = -4096;
          if ( v48 == v50 )
            goto LABEL_62;
          v51 = -4096;
          do
          {
            if ( v48 )
            {
              *v48 = 0;
              v48[1] = 0;
              v48[2] = v51;
              if ( v51 != -4096 && v51 != 0 && v51 != -8192 )
              {
                sub_BD6050(v48, v57 & 0xFFFFFFFFFFFFFFF8LL);
                v51 = v59;
              }
            }
            v48 += 3;
          }
          while ( v50 != v48 );
          if ( v51 == -4096 || v51 == 0 )
            goto LABEL_62;
        }
        if ( v51 != -8192 )
          sub_BD60C0(&v57);
      }
      else
      {
        if ( !*(_DWORD *)(a1 + 504) )
          goto LABEL_92;
        sub_C7D6A0(*(_QWORD *)(a1 + 488), v22, 8);
        *(_QWORD *)(a1 + 488) = 0;
        *(_QWORD *)(a1 + 496) = 0;
        *(_DWORD *)(a1 + 504) = 0;
      }
    }
    else
    {
      v57 = 0;
      v24 = -4096;
      v58 = 0;
      v59 = -4096;
      if ( v20 != v23 )
      {
        do
        {
          v25 = v20[2];
          if ( v25 != v24 )
          {
            if ( v25 != -4096 && v25 != 0 && v25 != -8192 )
              sub_BD60C0(v20);
            v20[2] = v24;
            if ( v24 != 0 && v24 != -4096 && v24 != -8192 )
              sub_BD73F0((__int64)v20);
            v24 = v59;
          }
          v20 += 3;
        }
        while ( v20 != v23 );
        *(_QWORD *)(a1 + 496) = 0;
        if ( v24 != -4096 && v24 != 0 && v24 != -8192 )
          sub_BD60C0(&v57);
        goto LABEL_62;
      }
LABEL_92:
      *(_QWORD *)(a1 + 496) = 0;
    }
  }
LABEL_62:
  v26 = *(_QWORD **)(a1 + 320);
  result = 3LL * *(unsigned int *)(a1 + 328);
  v28 = &v26[3 * *(unsigned int *)(a1 + 328)];
  while ( v26 != v28 )
  {
    while ( 1 )
    {
      result = *(v28 - 1);
      v28 -= 3;
      if ( result == 0 || result == -4096 || result == -8192 )
        break;
      result = sub_BD60C0(v28);
      if ( v26 == v28 )
        goto LABEL_67;
    }
  }
LABEL_67:
  *(_DWORD *)(a1 + 328) = 0;
  return result;
}
