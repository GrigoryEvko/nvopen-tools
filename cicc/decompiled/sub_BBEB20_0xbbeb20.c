// Function: sub_BBEB20
// Address: 0xbbeb20
//
__int64 __fastcall sub_BBEB20(__int64 *a1, __int64 a2, void **a3, __int64 *a4)
{
  __int64 v4; // rcx
  void **v5; // rax
  void **v6; // rdx
  __int64 v7; // rbx
  void **v9; // rax
  __int64 v10; // rcx
  void **v11; // rsi
  void **v12; // rdi
  void **v13; // rdx
  __int64 v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rcx
  int v17; // r9d
  unsigned int i; // eax
  __int64 v19; // rsi
  unsigned int v20; // eax
  void **v21; // rax
  __int64 v22; // rsi
  __int64 v23; // r8
  char v24; // al
  __int64 *v25; // rbx
  __int64 *v26; // r14
  __int64 v27; // r9
  __int64 v28; // r12
  char v29; // si
  __int64 v30; // rdi
  int v31; // edx
  __int64 v32; // rax
  __int64 v33; // r10
  __int64 v34; // rdx
  __int64 v35; // rax
  unsigned __int64 v36; // rdx
  __int64 v37; // r12
  __int64 *v38; // rbx
  __int64 v39; // r14
  __int64 v40; // rdx
  _QWORD *v41; // rax
  _QWORD *v42; // rax
  _QWORD *v43; // rax
  __int64 v44; // rax
  __int64 v45; // rdx
  __int64 v46; // rcx
  int v47; // r10d
  unsigned int j; // eax
  _QWORD *v49; // rsi
  unsigned int v50; // eax
  __int64 v51; // rax
  int v52; // eax
  __int64 v53; // r8
  __int64 v54; // rbx
  __int64 v55; // rax
  int v56; // r8d
  __int64 v57; // rdx
  __int64 v58; // rcx
  void **v59; // rcx
  void **v60; // rdx
  void **v61; // r8
  __int64 v62; // rcx
  void **v63; // rsi
  __int64 v64; // rdx
  __int64 v65; // rcx
  __int64 v66; // rcx
  unsigned __int64 v67; // [rsp+0h] [rbp-150h]
  __int64 v68; // [rsp+8h] [rbp-148h]
  bool v69; // [rsp+27h] [rbp-129h]
  __int64 v70; // [rsp+28h] [rbp-128h]
  __int64 v73; // [rsp+40h] [rbp-110h]
  __int64 *v74; // [rsp+48h] [rbp-108h]
  __int64 *v75; // [rsp+50h] [rbp-100h]
  __int64 v76; // [rsp+50h] [rbp-100h]
  __int64 v77; // [rsp+60h] [rbp-F0h]
  __int64 v79; // [rsp+70h] [rbp-E0h] BYREF
  char v80[8]; // [rsp+78h] [rbp-D8h] BYREF
  char v81[16]; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v82; // [rsp+90h] [rbp-C0h]
  _QWORD v83[20]; // [rsp+B0h] [rbp-A0h] BYREF

  v77 = (__int64)a3;
  if ( *((_DWORD *)a3 + 18) == *((_DWORD *)a3 + 17) )
  {
    if ( *((_BYTE *)a3 + 28) )
    {
      v21 = (void **)a3[1];
      a3 = &v21[*((unsigned int *)a3 + 5)];
      if ( v21 != a3 )
      {
        while ( *v21 != &unk_4F82400 )
        {
          if ( a3 == ++v21 )
            goto LABEL_2;
        }
        return 0;
      }
    }
    else if ( sub_C8CA60(a3, &unk_4F82400, a3, a3) )
    {
      return 0;
    }
  }
LABEL_2:
  v68 = v77 + 48;
  if ( *(_BYTE *)(v77 + 76) )
  {
    v4 = v77;
    v5 = *(void ***)(v77 + 56);
    v6 = &v5[*(unsigned int *)(v77 + 68)];
    if ( v5 != v6 )
    {
      while ( *v5 != &unk_4F82418 )
      {
        if ( v6 == ++v5 )
          goto LABEL_9;
      }
      goto LABEL_7;
    }
  }
  else if ( sub_C8CA60(v68, &unk_4F82418, a3, v77 + 48) )
  {
    goto LABEL_7;
  }
LABEL_9:
  if ( *(_BYTE *)(v77 + 28) )
  {
    v9 = *(void ***)(v77 + 8);
    v10 = *(unsigned int *)(v77 + 20);
    v11 = &v9[v10];
    v12 = v9;
    if ( v9 != v11 )
    {
      v13 = *(void ***)(v77 + 8);
      while ( *v13 != &unk_4F82400 )
      {
        if ( v11 == ++v13 )
          goto LABEL_131;
      }
      goto LABEL_14;
    }
    goto LABEL_7;
  }
  if ( !sub_C8CA60(v77, &unk_4F82400, v6, v4) )
  {
    if ( *(_BYTE *)(v77 + 28) )
    {
      v9 = *(void ***)(v77 + 8);
      v10 = *(unsigned int *)(v77 + 20);
      v12 = v9;
      v13 = &v9[v10];
      if ( v13 != v9 )
      {
LABEL_131:
        v61 = v9;
        while ( *v61 != &unk_4F82418 )
        {
          if ( ++v61 == v13 )
            goto LABEL_139;
        }
        goto LABEL_14;
      }
      goto LABEL_7;
    }
    if ( !sub_C8CA60(v77, &unk_4F82418, v57, v58) )
    {
      if ( *(_BYTE *)(v77 + 28) )
      {
        v9 = *(void ***)(v77 + 8);
        v10 = *(unsigned int *)(v77 + 20);
        v13 = &v9[v10];
        v12 = v9;
        if ( v9 != v13 )
        {
LABEL_139:
          v63 = v9;
          while ( *v12 != &unk_4F82400 )
          {
            if ( ++v12 == v13 )
              goto LABEL_149;
          }
          goto LABEL_14;
        }
        goto LABEL_7;
      }
      if ( !sub_C8CA60(v77, &unk_4F82400, v57, v62) )
      {
        if ( *(_BYTE *)(v77 + 28) )
        {
          v9 = *(void ***)(v77 + 8);
          v10 = *(unsigned int *)(v77 + 20);
          v63 = v9;
          v13 = &v9[v10];
          if ( v9 != v13 )
          {
LABEL_149:
            while ( *v63 != &unk_4F82428 )
            {
              if ( ++v63 == v13 )
                goto LABEL_7;
            }
LABEL_14:
            if ( *(_DWORD *)(v77 + 68) != *(_DWORD *)(v77 + 72) )
            {
LABEL_15:
              v69 = 0;
              goto LABEL_16;
            }
            goto LABEL_125;
          }
LABEL_7:
          v7 = *a1;
          sub_BBC150(*a1 + 64);
          sub_BBC340(v7 + 32);
          return 1;
        }
        if ( !sub_C8CA60(v77, &unk_4F82428, v57, v66) )
          goto LABEL_7;
      }
    }
  }
  if ( *(_DWORD *)(v77 + 68) != *(_DWORD *)(v77 + 72) )
    goto LABEL_15;
  if ( *(_BYTE *)(v77 + 28) )
  {
    v9 = *(void ***)(v77 + 8);
    v10 = *(unsigned int *)(v77 + 20);
LABEL_125:
    v59 = &v9[v10];
    if ( v9 == v59 )
      goto LABEL_15;
    v60 = v9;
    while ( *v60 != &unk_4F82400 )
    {
      if ( v59 == ++v60 )
        goto LABEL_158;
    }
    goto LABEL_129;
  }
  if ( sub_C8CA60(v77, &unk_4F82400, v57, v77) )
  {
LABEL_129:
    v69 = 1;
    goto LABEL_16;
  }
  if ( *(_BYTE *)(v77 + 28) )
  {
    v9 = *(void ***)(v77 + 8);
    v60 = &v9[*(unsigned int *)(v77 + 20)];
    if ( v9 == v60 )
      goto LABEL_15;
LABEL_158:
    while ( *v9 != &unk_4F82420 )
    {
      if ( ++v9 == v60 )
        goto LABEL_15;
    }
    goto LABEL_129;
  }
  v69 = sub_C8CA60(v77, &unk_4F82420, v64, v65) != 0;
LABEL_16:
  v73 = *(_QWORD *)(a2 + 32);
  if ( v73 == a2 + 24 )
    return 0;
  do
  {
    v14 = v73 - 56;
    memset(v83, 0, 0x68u);
    if ( !v73 )
      v14 = 0;
    v70 = v14;
    v15 = *(unsigned int *)(*a1 + 88);
    v16 = *(_QWORD *)(*a1 + 72);
    if ( !(_DWORD)v15 )
      goto LABEL_33;
    v17 = 1;
    v67 = (unsigned __int64)(((unsigned int)&unk_4F82410 >> 9) ^ ((unsigned int)&unk_4F82410 >> 4)) << 32;
    for ( i = (v15 - 1)
            & (((0xBF58476D1CE4E5B9LL * (v67 | ((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4))) >> 31)
             ^ (484763065 * (v67 | ((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4)))); ; i = (v15 - 1) & v20 )
    {
      v19 = v16 + 24LL * i;
      if ( *(_UNKNOWN **)v19 == &unk_4F82410 && v70 == *(_QWORD *)(v19 + 8) )
        break;
      if ( *(_QWORD *)v19 == -4096 && *(_QWORD *)(v19 + 8) == -4096 )
        goto LABEL_33;
      v20 = v17 + i;
      ++v17;
    }
    if ( v19 == v16 + 24 * v15 )
      goto LABEL_33;
    v23 = *(_QWORD *)(*(_QWORD *)(v19 + 16) + 24LL);
    if ( !v23 )
      goto LABEL_33;
    v24 = *(_BYTE *)(v23 + 24) & 1;
    if ( *(_DWORD *)(v23 + 24) >> 1 )
    {
      if ( v24 )
      {
        v25 = (__int64 *)(v23 + 32);
        v26 = (__int64 *)(v23 + 64);
      }
      else
      {
        v25 = *(__int64 **)(v23 + 32);
        v53 = 2LL * *(unsigned int *)(v23 + 40);
        v26 = &v25[v53];
        if ( v25 == &v25[v53] )
          goto LABEL_49;
      }
      do
      {
        if ( *v25 != -8192 && *v25 != -4096 )
          break;
        v25 += 2;
      }
      while ( v25 != v26 );
    }
    else
    {
      if ( v24 )
      {
        v54 = v23 + 32;
        v55 = 32;
      }
      else
      {
        v54 = *(_QWORD *)(v23 + 32);
        v55 = 16LL * *(unsigned int *)(v23 + 40);
      }
      v25 = (__int64 *)(v55 + v54);
      v26 = v25;
    }
LABEL_49:
    if ( v26 == v25 )
      goto LABEL_33;
    do
    {
      v27 = *v25;
      v28 = *a4;
      v29 = *(_BYTE *)(*a4 + 8) & 1;
      if ( v29 )
      {
        v30 = v28 + 16;
        v31 = 7;
      }
      else
      {
        v35 = *(unsigned int *)(v28 + 24);
        v30 = *(_QWORD *)(v28 + 16);
        if ( !(_DWORD)v35 )
          goto LABEL_102;
        v31 = v35 - 1;
      }
      v16 = v31 & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
      v32 = v30 + 16 * v16;
      v33 = *(_QWORD *)v32;
      if ( v27 == *(_QWORD *)v32 )
        goto LABEL_53;
      v52 = 1;
      while ( v33 != -4096 )
      {
        v56 = v52 + 1;
        v16 = v31 & (unsigned int)(v52 + v16);
        v32 = v30 + 16LL * (unsigned int)v16;
        v33 = *(_QWORD *)v32;
        if ( v27 == *(_QWORD *)v32 )
          goto LABEL_53;
        v52 = v56;
      }
      if ( v29 )
      {
        v51 = 128;
        goto LABEL_103;
      }
      v35 = *(unsigned int *)(v28 + 24);
LABEL_102:
      v51 = 16 * v35;
LABEL_103:
      v32 = v30 + v51;
LABEL_53:
      v34 = 128;
      if ( !v29 )
        v34 = 16LL * *(unsigned int *)(v28 + 24);
      if ( v32 == v30 + v34 )
      {
        v44 = a4[1];
        v45 = *(unsigned int *)(v44 + 24);
        v46 = *(_QWORD *)(v44 + 8);
        if ( (_DWORD)v45 )
        {
          v47 = 1;
          for ( j = (v45 - 1)
                  & (((0xBF58476D1CE4E5B9LL
                     * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)
                      | ((unsigned __int64)(((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4)) << 32))) >> 31)
                   ^ (484763065 * (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4)))); ; j = (v45 - 1) & v50 )
          {
            v49 = (_QWORD *)(v46 + 24LL * j);
            if ( v27 == *v49 && a2 == v49[1] )
              break;
            if ( *v49 == -4096 && v49[1] == -4096 )
              goto LABEL_111;
            v50 = v47 + j;
            ++v47;
          }
        }
        else
        {
LABEL_111:
          v49 = (_QWORD *)(v46 + 24 * v45);
        }
        v76 = *v25;
        v80[0] = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64 *))(**(_QWORD **)(v49[2] + 24LL) + 16LL))(
                   *(_QWORD *)(v49[2] + 24LL),
                   a2,
                   v77,
                   a4);
        v79 = v76;
        sub_BBCF50((__int64)v81, v28, &v79, v80);
        v32 = v82;
      }
      if ( !*(_BYTE *)(v32 + 8) )
      {
        v25 += 2;
        goto LABEL_61;
      }
      if ( !LOBYTE(v83[12]) )
      {
        sub_C8CD80(v83, &v83[4], v77);
        sub_C8CD80(&v83[6], &v83[10], v68);
        LOBYTE(v83[12]) = 1;
      }
      v36 = v25[1] & 0xFFFFFFFFFFFFFFF8LL;
      if ( (v25[1] & 4) != 0 )
      {
        v16 = *(_QWORD *)v36;
        v25 += 2;
        v37 = *(_QWORD *)v36 + 8LL * *(unsigned int *)(v36 + 8);
      }
      else
      {
        v16 = (__int64)(v25 + 1);
        v25 += 2;
        if ( !v36 )
          goto LABEL_61;
        v37 = (__int64)v25;
      }
      if ( v37 != v16 )
      {
        v75 = v26;
        v74 = v25;
        v38 = (__int64 *)v16;
        while ( 1 )
        {
          v39 = *v38;
          if ( BYTE4(v83[3]) )
          {
            v40 = v83[1] + 8LL * HIDWORD(v83[2]);
            v41 = (_QWORD *)v83[1];
            if ( v83[1] != v40 )
            {
              while ( v39 != *v41 )
              {
                if ( (_QWORD *)v40 == ++v41 )
                  goto LABEL_83;
              }
              --HIDWORD(v83[2]);
              *v41 = *(_QWORD *)(v83[1] + 8LL * HIDWORD(v83[2]));
              ++v83[0];
            }
          }
          else
          {
            v43 = (_QWORD *)sub_C8CA60(v83, v39, v36, v16);
            if ( v43 )
            {
              *v43 = -2;
              ++LODWORD(v83[3]);
              ++v83[0];
            }
          }
LABEL_83:
          if ( !BYTE4(v83[9]) )
            goto LABEL_90;
          v42 = (_QWORD *)v83[7];
          v36 = v83[7] + 8LL * HIDWORD(v83[8]);
          if ( v83[7] == v36 )
            break;
          while ( v39 != *v42 )
          {
            if ( (_QWORD *)v36 == ++v42 )
              goto LABEL_91;
          }
LABEL_88:
          if ( (__int64 *)v37 == ++v38 )
          {
            v26 = v75;
            v25 = v74;
            goto LABEL_61;
          }
        }
LABEL_91:
        if ( HIDWORD(v83[8]) < LODWORD(v83[8]) )
        {
          ++HIDWORD(v83[8]);
          *(_QWORD *)v36 = v39;
          ++v83[6];
          goto LABEL_88;
        }
LABEL_90:
        sub_C8CC70(&v83[6], v39);
        goto LABEL_88;
      }
      while ( 1 )
      {
LABEL_61:
        if ( v25 == v26 )
          goto LABEL_62;
        if ( *v25 != -8192 && *v25 != -4096 )
          break;
        v25 += 2;
      }
    }
    while ( v25 != v26 );
LABEL_62:
    if ( LOBYTE(v83[12]) )
    {
      v22 = v70;
      sub_BBE020(*a1, v70, (__int64)v83, v16);
      if ( LOBYTE(v83[12]) )
      {
        if ( !BYTE4(v83[9]) )
LABEL_65:
          _libc_free(v83[7], v22);
LABEL_38:
        if ( !BYTE4(v83[3]) )
          _libc_free(v83[1], v22);
      }
      goto LABEL_34;
    }
LABEL_33:
    if ( !v69 )
    {
      v22 = v70;
      sub_BBE020(*a1, v70, v77, v16);
      if ( LOBYTE(v83[12]) )
      {
        if ( !BYTE4(v83[9]) )
          goto LABEL_65;
        goto LABEL_38;
      }
    }
LABEL_34:
    v73 = *(_QWORD *)(v73 + 8);
  }
  while ( a2 + 24 != v73 );
  return 0;
}
