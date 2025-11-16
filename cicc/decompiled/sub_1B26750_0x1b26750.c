// Function: sub_1B26750
// Address: 0x1b26750
//
__int64 __fastcall sub_1B26750(__int64 a1, _QWORD *a2)
{
  _QWORD *v2; // r12
  char *v3; // rax
  char *v4; // rsi
  unsigned int v5; // edx
  int v6; // ecx
  _QWORD *v7; // r14
  _QWORD *v8; // rbx
  __int64 v9; // r15
  const char *v10; // rax
  unsigned __int64 v11; // rdx
  _QWORD *v12; // r14
  _QWORD *v13; // rbx
  __int64 v14; // r15
  const char *v15; // rax
  unsigned __int64 v16; // rdx
  __int64 *v17; // rbx
  __int64 *v18; // r15
  unsigned __int64 v19; // r13
  size_t v20; // rdx
  const char *v21; // rsi
  __int64 **v22; // r12
  __int64 v23; // rdx
  __int64 *v24; // rdx
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // r12
  const char *v28; // rax
  unsigned __int64 v29; // rdx
  const char *v30; // r15
  unsigned __int64 v31; // rbx
  char **v32; // rsi
  char *v33; // rax
  __int64 v34; // rbx
  __int64 v35; // rdx
  __int64 v36; // rdx
  __int64 v37; // rdi
  __int64 i; // rbx
  __int64 j; // r14
  __int64 v41; // [rsp+8h] [rbp-188h]
  unsigned __int64 v42; // [rsp+10h] [rbp-180h]
  _QWORD *v43; // [rsp+18h] [rbp-178h]
  _QWORD *v44; // [rsp+20h] [rbp-170h]
  _QWORD *v45; // [rsp+20h] [rbp-170h]
  __int64 v46; // [rsp+28h] [rbp-168h]
  _QWORD v47[2]; // [rsp+30h] [rbp-160h] BYREF
  __int16 v48; // [rsp+40h] [rbp-150h]
  __int64 v49; // [rsp+50h] [rbp-140h] BYREF
  __int64 v50; // [rsp+58h] [rbp-138h]
  __int64 v51; // [rsp+60h] [rbp-130h]
  __int64 v52; // [rsp+68h] [rbp-128h]
  __int64 v53; // [rsp+70h] [rbp-120h]
  __int64 v54; // [rsp+78h] [rbp-118h]
  __int64 v55; // [rsp+80h] [rbp-110h]
  __int64 v56; // [rsp+88h] [rbp-108h]
  __int64 v57; // [rsp+90h] [rbp-100h]
  __int64 v58; // [rsp+98h] [rbp-F8h]
  __int64 v59; // [rsp+A0h] [rbp-F0h]
  __int64 v60; // [rsp+A8h] [rbp-E8h]
  __int64 *v61; // [rsp+B0h] [rbp-E0h]
  __int64 *v62; // [rsp+B8h] [rbp-D8h]
  __int64 v63; // [rsp+C0h] [rbp-D0h]
  char v64; // [rsp+C8h] [rbp-C8h]
  char *v65; // [rsp+D0h] [rbp-C0h] BYREF
  __int64 v66; // [rsp+D8h] [rbp-B8h]
  _WORD v67[88]; // [rsp+E0h] [rbp-B0h] BYREF

  v2 = a2;
  v3 = (char *)a2[22];
  v4 = &v3[a2[23]];
  if ( v4 == v3 )
  {
    v42 = 0;
  }
  else
  {
    v5 = 0;
    do
    {
      v6 = *v3++;
      v5 += v6;
    }
    while ( v4 != v3 );
    v42 = v5;
  }
  v7 = (_QWORD *)v2[6];
  v8 = v2 + 5;
  if ( v2 + 5 != v7 )
  {
    while ( 1 )
    {
      v9 = (__int64)(v7 - 6);
      if ( !v7 )
        v9 = 0;
      v10 = sub_1649960(v9);
      if ( v11 > 4 )
      {
        if ( *(_DWORD *)v10 != 1836477548 || v10[4] != 46 )
          goto LABEL_8;
        v7 = (_QWORD *)v7[1];
        if ( v8 == v7 )
          break;
      }
      else
      {
        if ( !v11 )
        {
LABEL_9:
          v65 = "alias";
          v67[0] = 259;
          sub_164B780(v9, (__int64 *)&v65);
          goto LABEL_10;
        }
LABEL_8:
        if ( *v10 != 1 )
          goto LABEL_9;
LABEL_10:
        v7 = (_QWORD *)v7[1];
        if ( v8 == v7 )
          break;
      }
    }
  }
  v12 = (_QWORD *)v2[2];
  v13 = v2 + 1;
  if ( v2 + 1 != v12 )
  {
    while ( 1 )
    {
      v14 = (__int64)(v12 - 7);
      if ( !v12 )
        v14 = 0;
      v15 = sub_1649960(v14);
      if ( v16 > 4 )
      {
        if ( *(_DWORD *)v15 != 1836477548 || v15[4] != 46 )
          goto LABEL_21;
        v12 = (_QWORD *)v12[1];
        if ( v13 == v12 )
          break;
      }
      else
      {
        if ( !v16 )
        {
LABEL_22:
          v65 = "global";
          v67[0] = 259;
          sub_164B780(v14, (__int64 *)&v65);
          goto LABEL_23;
        }
LABEL_21:
        if ( *v15 != 1 )
          goto LABEL_22;
LABEL_23:
        v12 = (_QWORD *)v12[1];
        if ( v13 == v12 )
          break;
      }
    }
  }
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  sub_1648080((__int64)&v49, v2, 1);
  v17 = v62;
  v18 = v61;
  if ( v62 != v61 )
  {
    v44 = v2;
    v19 = v42;
    do
    {
      v22 = (__int64 **)*v18;
      if ( (*(_BYTE *)(*v18 + 9) & 4) == 0 )
      {
        sub_1643640(*v18);
        if ( v23 )
        {
          v66 = 0x8000000000LL;
          v65 = (char *)v67;
          v19 = 1103515245 * v19 + 12345;
          if ( *off_49852C0[((v19 >> 16) & 0x7FFF)
                          - ((((((v19 >> 16) & 0x7FFF) * (unsigned __int128)0xF0F0F0F0F0F0F0F1LL) >> 64)
                            & 0xFFFFFFFFFFFFFFF0LL)
                           + ((v19 >> 16) & 0x7FFF) / 0x11)] )
          {
            v47[1] = off_49852C0[((v19 >> 16) & 0x7FFF)
                               - ((((((v19 >> 16) & 0x7FFF) * (unsigned __int128)0xF0F0F0F0F0F0F0F1LL) >> 64)
                                 & 0xFFFFFFFFFFFFFFF0LL)
                                + ((v19 >> 16) & 0x7FFF) / 0x11)];
            v48 = 771;
            v47[0] = "struct.";
            sub_16E2F40((__int64)v47, (__int64)&v65);
            v20 = (unsigned int)v66;
            v21 = v65;
          }
          else
          {
            v20 = 7;
            v47[0] = "struct.";
            v21 = "struct.";
            v48 = 259;
          }
          sub_1643660(v22, v21, v20);
          if ( v65 != (char *)v67 )
            _libc_free((unsigned __int64)v65);
        }
      }
      ++v18;
    }
    while ( v17 != v18 );
    v42 = v19;
    v2 = v44;
  }
  v24 = *(__int64 **)(a1 + 8);
  v25 = *v24;
  v26 = v24[1];
  if ( v25 == v26 )
LABEL_87:
    BUG();
  while ( *(_UNKNOWN **)v25 != &unk_4F9B6E8 )
  {
    v25 += 16;
    if ( v26 == v25 )
      goto LABEL_87;
  }
  v41 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v25 + 8) + 104LL))(*(_QWORD *)(v25 + 8), &unk_4F9B6E8);
  v43 = v2 + 3;
  v45 = (_QWORD *)v2[4];
  if ( v2 + 3 != v45 )
  {
    do
    {
      v27 = (__int64)(v45 - 7);
      if ( !v45 )
        v27 = 0;
      v28 = sub_1649960(v27);
      v30 = v28;
      v31 = v29;
      if ( v29 <= 4 )
      {
        if ( !v29 )
        {
          if ( !sub_149CB50(*(_QWORD *)(v41 + 360), v27, (unsigned int *)v47) )
            goto LABEL_55;
          goto LABEL_73;
        }
      }
      else if ( *(_DWORD *)v28 == 1836477548 && v28[4] == 46 )
      {
        goto LABEL_73;
      }
      if ( *v28 != 1 )
      {
        v32 = (char **)v27;
        if ( !sub_149CB50(*(_QWORD *)(v41 + 360), v27, (unsigned int *)v47) )
        {
          if ( v31 != 4 || *(_DWORD *)v30 != 1852399981 )
          {
LABEL_55:
            v42 = 1103515245 * v42 + 12345;
            v33 = off_49852C0[((v42 >> 16) & 0x7FFF)
                            - (((v42 >> 16) & 0x7FFF) / 0x11
                             + (((0xF0F0F0F0F0F0F0F1LL * (unsigned __int128)((v42 >> 16) & 0x7FFF)) >> 64)
                              & 0xFFFFFFFFFFFFFFF0LL))];
            v67[0] = 257;
            if ( *v33 )
            {
              v65 = v33;
              LOBYTE(v67[0]) = 3;
            }
            v32 = &v65;
            sub_164B780(v27, (__int64 *)&v65);
          }
          if ( (*(_BYTE *)(v27 + 18) & 1) != 0 )
          {
            sub_15E08E0(v27, (__int64)v32);
            v34 = *(_QWORD *)(v27 + 88);
            if ( (*(_BYTE *)(v27 + 18) & 1) != 0 )
              sub_15E08E0(v27, (__int64)v32);
            v35 = *(_QWORD *)(v27 + 88);
          }
          else
          {
            v34 = *(_QWORD *)(v27 + 88);
            v35 = v34;
          }
          v36 = v35 + 40LL * *(_QWORD *)(v27 + 96);
          if ( v36 != v34 )
          {
            do
            {
              while ( !*(_BYTE *)(*(_QWORD *)v34 + 8LL) )
              {
                v34 += 40;
                if ( v34 == v36 )
                  goto LABEL_65;
              }
              v37 = v34;
              v46 = v36;
              v34 += 40;
              v65 = "arg";
              v67[0] = 259;
              sub_164B780(v37, (__int64 *)&v65);
              v36 = v46;
            }
            while ( v34 != v46 );
          }
LABEL_65:
          for ( i = *(_QWORD *)(v27 + 80); v27 + 72 != i; i = *(_QWORD *)(i + 8) )
          {
            v65 = "bb";
            v67[0] = 259;
            if ( !i )
            {
              sub_164B780(0, (__int64 *)&v65);
              BUG();
            }
            sub_164B780(i - 24, (__int64 *)&v65);
            for ( j = *(_QWORD *)(i + 24); i + 16 != j; j = *(_QWORD *)(j + 8) )
            {
              if ( !j )
                BUG();
              if ( *(_BYTE *)(*(_QWORD *)(j - 24) + 8LL) )
              {
                v65 = "tmp";
                v67[0] = 259;
                sub_164B780(j - 24, (__int64 *)&v65);
              }
            }
          }
        }
      }
LABEL_73:
      v45 = (_QWORD *)v45[1];
    }
    while ( v43 != v45 );
  }
  if ( v61 )
    j_j___libc_free_0(v61, v63 - (_QWORD)v61);
  j___libc_free_0(v58);
  j___libc_free_0(v54);
  j___libc_free_0(v50);
  return 1;
}
