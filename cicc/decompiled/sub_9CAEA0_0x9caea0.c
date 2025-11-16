// Function: sub_9CAEA0
// Address: 0x9caea0
//
__int64 *__fastcall sub_9CAEA0(__int64 *a1, _QWORD *a2, unsigned __int8 *a3, __int64 a4)
{
  __int64 v5; // rax
  int v6; // edx
  int v7; // eax
  __int64 v8; // rbx
  __int64 v9; // rax
  __int64 v10; // rdx
  __int64 v11; // r13
  int v12; // r13d
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rsi
  unsigned int *v17; // rbx
  unsigned int i; // r14d
  int v19; // edx
  __int64 v20; // rsi
  __int64 v21; // r8
  __int64 v22; // rax
  unsigned int v23; // eax
  unsigned int v25; // eax
  __int64 v26; // rsi
  __int64 v27; // rdx
  __int64 v28; // rax
  __int64 v29; // rdx
  const char *v30; // r15
  __int64 v31; // r13
  const char *v32; // r14
  const char *v33; // rbx
  __int64 v34; // rax
  const char *v35; // r15
  const char *v36; // r13
  const char *v37; // r15
  const char *v38; // rbx
  __int64 v39; // rdx
  __int64 v40; // rax
  __int64 v41; // rsi
  const char *v42; // r12
  __int64 v43; // rdx
  const char *v44; // r13
  const char *v45; // rbx
  __int64 v46; // r15
  const char *v47; // r14
  const char *v48; // r15
  const char *v49; // r13
  const char *v50; // rbx
  const char *v51; // [rsp+8h] [rbp-3A8h]
  const char *v54; // [rsp+28h] [rbp-388h]
  const char *v55; // [rsp+28h] [rbp-388h]
  unsigned int v56; // [rsp+34h] [rbp-37Ch] BYREF
  __int64 v57; // [rsp+38h] [rbp-378h] BYREF
  unsigned int v58[8]; // [rsp+40h] [rbp-370h] BYREF
  char v59; // [rsp+60h] [rbp-350h]
  char v60; // [rsp+61h] [rbp-34Fh]
  const char *v61; // [rsp+70h] [rbp-340h] BYREF
  unsigned int v62; // [rsp+78h] [rbp-338h]
  _BYTE v63[16]; // [rsp+80h] [rbp-330h] BYREF
  char v64; // [rsp+90h] [rbp-320h]
  char v65; // [rsp+91h] [rbp-31Fh]

  v5 = *((_QWORD *)a3 + 9);
  v6 = *a3;
  v57 = v5;
  v7 = v6 - 29;
  v58[0] = 0;
  if ( v6 != 40 )
    goto LABEL_2;
LABEL_18:
  v8 = 32LL * (unsigned int)sub_B491D0(a3);
  if ( (a3[7] & 0x80u) == 0 )
    goto LABEL_19;
  while ( 1 )
  {
    v9 = sub_BD2BC0(a3);
    v11 = v9 + v10;
    if ( (a3[7] & 0x80u) == 0 )
    {
      if ( (unsigned int)(v11 >> 4) )
LABEL_118:
        BUG();
LABEL_19:
      v15 = 0;
      goto LABEL_11;
    }
    if ( !(unsigned int)((v11 - sub_BD2BC0(a3)) >> 4) )
      goto LABEL_19;
    if ( (a3[7] & 0x80u) == 0 )
      goto LABEL_118;
    v12 = *(_DWORD *)(sub_BD2BC0(a3) + 8);
    if ( (a3[7] & 0x80u) == 0 )
      BUG();
    v13 = sub_BD2BC0(a3);
    v15 = 32LL * (unsigned int)(*(_DWORD *)(v13 + v14 - 4) - v12);
LABEL_11:
    v16 = v58[0];
    if ( v58[0] == (unsigned int)((32LL * (*((_DWORD *)a3 + 1) & 0x7FFFFFF) - 32 - v8 - v15) >> 5) )
      break;
    v17 = (unsigned int *)&unk_3F22290;
    for ( i = 81; ; i = *v17 )
    {
      if ( (unsigned __int8)sub_A74710(&v57, (unsigned int)(v16 + 1), i) )
      {
        v61 = (const char *)sub_A747F0(&v57, v58[0] + 1, i);
        if ( !sub_A72A60(&v61) )
        {
          v20 = sub_9CAE40(a2, *(_DWORD *)(a4 + 4LL * v58[0]));
          if ( !v20 )
          {
            v65 = 1;
            v61 = "Missing element type for typed attribute upgrade";
            v64 = 3;
            sub_9C81F0(a1, (__int64)(a2 + 1), (__int64)&v61);
            return a1;
          }
          switch ( i )
          {
            case 'S':
              v21 = sub_A77E50(a2[54], v20);
              break;
            case 'U':
              v21 = sub_A77E40(a2[54], v20);
              break;
            case 'Q':
              v21 = sub_A77E30(a2[54], v20);
              break;
            default:
              goto LABEL_119;
          }
          v57 = sub_A7B660(&v57, a2[54], v58, 1, v21);
        }
      }
      ++v17;
      LODWORD(v16) = v58[0];
      if ( &unk_3F2229C == (_UNKNOWN *)v17 )
        break;
    }
    v19 = *a3;
    ++v58[0];
    v7 = v19 - 29;
    if ( v19 == 40 )
      goto LABEL_18;
LABEL_2:
    v8 = 0;
    if ( v7 != 56 )
    {
      if ( v7 != 5 )
LABEL_119:
        BUG();
      v8 = 64;
    }
    if ( (a3[7] & 0x80u) == 0 )
      goto LABEL_19;
  }
  v22 = *((_QWORD *)a3 - 4);
  if ( *(_BYTE *)v22 != 25 )
    goto LABEL_31;
  v16 = *(_QWORD *)(v22 + 56);
  v29 = *(_QWORD *)(v22 + 64);
  v56 = 0;
  sub_B428A0(&v61, v16, v29);
  v30 = v61;
  v31 = 192LL * v62;
  v54 = &v61[v31];
  if ( v61 == &v61[v31] )
  {
LABEL_83:
    if ( v54 != v63 )
      _libc_free(v54, v16);
LABEL_31:
    v23 = sub_B49240(a3, v16);
    if ( v23 == 1147 )
      goto LABEL_42;
    if ( v23 > 0x47B )
    {
      if ( v23 == 3788 )
        goto LABEL_42;
      if ( v23 <= 0xECC )
      {
        if ( v23 != 3382 && v23 != 3388 )
          goto LABEL_37;
        goto LABEL_42;
      }
      if ( v23 == 3790 )
        goto LABEL_42;
    }
    else
    {
      if ( v23 == 563 )
        goto LABEL_42;
      if ( v23 > 0x233 )
      {
        if ( v23 != 566 && v23 != 1145 )
          goto LABEL_37;
LABEL_42:
        v25 = sub_B49240(a3, v16);
        if ( v25 != 3788 )
        {
          if ( v25 > 0xECC )
          {
            if ( v25 != 3790 )
              goto LABEL_45;
          }
          else if ( (v25 & 0xFFFFFFFD) != 0x479 )
          {
LABEL_45:
            v58[0] = 0;
            v26 = 0;
            goto LABEL_46;
          }
        }
        v58[0] = 1;
        v26 = 1;
LABEL_46:
        if ( !sub_A74920(&v57, v26) )
        {
          v27 = sub_9CAE40(a2, *(_DWORD *)(a4 + 4LL * v58[0]));
          if ( !v27 )
          {
            v65 = 1;
            v61 = "Missing element type for elementtype upgrade";
            v64 = 3;
            sub_9C81F0(a1, (__int64)(a2 + 1), (__int64)&v61);
            return a1;
          }
          v28 = sub_A77D30(a2[54], 82, v27);
          v57 = sub_A7B660(&v57, a2[54], v58, 1, v28);
        }
        goto LABEL_37;
      }
      if ( ((v23 - 287) & 0xFFFFFFFD) == 0 )
        goto LABEL_42;
    }
LABEL_37:
    *((_QWORD *)a3 + 9) = v57;
    *a1 = 1;
    return a1;
  }
  while ( 2 )
  {
    while ( 2 )
    {
      if ( *(_DWORD *)v30 )
      {
        if ( *(_DWORD *)v30 != 1 || !v30[10] )
        {
          v30 += 192;
          if ( v54 == v30 )
            goto LABEL_61;
          continue;
        }
      }
      else if ( !v30[10] )
      {
        goto LABEL_60;
      }
      break;
    }
    v16 = v56;
    if ( sub_A74920(&v57, v56) )
    {
LABEL_60:
      ++v56;
      v30 += 192;
      if ( v54 == v30 )
      {
LABEL_61:
        v51 = v61;
        v54 = &v61[192 * v62];
        if ( v61 != v54 )
        {
          do
          {
            v54 -= 192;
            v32 = (const char *)*((_QWORD *)v54 + 8);
            v33 = &v32[56 * *((unsigned int *)v54 + 18)];
            if ( v32 != v33 )
            {
              do
              {
                v34 = *((unsigned int *)v33 - 10);
                v35 = (const char *)*((_QWORD *)v33 - 6);
                v33 -= 56;
                v34 *= 32;
                v36 = &v35[v34];
                if ( v35 != &v35[v34] )
                {
                  do
                  {
                    v36 -= 32;
                    if ( *(const char **)v36 != v36 + 16 )
                    {
                      v16 = *((_QWORD *)v36 + 2) + 1LL;
                      j_j___libc_free_0(*(_QWORD *)v36, v16);
                    }
                  }
                  while ( v35 != v36 );
                  v35 = (const char *)*((_QWORD *)v33 + 1);
                }
                if ( v35 != v33 + 24 )
                  _libc_free(v35, v16);
              }
              while ( v32 != v33 );
              v32 = (const char *)*((_QWORD *)v54 + 8);
            }
            if ( v32 != v54 + 80 )
              _libc_free(v32, v16);
            v37 = (const char *)*((_QWORD *)v54 + 2);
            v38 = &v37[32 * *((unsigned int *)v54 + 6)];
            if ( v37 != v38 )
            {
              do
              {
                v38 -= 32;
                if ( *(const char **)v38 != v38 + 16 )
                {
                  v16 = *((_QWORD *)v38 + 2) + 1LL;
                  j_j___libc_free_0(*(_QWORD *)v38, v16);
                }
              }
              while ( v37 != v38 );
              v37 = (const char *)*((_QWORD *)v54 + 2);
            }
            if ( v37 != v54 + 32 )
              _libc_free(v37, v16);
          }
          while ( v51 != v54 );
          v54 = v61;
        }
        goto LABEL_83;
      }
      continue;
    }
    break;
  }
  v39 = sub_9CAE40(a2, *(_DWORD *)(a4 + 4LL * v56));
  if ( v39 )
  {
    v40 = sub_A77D30(a2[54], 82, v39);
    v16 = a2[54];
    v57 = sub_A7B660(&v57, v16, &v56, 1, v40);
    goto LABEL_60;
  }
  *(_QWORD *)v58 = "Missing element type for inline asm upgrade";
  v41 = (__int64)(a2 + 1);
  v60 = 1;
  v59 = 3;
  sub_9C81F0(a1, (__int64)(a2 + 1), (__int64)v58);
  v55 = v61;
  v42 = &v61[192 * v62];
  if ( v61 != v42 )
  {
    do
    {
      v43 = *((unsigned int *)v42 - 30);
      v44 = (const char *)*((_QWORD *)v42 - 16);
      v42 -= 192;
      v45 = &v44[56 * v43];
      if ( v44 != v45 )
      {
        do
        {
          v46 = *((unsigned int *)v45 - 10);
          v47 = (const char *)*((_QWORD *)v45 - 6);
          v45 -= 56;
          v48 = &v47[32 * v46];
          if ( v47 != v48 )
          {
            do
            {
              v48 -= 32;
              if ( *(const char **)v48 != v48 + 16 )
              {
                v41 = *((_QWORD *)v48 + 2) + 1LL;
                j_j___libc_free_0(*(_QWORD *)v48, v41);
              }
            }
            while ( v47 != v48 );
            v47 = (const char *)*((_QWORD *)v45 + 1);
          }
          if ( v47 != v45 + 24 )
            _libc_free(v47, v41);
        }
        while ( v44 != v45 );
        v44 = (const char *)*((_QWORD *)v42 + 8);
      }
      if ( v44 != v42 + 80 )
        _libc_free(v44, v41);
      v49 = (const char *)*((_QWORD *)v42 + 2);
      v50 = &v49[32 * *((unsigned int *)v42 + 6)];
      if ( v49 != v50 )
      {
        do
        {
          v50 -= 32;
          if ( *(const char **)v50 != v50 + 16 )
          {
            v41 = *((_QWORD *)v50 + 2) + 1LL;
            j_j___libc_free_0(*(_QWORD *)v50, v41);
          }
        }
        while ( v49 != v50 );
        v49 = (const char *)*((_QWORD *)v42 + 2);
      }
      if ( v49 != v42 + 32 )
        _libc_free(v49, v41);
    }
    while ( v55 != v42 );
    v55 = v61;
  }
  if ( v55 != v63 )
    _libc_free(v55, v41);
  return a1;
}
