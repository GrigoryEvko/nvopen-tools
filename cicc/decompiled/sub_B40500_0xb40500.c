// Function: sub_B40500
// Address: 0xb40500
//
__int64 __fastcall sub_B40500(__int64 a1, char *p_s1)
{
  __int64 v2; // rax
  __int64 v3; // rcx
  __int64 v4; // r12
  __int64 v5; // rbx
  __int64 v6; // r15
  _DWORD *v7; // r14
  char *v8; // rsi
  char *v9; // rdi
  __int64 v10; // rdx
  __int64 v11; // r13
  unsigned int v12; // eax
  _DWORD *v13; // r14
  int v14; // eax
  unsigned int v15; // esi
  _DWORD *v16; // rdi
  char v17; // al
  __int16 v18; // ax
  __int64 v19; // rbx
  _QWORD *v20; // r15
  int v21; // r12d
  size_t v22; // r13
  __int64 v23; // r14
  __int16 v24; // r8
  int v25; // eax
  __int64 v26; // rdx
  __int64 v27; // rax
  __int64 v28; // r13
  __int64 v29; // rdx
  __int64 v30; // r12
  __int64 v31; // r15
  __int64 v32; // rbx
  _QWORD *v33; // r14
  _QWORD *v34; // rbx
  _QWORD *v35; // r12
  _QWORD *v36; // rbx
  __int64 v37; // rdi
  __int64 result; // rax
  size_t v39; // rdx
  int v40; // eax
  void *v41; // r13
  _QWORD *v42; // r8
  unsigned int v43; // eax
  __int64 v44; // rdx
  char *v45; // rdi
  size_t v46; // rdx
  int v47; // eax
  _QWORD *v48; // r8
  char *v49; // [rsp+0h] [rbp-C0h]
  int v50; // [rsp+8h] [rbp-B8h]
  __int16 v51; // [rsp+Ch] [rbp-B4h]
  __int16 v52; // [rsp+Eh] [rbp-B2h]
  _QWORD *v53; // [rsp+10h] [rbp-B0h]
  __int64 v54; // [rsp+18h] [rbp-A8h]
  __int64 v55; // [rsp+20h] [rbp-A0h]
  __int64 v56; // [rsp+28h] [rbp-98h]
  __int64 v57; // [rsp+30h] [rbp-90h]
  __int64 v58; // [rsp+38h] [rbp-88h]
  __int64 v59; // [rsp+40h] [rbp-80h]
  __int64 v60; // [rsp+48h] [rbp-78h]
  _DWORD *v61; // [rsp+58h] [rbp-68h] BYREF
  void *s1; // [rsp+60h] [rbp-60h] BYREF
  size_t n; // [rsp+68h] [rbp-58h]
  _QWORD v64[2]; // [rsp+70h] [rbp-50h] BYREF
  __int16 v65; // [rsp+80h] [rbp-40h]

  v2 = *(unsigned int *)(a1 + 1560);
  v3 = *(_QWORD *)(a1 + 1552);
  v57 = a1;
  v49 = p_s1;
  v59 = v3 + 832 * v2;
  if ( v3 == v59 )
  {
    v26 = v59;
    goto LABEL_60;
  }
  v4 = a1;
  v5 = v3;
  v6 = v3;
  do
  {
    if ( *(_QWORD *)(v6 + 824) >= *(_QWORD *)v49 )
    {
      if ( v6 == v5 )
        goto LABEL_5;
      v8 = *(char **)v6;
      v9 = *(char **)v5;
      if ( *(_QWORD *)v6 != v6 + 16 )
      {
        if ( v9 == (char *)(v5 + 16) )
        {
          *(_QWORD *)v5 = v8;
          *(_QWORD *)(v5 + 8) = *(_QWORD *)(v6 + 8);
          *(_QWORD *)(v5 + 16) = *(_QWORD *)(v6 + 16);
        }
        else
        {
          *(_QWORD *)v5 = v8;
          v10 = *(_QWORD *)(v5 + 16);
          *(_QWORD *)(v5 + 8) = *(_QWORD *)(v6 + 8);
          *(_QWORD *)(v5 + 16) = *(_QWORD *)(v6 + 16);
          if ( v9 )
          {
            *(_QWORD *)v6 = v9;
            *(_QWORD *)(v6 + 16) = v10;
            goto LABEL_13;
          }
        }
        *(_QWORD *)v6 = v6 + 16;
        v9 = (char *)(v6 + 16);
LABEL_13:
        *(_QWORD *)(v6 + 8) = 0;
        *v9 = 0;
        *(_BYTE *)(v5 + 32) = *(_BYTE *)(v6 + 32);
        *(_BYTE *)(v5 + 33) = *(_BYTE *)(v6 + 33);
        sub_B3E030((__int64 *)(v5 + 40), v6 + 40);
        p_s1 = (char *)v5;
        *(_QWORD *)(v5 + 824) = *(_QWORD *)(v6 + 824);
        v11 = 0x4EC4EC4EC4EC4EC5LL * ((v5 - *(_QWORD *)(v4 + 1552)) >> 6);
        if ( (unsigned __int8)sub_B3C4F0(v4, v5, &v61) )
        {
          v7 = v61 + 10;
LABEL_4:
          *v7 = v11;
LABEL_5:
          v5 += 832;
          goto LABEL_6;
        }
        v12 = *(_DWORD *)(v4 + 8);
        v13 = v61;
        ++*(_QWORD *)v4;
        s1 = v13;
        v14 = (v12 >> 1) + 1;
        if ( (*(_BYTE *)(v4 + 8) & 1) != 0 )
        {
          v15 = 32;
          if ( (unsigned int)(4 * v14) < 0x60 )
          {
LABEL_16:
            if ( v15 - (v14 + *(_DWORD *)(v4 + 12)) > v15 >> 3 )
              goto LABEL_17;
            goto LABEL_73;
          }
        }
        else
        {
          v15 = *(_DWORD *)(v4 + 24);
          if ( 3 * v15 > 4 * v14 )
            goto LABEL_16;
        }
        v15 *= 2;
LABEL_73:
        sub_B3F770(v4, v15);
        sub_B3C4F0(v4, v5, &s1);
        v13 = s1;
        v14 = (*(_DWORD *)(v4 + 8) >> 1) + 1;
LABEL_17:
        *(_DWORD *)(v4 + 8) = *(_DWORD *)(v4 + 8) & 1 | (2 * v14);
        if ( *((_WORD *)v13 + 16) || *((_BYTE *)v13 + 32) && !*((_BYTE *)v13 + 33) && *((_QWORD *)v13 + 1) )
          --*(_DWORD *)(v4 + 12);
        v16 = v13;
        p_s1 = (char *)v5;
        v7 = v13 + 10;
        sub_2240AE0(v16, v5);
        *((_BYTE *)v7 - 8) = *(_BYTE *)(v5 + 32);
        v17 = *(_BYTE *)(v5 + 33);
        *v7 = 0;
        *((_BYTE *)v7 - 7) = v17;
        goto LABEL_4;
      }
      v39 = *(_QWORD *)(v6 + 8);
      if ( v39 )
      {
        if ( v39 == 1 )
        {
          *v9 = *(_BYTE *)(v6 + 16);
          v44 = *(_QWORD *)(v6 + 8);
          v45 = *(char **)v5;
          *(_QWORD *)(v5 + 8) = v44;
          v45[v44] = 0;
          v9 = *(char **)v6;
          goto LABEL_13;
        }
        memcpy(v9, v8, v39);
        v39 = *(_QWORD *)(v6 + 8);
        v9 = *(char **)v5;
      }
      *(_QWORD *)(v5 + 8) = v39;
      v9[v39] = 0;
      v9 = *(char **)v6;
      goto LABEL_13;
    }
    if ( (*(_BYTE *)(v4 + 8) & 1) != 0 )
    {
      LODWORD(v58) = 31;
      v56 = v4 + 16;
    }
    else
    {
      v56 = *(_QWORD *)(v4 + 16);
      v40 = *(_DWORD *)(v4 + 24);
      LODWORD(v58) = v40 - 1;
      if ( !v40 )
        goto LABEL_6;
    }
    n = 0;
    s1 = v64;
    LOBYTE(v64[0]) = 0;
    v65 = 0;
    p_s1 = (char *)(*(_QWORD *)v6 + *(_QWORD *)(v6 + 8));
    LODWORD(v60) = v58 & sub_B3B940(*(char **)v6, p_s1);
    v18 = *(_WORD *)(v6 + 32);
    v55 = v5;
    v19 = v6;
    v20 = s1;
    v52 = v18;
    v54 = v4;
    v21 = 1;
    v51 = v65;
    v53 = v64;
    v22 = n;
    while ( 1 )
    {
      v23 = v56 + 48LL * (unsigned int)v60;
      v24 = *(_WORD *)(v23 + 32);
      if ( v24 == v52 )
      {
        if ( !*(_BYTE *)(v19 + 32) )
          break;
        if ( *(_BYTE *)(v19 + 33) )
          break;
        v46 = *(_QWORD *)(v23 + 8);
        if ( v46 == *(_QWORD *)(v19 + 8) )
        {
          v50 = *(unsigned __int16 *)(v23 + 32);
          if ( !v46 )
            break;
          p_s1 = *(char **)v19;
          v47 = memcmp(*(const void **)v23, *(const void **)v19, v46);
          v24 = v50;
          if ( !v47 )
            break;
        }
      }
      if ( v24 == v51 )
      {
        if ( !*(_BYTE *)(v23 + 32)
          || *(_BYTE *)(v23 + 33)
          || v22 == *(_QWORD *)(v23 + 8) && (!v22 || (p_s1 = *(char **)v23, !memcmp(v20, *(const void **)v23, v22))) )
        {
          v48 = v20;
          v4 = v54;
          v6 = v19;
          v5 = v55;
          if ( v48 != v53 )
          {
            p_s1 = (char *)(v64[0] + 1LL);
            j_j___libc_free_0(v48, v64[0] + 1LL);
          }
          goto LABEL_6;
        }
      }
      v25 = v58 & (v21 + v60);
      ++v21;
      LODWORD(v60) = v25;
    }
    v41 = v53;
    v42 = v20;
    v4 = v54;
    v6 = v19;
    v5 = v55;
    if ( v42 != v53 )
      j_j___libc_free_0(v42, v64[0] + 1LL);
    p_s1 = (char *)&s1;
    s1 = v53;
    v65 = 257;
    n = 0;
    LOBYTE(v64[0]) = 0;
    sub_2240AE0(v23, &s1);
    *(_WORD *)(v23 + 32) = v65;
    if ( s1 != v41 )
    {
      p_s1 = (char *)(v64[0] + 1LL);
      j_j___libc_free_0(s1, v64[0] + 1LL);
    }
    v43 = *(_DWORD *)(v4 + 8);
    ++*(_DWORD *)(v4 + 12);
    *(_DWORD *)(v4 + 8) = (2 * (v43 >> 1) - 2) | v43 & 1;
LABEL_6:
    v6 += 832;
  }
  while ( v59 != v6 );
  v58 = v5;
  v26 = *(_QWORD *)(v57 + 1552);
  v59 = v26 + 832LL * *(unsigned int *)(v57 + 1560);
  if ( v5 != v59 )
  {
    do
    {
      v59 -= 832;
      v27 = *(unsigned int *)(v59 + 48);
      v60 = *(_QWORD *)(v59 + 40);
      v28 = v60 + 192 * v27;
      if ( v60 != v28 )
      {
        do
        {
          v29 = *(unsigned int *)(v28 - 120);
          v30 = *(_QWORD *)(v28 - 128);
          v28 -= 192;
          v31 = v30 + 56 * v29;
          if ( v30 != v31 )
          {
            do
            {
              v32 = *(unsigned int *)(v31 - 40);
              v33 = *(_QWORD **)(v31 - 48);
              v31 -= 56;
              v34 = &v33[4 * v32];
              if ( v33 != v34 )
              {
                do
                {
                  v34 -= 4;
                  if ( (_QWORD *)*v34 != v34 + 2 )
                  {
                    p_s1 = (char *)(v34[2] + 1LL);
                    j_j___libc_free_0(*v34, p_s1);
                  }
                }
                while ( v33 != v34 );
                v33 = *(_QWORD **)(v31 + 8);
              }
              if ( v33 != (_QWORD *)(v31 + 24) )
                _libc_free(v33, p_s1);
            }
            while ( v30 != v31 );
            v30 = *(_QWORD *)(v28 + 64);
          }
          if ( v30 != v28 + 80 )
            _libc_free(v30, p_s1);
          v35 = *(_QWORD **)(v28 + 16);
          v36 = &v35[4 * *(unsigned int *)(v28 + 24)];
          if ( v35 != v36 )
          {
            do
            {
              v36 -= 4;
              if ( (_QWORD *)*v36 != v36 + 2 )
              {
                p_s1 = (char *)(v36[2] + 1LL);
                j_j___libc_free_0(*v36, p_s1);
              }
            }
            while ( v35 != v36 );
            v35 = *(_QWORD **)(v28 + 16);
          }
          if ( v35 != (_QWORD *)(v28 + 32) )
            _libc_free(v35, p_s1);
        }
        while ( v60 != v28 );
        v60 = *(_QWORD *)(v59 + 40);
      }
      if ( v60 != v59 + 56 )
        _libc_free(v60, p_s1);
      v37 = *(_QWORD *)v59;
      if ( *(_QWORD *)v59 != v59 + 16 )
      {
        p_s1 = (char *)(*(_QWORD *)(v59 + 16) + 1LL);
        v60 = *(_QWORD *)(v59 + 16);
        j_j___libc_free_0(v37, p_s1);
      }
    }
    while ( v58 != v59 );
    v26 = *(_QWORD *)(v57 + 1552);
  }
LABEL_60:
  result = 0x4EC4EC4EC4EC4EC5LL * ((v59 - v26) >> 6);
  *(_DWORD *)(v57 + 1560) = result;
  return result;
}
