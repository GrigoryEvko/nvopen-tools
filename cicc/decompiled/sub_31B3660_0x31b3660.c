// Function: sub_31B3660
// Address: 0x31b3660
//
__int64 __fastcall sub_31B3660(__int64 a1)
{
  __int64 *v1; // rbx
  __int64 v2; // rax
  int v3; // ecx
  __int64 *v4; // r12
  __int64 v5; // rax
  _QWORD *v6; // rbx
  _QWORD *v7; // r12
  unsigned __int64 v8; // rdi
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // r9
  int v14; // r10d
  _QWORD *v15; // r13
  unsigned int v16; // eax
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // rdx
  __int64 v20; // rax
  _QWORD *v21; // r13
  __int64 v22; // rsi
  _QWORD *v23; // rax
  _QWORD *v24; // r13
  _QWORD *v25; // rdx
  _QWORD *v26; // rbx
  _QWORD *v27; // r12
  _QWORD *v28; // rbx
  __int64 v29; // rcx
  __int64 v30; // r13
  _QWORD *v31; // r15
  __int64 v32; // rdx
  char *v33; // rdi
  __int64 v34; // r12
  unsigned __int64 v35; // rax
  char *v36; // r14
  __int64 v37; // r15
  char *i; // r13
  __int64 v39; // rax
  char *v40; // r12
  int v41; // ecx
  _QWORD *v42; // rax
  _QWORD *v43; // rdi
  unsigned int v44; // eax
  __int64 v45; // rsi
  int v46; // edx
  unsigned int v47; // eax
  __int64 v48; // rdi
  int v49; // esi
  _QWORD *v50; // rdx
  char *v52; // [rsp+10h] [rbp-80h]
  __int64 v53; // [rsp+18h] [rbp-78h]
  _QWORD *v54; // [rsp+18h] [rbp-78h]
  __int64 v55; // [rsp+18h] [rbp-78h]
  unsigned int v56; // [rsp+18h] [rbp-78h]
  _QWORD v57[4]; // [rsp+20h] [rbp-70h] BYREF
  __int64 v58; // [rsp+40h] [rbp-50h] BYREF
  _QWORD *v59; // [rsp+48h] [rbp-48h]
  __int64 v60; // [rsp+50h] [rbp-40h]
  unsigned int v61; // [rsp+58h] [rbp-38h]

  v1 = *(__int64 **)(a1 + 64);
  v2 = *(unsigned int *)(a1 + 80);
  v3 = *(_DWORD *)(a1 + 72);
  v58 = 0;
  v59 = 0;
  v4 = &v1[v2];
  v60 = 0;
  v61 = 0;
  if ( v3 && v1 != v4 )
  {
    while ( *v1 == -8192 || *v1 == -4096 )
    {
      if ( ++v1 == v4 )
        goto LABEL_2;
    }
    if ( v1 != v4 )
    {
      while ( 1 )
      {
        v10 = *v1;
        v11 = sub_318B4F0(*v1);
        v12 = v11;
        if ( !v61 )
          break;
        v13 = v61 - 1;
        v14 = 1;
        v15 = 0;
        v16 = ((unsigned int)v11 >> 4) ^ ((unsigned int)v11 >> 9);
        LODWORD(v17) = v13 & v16;
        v18 = (__int64)&v59[9 * ((unsigned int)v13 & v16)];
        v19 = *(_QWORD *)v18;
        if ( v12 == *(_QWORD *)v18 )
        {
LABEL_19:
          v20 = *(unsigned int *)(v18 + 16);
          v21 = (_QWORD *)(v18 + 8);
          if ( *(unsigned int *)(v18 + 20) < (unsigned __int64)(v20 + 1) )
          {
            v55 = v18;
            sub_C8D5F0(v18 + 8, (const void *)(v18 + 24), v20 + 1, 8u, v18, v13);
            v20 = *(unsigned int *)(v55 + 16);
          }
          goto LABEL_21;
        }
        while ( v19 != -4096 )
        {
          if ( !v15 && v19 == -8192 )
            v15 = (_QWORD *)v18;
          v17 = (unsigned int)v13 & ((_DWORD)v17 + v14);
          v18 = (__int64)&v59[9 * v17];
          v19 = *(_QWORD *)v18;
          if ( v12 == *(_QWORD *)v18 )
            goto LABEL_19;
          ++v14;
        }
        if ( !v15 )
          v15 = (_QWORD *)v18;
        ++v58;
        v41 = v60 + 1;
        if ( 4 * ((int)v60 + 1) >= 3 * v61 )
          goto LABEL_89;
        if ( v61 - HIDWORD(v60) - v41 <= v61 >> 3 )
        {
          v56 = v16;
          sub_31B3340((__int64)&v58, v61);
          if ( !v61 )
            goto LABEL_106;
          v43 = 0;
          v44 = (v61 - 1) & v56;
          v15 = &v59[9 * v44];
          v45 = *v15;
          v41 = v60 + 1;
          v46 = 1;
          if ( v12 != *v15 )
          {
            while ( v45 != -4096 )
            {
              if ( !v43 && v45 == -8192 )
                v43 = v15;
              v44 = (v61 - 1) & (v44 + v46);
              v15 = &v59[9 * v44];
              v45 = *v15;
              if ( v12 == *v15 )
                goto LABEL_79;
              ++v46;
            }
            if ( v43 )
              v15 = v43;
          }
        }
LABEL_79:
        LODWORD(v60) = v41;
        if ( *v15 != -4096 )
          --HIDWORD(v60);
        v42 = v15 + 3;
        *v15 = v12;
        v21 = v15 + 1;
        *v21 = v42;
        v21[1] = 0x600000000LL;
        v20 = 0;
LABEL_21:
        *(_QWORD *)(*v21 + 8 * v20) = v10;
        ++*((_DWORD *)v21 + 2);
        do
        {
          if ( ++v1 == v4 )
            goto LABEL_26;
        }
        while ( *v1 == -4096 || *v1 == -8192 );
        if ( v1 == v4 )
        {
LABEL_26:
          if ( (_DWORD)v60 )
          {
            v22 = v61;
            v23 = v59;
            v24 = &v59[9 * v61];
            if ( v59 == v24 )
              goto LABEL_32;
            v25 = v59;
            while ( 1 )
            {
              v26 = v25;
              if ( *v25 != -4096 && *v25 != -8192 )
                break;
              v25 += 9;
              if ( v24 == v25 )
                goto LABEL_32;
            }
            if ( v25 == v24 )
            {
LABEL_32:
              v27 = &v23[9 * v22];
              if ( v23 != v27 )
              {
                while ( 1 )
                {
                  v28 = v23;
                  if ( *v23 != -8192 && *v23 != -4096 )
                    break;
                  v23 += 9;
                  if ( v27 == v23 )
                    goto LABEL_2;
                }
                if ( v27 != v23 )
                {
                  do
                  {
                    v29 = v28[1];
                    v30 = v29 + 8LL * *((unsigned int *)v28 + 4);
                    v53 = v29;
                    while ( v53 != v30 )
                    {
                      while ( 1 )
                      {
                        v31 = *(_QWORD **)(v30 - 8);
                        sub_318EB30(v57, (__int64)v31);
                        if ( !v57[0] && !v57[1] )
                          break;
                        v30 -= 8;
                        if ( v53 == v30 )
                          goto LABEL_44;
                      }
                      v30 -= 8;
                      sub_318C940(v31, (__int64)v31, v32);
                    }
LABEL_44:
                    v28 += 9;
                    if ( v28 == v27 )
                      break;
                    while ( *v28 == -8192 || *v28 == -4096 )
                    {
                      v28 += 9;
                      if ( v27 == v28 )
                        goto LABEL_2;
                    }
                  }
                  while ( v28 != v27 );
                }
              }
            }
            else
            {
              v54 = &v59[9 * v61];
              do
              {
                v33 = (char *)v26[1];
                v34 = 8LL * *((unsigned int *)v26 + 4);
                v52 = &v33[v34];
                if ( v33 != &v33[v34] )
                {
                  _BitScanReverse64(&v35, v34 >> 3);
                  sub_31AFAC0(v33, (__int64 *)&v33[v34], 2LL * (int)(63 - (v35 ^ 0x3F)));
                  if ( (unsigned __int64)v34 <= 0x80 )
                  {
                    sub_31AFD00(v33, v52);
                  }
                  else
                  {
                    v36 = v33 + 128;
                    sub_31AFD00(v33, v33 + 128);
                    if ( &v33[v34] != v33 + 128 )
                    {
                      do
                      {
                        v37 = *(_QWORD *)v36;
                        for ( i = v36; ; *((_QWORD *)i + 1) = *(_QWORD *)i )
                        {
                          v39 = *((_QWORD *)i - 1);
                          v40 = i;
                          i -= 8;
                          if ( !sub_B445A0(*(_QWORD *)(v37 + 16), *(_QWORD *)(v39 + 16)) )
                            break;
                        }
                        *(_QWORD *)v40 = v37;
                        v36 += 8;
                      }
                      while ( v52 != v36 );
                    }
                  }
                }
                v26 += 9;
                if ( v26 == v54 )
                  break;
                while ( *v26 == -8192 || *v26 == -4096 )
                {
                  v26 += 9;
                  if ( v54 == v26 )
                    goto LABEL_64;
                }
              }
              while ( v26 != v54 );
LABEL_64:
              if ( (_DWORD)v60 )
              {
                v23 = v59;
                v22 = v61;
                goto LABEL_32;
              }
            }
          }
          goto LABEL_2;
        }
      }
      ++v58;
LABEL_89:
      sub_31B3340((__int64)&v58, 2 * v61);
      if ( !v61 )
      {
LABEL_106:
        LODWORD(v60) = v60 + 1;
        BUG();
      }
      v47 = (v61 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v15 = &v59[9 * v47];
      v48 = *v15;
      v41 = v60 + 1;
      if ( v12 != *v15 )
      {
        v49 = 1;
        v50 = 0;
        while ( v48 != -4096 )
        {
          if ( v48 == -8192 && !v50 )
            v50 = v15;
          v47 = (v61 - 1) & (v47 + v49);
          v15 = &v59[9 * v47];
          v48 = *v15;
          if ( v12 == *v15 )
            goto LABEL_79;
          ++v49;
        }
        if ( v50 )
          v15 = v50;
      }
      goto LABEL_79;
    }
  }
LABEL_2:
  sub_31B1680(a1 + 56);
  v5 = v61;
  if ( v61 )
  {
    v6 = v59;
    v7 = &v59[9 * v61];
    do
    {
      if ( *v6 != -8192 && *v6 != -4096 )
      {
        v8 = v6[1];
        if ( (_QWORD *)v8 != v6 + 3 )
          _libc_free(v8);
      }
      v6 += 9;
    }
    while ( v7 != v6 );
    v5 = v61;
  }
  return sub_C7D6A0((__int64)v59, 72 * v5, 8);
}
