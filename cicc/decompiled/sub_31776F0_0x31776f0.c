// Function: sub_31776F0
// Address: 0x31776f0
//
void __fastcall sub_31776F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r12
  __int64 v5; // r14
  unsigned __int8 *v6; // r15
  int v7; // eax
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rax
  unsigned __int64 v13; // rdx
  __int64 v14; // r15
  __int64 v15; // r13
  char *v16; // r14
  __int64 v17; // r12
  __int64 v18; // rax
  __int64 v19; // r12
  char *v20; // r12
  __int64 v21; // rax
  __int64 v22; // rax
  bool v23; // zf
  __int64 v24; // rdx
  __int64 v25; // rdx
  __int64 v26; // rdx
  __int64 *v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 *v30; // rax
  int v31; // [rsp+14h] [rbp-ACh]
  __int64 *v33; // [rsp+20h] [rbp-A0h]
  __int64 v34; // [rsp+28h] [rbp-98h]
  __int64 *v35; // [rsp+30h] [rbp-90h]
  char *v37; // [rsp+40h] [rbp-80h]
  __int64 v38; // [rsp+48h] [rbp-78h]
  __int64 *v39; // [rsp+50h] [rbp-70h] BYREF
  __int64 v40; // [rsp+58h] [rbp-68h]
  _BYTE v41[96]; // [rsp+60h] [rbp-60h] BYREF

  v4 = *(_QWORD *)(a2 + 16);
  v39 = (__int64 *)v41;
  v40 = 0x600000000LL;
  if ( !v4 )
    goto LABEL_62;
  v5 = 0x8000000000041LL;
  do
  {
    while ( 1 )
    {
      v6 = *(unsigned __int8 **)(v4 + 24);
      v7 = *v6;
      if ( (unsigned __int8)v7 > 0x1Cu )
      {
        v8 = (unsigned int)(v7 - 34);
        if ( (unsigned __int8)v8 <= 0x33u )
        {
          if ( _bittest64(&v5, v8) )
          {
            v9 = *((_QWORD *)v6 - 4);
            if ( v9 )
            {
              if ( !*(_BYTE *)v9
                && *(_QWORD *)(v9 + 24) == *((_QWORD *)v6 + 10)
                && a2 == v9
                && (unsigned __int8)sub_2A64220(*(__int64 **)a1, *((_QWORD *)v6 + 5)) )
              {
                break;
              }
            }
          }
        }
      }
      v4 = *(_QWORD *)(v4 + 8);
      if ( !v4 )
        goto LABEL_15;
    }
    v12 = (unsigned int)v40;
    v13 = (unsigned int)v40 + 1LL;
    if ( v13 > HIDWORD(v40) )
    {
      sub_C8D5F0((__int64)&v39, v41, v13, 8u, v10, v11);
      v12 = (unsigned int)v40;
    }
    v39[v12] = (__int64)v6;
    LODWORD(v40) = v40 + 1;
    v4 = *(_QWORD *)(v4 + 8);
  }
  while ( v4 );
LABEL_15:
  v31 = v40;
  v33 = &v39[(unsigned int)v40];
  if ( v33 == v39 )
    goto LABEL_42;
  v35 = v39;
  do
  {
    v14 = *v35;
    v34 = sub_B43CB0(*v35);
    if ( a3 == a4 )
    {
LABEL_60:
      if ( a2 == v34 )
        goto LABEL_40;
      goto LABEL_41;
    }
    v38 = 0;
    v15 = a3;
    do
    {
      if ( !*(_QWORD *)(v15 + 8) || v38 && *(_DWORD *)(v15 + 104) <= *(_DWORD *)(v38 + 104) )
        goto LABEL_32;
      v16 = *(char **)(v15 + 24);
      v17 = 16LL * *(unsigned int *)(v15 + 32);
      v37 = &v16[v17];
      v18 = v17 >> 4;
      v19 = v17 >> 6;
      if ( v19 )
      {
        v20 = &v16[64 * v19];
        while ( *((_QWORD *)v16 + 1) == sub_3176FF0(
                                          (__int64 **)a1,
                                          *(_BYTE **)(v14
                                                    + 32
                                                    * (*(unsigned int *)(*(_QWORD *)v16 + 32LL)
                                                     - (unsigned __int64)(*(_DWORD *)(v14 + 4) & 0x7FFFFFF)))) )
        {
          if ( *((_QWORD *)v16 + 3) != sub_3176FF0(
                                         (__int64 **)a1,
                                         *(_BYTE **)(v14
                                                   + 32
                                                   * (*(unsigned int *)(*((_QWORD *)v16 + 2) + 32LL)
                                                    - (unsigned __int64)(*(_DWORD *)(v14 + 4) & 0x7FFFFFF)))) )
          {
            v16 += 16;
            goto LABEL_29;
          }
          if ( *((_QWORD *)v16 + 5) != sub_3176FF0(
                                         (__int64 **)a1,
                                         *(_BYTE **)(v14
                                                   + 32
                                                   * (*(unsigned int *)(*((_QWORD *)v16 + 4) + 32LL)
                                                    - (unsigned __int64)(*(_DWORD *)(v14 + 4) & 0x7FFFFFF)))) )
          {
            v16 += 32;
            goto LABEL_29;
          }
          if ( *((_QWORD *)v16 + 7) != sub_3176FF0(
                                         (__int64 **)a1,
                                         *(_BYTE **)(v14
                                                   + 32
                                                   * (*(unsigned int *)(*((_QWORD *)v16 + 6) + 32LL)
                                                    - (unsigned __int64)(*(_DWORD *)(v14 + 4) & 0x7FFFFFF)))) )
          {
            v16 += 48;
            goto LABEL_29;
          }
          v16 += 64;
          if ( v16 == v20 )
          {
            v18 = (v37 - v16) >> 4;
            goto LABEL_47;
          }
        }
        goto LABEL_29;
      }
LABEL_47:
      if ( v18 == 2 )
        goto LABEL_56;
      if ( v18 != 3 )
      {
        if ( v18 != 1 )
          goto LABEL_50;
LABEL_58:
        if ( *((_QWORD *)v16 + 1) == sub_3176FF0(
                                       (__int64 **)a1,
                                       *(_BYTE **)(v14
                                                 + 32
                                                 * (*(unsigned int *)(*(_QWORD *)v16 + 32LL)
                                                  - (unsigned __int64)(*(_DWORD *)(v14 + 4) & 0x7FFFFFF)))) )
        {
LABEL_50:
          v38 = v15;
          goto LABEL_32;
        }
        goto LABEL_29;
      }
      if ( *((_QWORD *)v16 + 1) == sub_3176FF0(
                                     (__int64 **)a1,
                                     *(_BYTE **)(v14
                                               + 32
                                               * (*(unsigned int *)(*(_QWORD *)v16 + 32LL)
                                                - (unsigned __int64)(*(_DWORD *)(v14 + 4) & 0x7FFFFFF)))) )
      {
        v16 += 16;
LABEL_56:
        if ( *((_QWORD *)v16 + 1) == sub_3176FF0(
                                       (__int64 **)a1,
                                       *(_BYTE **)(v14
                                                 + 32
                                                 * (*(unsigned int *)(*(_QWORD *)v16 + 32LL)
                                                  - (unsigned __int64)(*(_DWORD *)(v14 + 4) & 0x7FFFFFF)))) )
        {
          v16 += 16;
          goto LABEL_58;
        }
      }
LABEL_29:
      v21 = v38;
      if ( v37 == v16 )
        v21 = v15;
      v38 = v21;
LABEL_32:
      v15 += 176;
    }
    while ( a4 != v15 );
    if ( !v38 )
      goto LABEL_60;
    v22 = *(_QWORD *)(v38 + 8);
    v23 = *(_QWORD *)(v14 - 32) == 0;
    *(_QWORD *)(v14 + 80) = *(_QWORD *)(v22 + 24);
    if ( !v23 )
    {
      v24 = *(_QWORD *)(v14 - 24);
      **(_QWORD **)(v14 - 16) = v24;
      if ( v24 )
        *(_QWORD *)(v24 + 16) = *(_QWORD *)(v14 - 16);
    }
    *(_QWORD *)(v14 - 32) = v22;
    v25 = *(_QWORD *)(v22 + 16);
    *(_QWORD *)(v14 - 24) = v25;
    if ( v25 )
      *(_QWORD *)(v25 + 16) = v14 - 24;
    *(_QWORD *)(v14 - 16) = v22 + 16;
    *(_QWORD *)(v22 + 16) = v14 - 32;
LABEL_40:
    --v31;
LABEL_41:
    ++v35;
  }
  while ( v33 != v35 );
LABEL_42:
  if ( v31 )
    goto LABEL_43;
LABEL_62:
  if ( (unsigned __int8)sub_2A641A0(*(__int64 **)a1, a2) )
  {
    sub_2A65590(*(__int64 **)a1, a2);
    if ( *(_BYTE *)(a1 + 468) )
    {
      v30 = *(__int64 **)(a1 + 448);
      v26 = *(unsigned int *)(a1 + 460);
      v27 = &v30[v26];
      if ( v30 != v27 )
      {
        while ( a2 != *v30 )
        {
          if ( v27 == ++v30 )
            goto LABEL_67;
        }
        goto LABEL_43;
      }
LABEL_67:
      if ( (unsigned int)v26 < *(_DWORD *)(a1 + 456) )
      {
        *(_DWORD *)(a1 + 460) = v26 + 1;
        *v27 = a2;
        ++*(_QWORD *)(a1 + 440);
        goto LABEL_43;
      }
    }
    sub_C8CC70(a1 + 440, a2, v26, (__int64)v27, v28, v29);
  }
LABEL_43:
  if ( v39 != (__int64 *)v41 )
    _libc_free((unsigned __int64)v39);
}
