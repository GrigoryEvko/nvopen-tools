// Function: sub_2A486D0
// Address: 0x2a486d0
//
unsigned __int64 *__fastcall sub_2A486D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  unsigned __int64 *result; // rax
  __int64 v8; // rcx
  __int64 v9; // rax
  __int64 *v10; // rdx
  unsigned int v11; // eax
  __int64 v12; // r12
  __int64 *v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rbx
  __int64 *v16; // rdi
  __int64 *v17; // r13
  __int64 *v18; // rbx
  __int64 v19; // r12
  __int64 v20; // rax
  __int64 v21; // r8
  __int64 v22; // r9
  unsigned int v23; // esi
  __int64 *v24; // r10
  int v25; // r11d
  unsigned int i; // ecx
  __int64 *v27; // rdx
  __int64 v28; // rdi
  unsigned int v29; // ecx
  __int64 v30; // rdi
  _BYTE *v31; // rdi
  __int64 v32; // r13
  __int64 v33; // rax
  unsigned __int64 v34; // rdx
  __int64 v35; // rax
  _BYTE *v36; // rdi
  int v37; // esi
  int v38; // esi
  int v39; // edi
  __int64 *v40; // rax
  unsigned int k; // ecx
  unsigned int v42; // ecx
  int v43; // edi
  int v44; // ecx
  int v45; // esi
  int v46; // esi
  int v47; // edi
  unsigned int j; // ecx
  unsigned int v49; // ecx
  __int64 v50; // [rsp+10h] [rbp-150h]
  unsigned __int64 v52; // [rsp+20h] [rbp-140h]
  __int64 v54; // [rsp+48h] [rbp-118h]
  unsigned __int64 *v55; // [rsp+50h] [rbp-110h]
  int v56; // [rsp+60h] [rbp-100h]
  __int64 v58; // [rsp+70h] [rbp-F0h]
  _QWORD v60[2]; // [rsp+80h] [rbp-E0h] BYREF
  __int64 *v61; // [rsp+90h] [rbp-D0h] BYREF
  __int64 v62; // [rsp+98h] [rbp-C8h]
  _QWORD v63[4]; // [rsp+A0h] [rbp-C0h] BYREF
  __int64 *v64; // [rsp+C0h] [rbp-A0h] BYREF
  __int64 v65; // [rsp+C8h] [rbp-98h]
  __int64 v66; // [rsp+D0h] [rbp-90h] BYREF
  char v67; // [rsp+D8h] [rbp-88h] BYREF
  __int64 v68; // [rsp+F0h] [rbp-70h] BYREF
  __int64 *v69; // [rsp+F8h] [rbp-68h]
  __int64 v70; // [rsp+100h] [rbp-60h]
  int v71; // [rsp+108h] [rbp-58h]
  unsigned __int8 v72; // [rsp+10Ch] [rbp-54h]
  char v73; // [rsp+110h] [rbp-50h] BYREF

  v6 = *(_QWORD *)(a2 - 32);
  v60[1] = *(_QWORD *)(a2 - 64);
  v54 = v6;
  v52 = (unsigned __int64)(((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)) << 32;
  v55 = v60;
  v60[0] = v6;
  v50 = a1 + 1616;
  if ( a3 == v6 )
    goto LABEL_2;
LABEL_4:
  v68 = 0;
  v69 = (__int64 *)&v73;
  v8 = 1;
  v9 = *(_QWORD *)(a2 - 96);
  v61 = v63;
  v63[0] = v9;
  v62 = 0x400000001LL;
  v70 = 4;
  v71 = 0;
  v72 = 1;
  v10 = v63;
  v56 = ((0xBF58476D1CE4E5B9LL * (v52 | ((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4))) >> 31)
      ^ (484763065 * (v52 | ((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4)));
  v11 = 1;
  while ( 1 )
  {
    v12 = v10[v11 - 1];
    LODWORD(v62) = v11 - 1;
    if ( (_BYTE)v8 )
    {
      v13 = v69;
      v10 = &v69[HIDWORD(v70)];
      if ( v69 != v10 )
      {
        while ( v12 != *v13 )
        {
          if ( v10 == ++v13 )
            goto LABEL_47;
        }
        goto LABEL_10;
      }
LABEL_47:
      if ( HIDWORD(v70) < (unsigned int)v70 )
        break;
    }
    sub_C8CC70((__int64)&v68, v12, (__int64)v10, v8, a5, a6);
    v8 = v72;
    if ( (_BYTE)v10 )
      goto LABEL_13;
LABEL_10:
    v11 = v62;
    if ( !(_DWORD)v62 )
      goto LABEL_43;
LABEL_11:
    v10 = v61;
  }
  ++HIDWORD(v70);
  *v10 = v12;
  v8 = v72;
  ++v68;
LABEL_13:
  if ( (unsigned int)(HIDWORD(v70) - v71) > 8 )
    goto LABEL_43;
  if ( v54 == v6 )
  {
    if ( *(_BYTE *)v12 <= 0x1Cu )
      goto LABEL_22;
    v30 = *(_QWORD *)(v12 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v30 + 8) - 17 <= 1 )
      v30 = **(_QWORD **)(v30 + 16);
    if ( !sub_BCAC40(v30, 1) )
      goto LABEL_22;
    if ( *(_BYTE *)v12 != 57 )
    {
      if ( *(_BYTE *)v12 != 86 )
        goto LABEL_22;
      v15 = *(_QWORD *)(v12 - 96);
      if ( *(_QWORD *)(v15 + 8) != *(_QWORD *)(v12 + 8) )
        goto LABEL_22;
      v31 = *(_BYTE **)(v12 - 32);
      if ( *v31 > 0x15u )
        goto LABEL_22;
      v32 = *(_QWORD *)(v12 - 64);
      if ( !sub_AC30F0((__int64)v31) )
        goto LABEL_22;
LABEL_73:
      if ( !v32 )
        goto LABEL_22;
LABEL_64:
      v33 = (unsigned int)v62;
      v34 = (unsigned int)v62 + 1LL;
      if ( v34 > HIDWORD(v62) )
      {
        sub_C8D5F0((__int64)&v61, v63, v34, 8u, a5, a6);
        v33 = (unsigned int)v62;
      }
      v61[v33] = v32;
      v8 = HIDWORD(v62);
      LODWORD(v62) = v62 + 1;
      v35 = (unsigned int)v62;
      if ( (unsigned __int64)(unsigned int)v62 + 1 > HIDWORD(v62) )
      {
        sub_C8D5F0((__int64)&v61, v63, (unsigned int)v62 + 1LL, 8u, a5, a6);
        v35 = (unsigned int)v62;
      }
      v10 = v61;
      v61[v35] = v15;
      LODWORD(v62) = v62 + 1;
      goto LABEL_22;
    }
LABEL_60:
    if ( (*(_BYTE *)(v12 + 7) & 0x40) != 0 )
      v10 = *(__int64 **)(v12 - 8);
    else
      v10 = (__int64 *)(v12 - 32LL * (*(_DWORD *)(v12 + 4) & 0x7FFFFFF));
    v15 = *v10;
    if ( !*v10 )
      goto LABEL_22;
    v32 = v10[4];
    if ( !v32 )
      goto LABEL_22;
    goto LABEL_64;
  }
  if ( *(_BYTE *)v12 <= 0x1Cu )
    goto LABEL_22;
  v14 = *(_QWORD *)(v12 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v14 + 8) - 17 <= 1 )
    v14 = **(_QWORD **)(v14 + 16);
  if ( !sub_BCAC40(v14, 1) )
    goto LABEL_22;
  if ( *(_BYTE *)v12 == 58 )
    goto LABEL_60;
  if ( *(_BYTE *)v12 == 86 )
  {
    v15 = *(_QWORD *)(v12 - 96);
    if ( *(_QWORD *)(v15 + 8) == *(_QWORD *)(v12 + 8) )
    {
      v36 = *(_BYTE **)(v12 - 64);
      if ( *v36 <= 0x15u )
      {
        v32 = *(_QWORD *)(v12 - 32);
        if ( sub_AD7A80(v36, 1, (__int64)v10, v8, a5) )
          goto LABEL_73;
      }
    }
  }
LABEL_22:
  v66 = v12;
  v64 = &v66;
  v65 = 0x400000001LL;
  if ( (unsigned __int8)(*(_BYTE *)v12 - 82) > 1u )
  {
    v16 = &v66;
    v17 = (__int64 *)&v67;
    goto LABEL_24;
  }
  sub_2A45340(v12, (__int64)&v64, (__int64)v10, v8, a5, a6);
  v16 = v64;
  v17 = &v64[(unsigned int)v65];
  if ( v17 != v64 )
  {
LABEL_24:
    v58 = v12;
    v18 = v16;
    do
    {
      v19 = *v18;
      if ( sub_2A45310(*v18) )
      {
        v20 = sub_22077B0(0x50u);
        if ( v20 )
        {
          *(_QWORD *)(v20 + 8) = 0;
          *(_QWORD *)(v20 + 16) = 0;
          *(_DWORD *)(v20 + 24) = 0;
          *(_QWORD *)(v20 + 32) = v19;
          *(_QWORD *)(v20 + 64) = v6;
          *(_QWORD *)(v20 + 48) = v58;
          *(_QWORD *)(v20 + 56) = a3;
          *(_QWORD *)v20 = &unk_4A22E00;
          *(_BYTE *)(v20 + 72) = v54 == v6;
        }
        sub_2A481C0(a1, a4, v19, v20, v21, v22);
        if ( !sub_AA54C0(v6) )
        {
          v23 = *(_DWORD *)(a1 + 1640);
          if ( v23 )
          {
            a5 = v23 - 1;
            v24 = 0;
            v25 = 1;
            a6 = *(_QWORD *)(a1 + 1624);
            for ( i = a5 & v56; ; i = a5 & v29 )
            {
              v27 = (__int64 *)(a6 + 16LL * i);
              v28 = *v27;
              if ( a3 == *v27 && v27[1] == v6 )
                goto LABEL_25;
              if ( v28 == -4096 )
              {
                if ( v27[1] == -4096 )
                {
                  v43 = *(_DWORD *)(a1 + 1632);
                  if ( v24 )
                    v27 = v24;
                  ++*(_QWORD *)(a1 + 1616);
                  v44 = v43 + 1;
                  if ( 4 * (v43 + 1) < 3 * v23 )
                  {
                    a5 = v23 >> 3;
                    if ( v23 - *(_DWORD *)(a1 + 1636) - v44 > (unsigned int)a5 )
                    {
LABEL_92:
                      *(_DWORD *)(a1 + 1632) = v44;
                      if ( *v27 != -4096 || v27[1] != -4096 )
                        --*(_DWORD *)(a1 + 1636);
                      v27[1] = v6;
                      *v27 = a3;
                      goto LABEL_25;
                    }
                    sub_2884B10(v50, v23);
                    v45 = *(_DWORD *)(a1 + 1640);
                    if ( v45 )
                    {
                      v46 = v45 - 1;
                      a6 = *(_QWORD *)(a1 + 1624);
                      v47 = 1;
                      v40 = 0;
                      for ( j = v46 & v56; ; j = v46 & v49 )
                      {
                        v27 = (__int64 *)(a6 + 16LL * j);
                        a5 = *v27;
                        if ( a3 == *v27 && v27[1] == v6 )
                          break;
                        if ( a5 == -4096 )
                        {
                          if ( v27[1] == -4096 )
                            goto LABEL_111;
                        }
                        else if ( a5 == -8192 && v27[1] == -8192 && !v40 )
                        {
                          v40 = (__int64 *)(a6 + 16LL * j);
                        }
                        v49 = v47 + j;
                        ++v47;
                      }
LABEL_107:
                      v44 = *(_DWORD *)(a1 + 1632) + 1;
                      goto LABEL_92;
                    }
                    goto LABEL_116;
                  }
LABEL_78:
                  sub_2884B10(v50, 2 * v23);
                  v37 = *(_DWORD *)(a1 + 1640);
                  if ( v37 )
                  {
                    v38 = v37 - 1;
                    a6 = *(_QWORD *)(a1 + 1624);
                    v39 = 1;
                    v40 = 0;
                    for ( k = v38 & v56; ; k = v38 & v42 )
                    {
                      v27 = (__int64 *)(a6 + 16LL * k);
                      a5 = *v27;
                      if ( a3 == *v27 && v27[1] == v6 )
                        break;
                      if ( a5 == -4096 )
                      {
                        if ( v27[1] == -4096 )
                        {
LABEL_111:
                          if ( v40 )
                            v27 = v40;
                          v44 = *(_DWORD *)(a1 + 1632) + 1;
                          goto LABEL_92;
                        }
                      }
                      else if ( a5 == -8192 && v27[1] == -8192 && !v40 )
                      {
                        v40 = (__int64 *)(a6 + 16LL * k);
                      }
                      v42 = v39 + k;
                      ++v39;
                    }
                    goto LABEL_107;
                  }
LABEL_116:
                  ++*(_DWORD *)(a1 + 1632);
                  BUG();
                }
              }
              else if ( v28 == -8192 && v27[1] == -8192 && !v24 )
              {
                v24 = (__int64 *)(a6 + 16LL * i);
              }
              v29 = v25 + i;
              ++v25;
            }
          }
          ++*(_QWORD *)(a1 + 1616);
          goto LABEL_78;
        }
      }
LABEL_25:
      ++v18;
    }
    while ( v17 != v18 );
    v16 = v64;
  }
  if ( v16 != &v66 )
    _libc_free((unsigned __int64)v16);
  v11 = v62;
  v8 = v72;
  if ( (_DWORD)v62 )
    goto LABEL_11;
LABEL_43:
  if ( !(_BYTE)v8 )
    _libc_free((unsigned __int64)v69);
  if ( v61 != v63 )
    _libc_free((unsigned __int64)v61);
LABEL_2:
  while ( 2 )
  {
    result = ++v55;
    if ( v55 != (unsigned __int64 *)&v61 )
    {
      v6 = *v55;
      if ( a3 == *v55 )
        continue;
      goto LABEL_4;
    }
    return result;
  }
}
