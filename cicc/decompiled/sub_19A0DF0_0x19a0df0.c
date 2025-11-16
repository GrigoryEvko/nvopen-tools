// Function: sub_19A0DF0
// Address: 0x19a0df0
//
int __fastcall sub_19A0DF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  __int64 v6; // rax
  __int64 v7; // rdi
  __int64 v8; // rsi
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // rcx
  __int64 v11; // r14
  __int64 v12; // r9
  __int64 v13; // r11
  __int64 v14; // r15
  __int64 v15; // r9
  __int64 v16; // rcx
  __int64 v17; // r13
  __int64 v18; // rbx
  __int64 v19; // r12
  __int64 v20; // r14
  __int64 v21; // r12
  __int64 v22; // rbx
  __int64 v23; // r13
  size_t v24; // r15
  bool v25; // zf
  __int64 v26; // r9
  __int64 v27; // rdx
  _QWORD *v28; // rax
  _QWORD *v29; // r13
  __int64 v30; // r14
  _QWORD *v31; // r15
  unsigned int v32; // eax
  _QWORD *v33; // rbx
  __int64 v34; // rax
  __int64 v35; // r13
  __int64 v36; // rbx
  __int64 *v37; // r14
  char v38; // al
  __int64 v39; // rcx
  unsigned int v40; // eax
  __int64 v41; // rdx
  __int64 v42; // rsi
  __int64 v43; // rax
  __int64 v44; // rbx
  unsigned __int64 v45; // rdi
  __int64 v46; // r13
  unsigned __int64 v47; // r12
  unsigned __int64 v48; // rdi
  __int64 v49; // r13
  unsigned __int64 v50; // r12
  unsigned __int64 v51; // rdi
  __int64 v52; // rax
  unsigned __int64 *v53; // r13
  unsigned __int64 *v54; // r12
  int v55; // r9d
  unsigned __int64 v56; // r13
  _QWORD *v57; // rdx
  _QWORD *v58; // r14
  _QWORD *v59; // rax
  _QWORD *v60; // r14
  unsigned __int64 v61; // rsi
  _QWORD *v62; // r12
  unsigned __int64 *v63; // rdi
  unsigned __int64 v64; // rcx
  __int64 v65; // r10
  __int64 v66; // rdx
  unsigned __int64 v67; // rcx
  unsigned int v68; // esi
  __int64 v69; // rcx
  unsigned __int64 v71; // [rsp-10h] [rbp-90h]
  __int64 v72; // [rsp-8h] [rbp-88h]
  __int64 v73; // [rsp+8h] [rbp-78h]
  __int64 v74; // [rsp+10h] [rbp-70h]
  __int64 v75; // [rsp+18h] [rbp-68h]
  __int64 v76; // [rsp+20h] [rbp-60h]
  __int64 v77; // [rsp+28h] [rbp-58h]
  __int64 v78; // [rsp+28h] [rbp-58h]
  __int64 v79; // [rsp+28h] [rbp-58h]
  __int64 v80; // [rsp+28h] [rbp-58h]
  __int64 v81; // [rsp+30h] [rbp-50h]
  __int64 v82; // [rsp+38h] [rbp-48h]
  __int64 v83; // [rsp+38h] [rbp-48h]
  char v84; // [rsp+38h] [rbp-48h]
  __int64 v85; // [rsp+38h] [rbp-48h]
  __int64 v86; // [rsp+40h] [rbp-40h]
  char v87; // [rsp+40h] [rbp-40h]
  unsigned int v88; // [rsp+40h] [rbp-40h]
  unsigned __int64 v89; // [rsp+48h] [rbp-38h]

  v5 = a1;
  v6 = *(unsigned int *)(a1 + 376);
  v7 = *(_QWORD *)(a1 + 368);
  v81 = v6;
  v8 = v7 + 1984 * v6;
  if ( v7 != v8 )
  {
    v6 = v7;
    v9 = 1;
    while ( 1 )
    {
      v10 = *(unsigned int *)(v6 + 752);
      if ( v10 > 0xFFFE )
        break;
      v9 *= v10;
      if ( v9 > 0x3FFFB )
        break;
      v6 += 1984;
      if ( v8 == v6 )
      {
        if ( v9 <= 0xFFFE )
          return v6;
        break;
      }
    }
    if ( v81 )
    {
      v89 = 0;
      while ( 1 )
      {
        v11 = v7 + 1984 * v89;
        v12 = *(_QWORD *)(v11 + 744);
        v6 = 96LL * *(unsigned int *)(v11 + 752);
        v13 = v12 + v6;
        if ( v12 + v6 == v12 )
        {
          ++v89;
LABEL_81:
          v9 = v89;
          if ( v81 == v89 )
            return v6;
        }
        else
        {
          v6 = v5;
          v14 = *(_QWORD *)(v11 + 744);
          v15 = v6;
          while ( 1 )
          {
            if ( *(_QWORD *)(v14 + 8) )
            {
              if ( *(_QWORD *)(v14 + 24) <= 1u )
              {
                v16 = *(unsigned int *)(v15 + 376);
                if ( *(_DWORD *)(v15 + 376) )
                {
                  v17 = 0;
                  v18 = 0;
                  while ( 1 )
                  {
                    v19 = v17 + *(_QWORD *)(v15 + 368);
                    if ( v11 != v19 )
                    {
                      LODWORD(v6) = *(_DWORD *)(v19 + 32);
                      if ( (_DWORD)v6 != 3 && (_DWORD)v6 == *(_DWORD *)(v11 + 32) )
                      {
                        v6 = *(_QWORD *)(v11 + 40);
                        if ( *(_QWORD *)(v19 + 40) == v6 )
                        {
                          LODWORD(v6) = *(_DWORD *)(v11 + 48);
                          if ( *(_DWORD *)(v19 + 48) == (_DWORD)v6 )
                          {
                            v6 = *(_QWORD *)(v11 + 736);
                            if ( *(_QWORD *)(v19 + 736) == v6 )
                            {
                              v77 = v15;
                              v82 = v16;
                              v86 = v13;
                              LODWORD(v6) = sub_19A07E0(v17 + *(_QWORD *)(v15 + 368), v14, v9, v16, a5, v15);
                              v13 = v86;
                              v16 = v82;
                              v15 = v77;
                              if ( (_BYTE)v6 )
                              {
                                a5 = *(_QWORD *)(v19 + 744);
                                v6 = a5 + 96LL * *(unsigned int *)(v19 + 752);
                                if ( a5 != v6 )
                                  break;
                              }
                            }
                          }
                        }
                      }
                    }
LABEL_15:
                    ++v18;
                    v17 += 1984;
                    if ( v16 == v18 )
                      goto LABEL_10;
                  }
                  v83 = v11;
                  v20 = *(_QWORD *)(v19 + 744);
                  v73 = v19;
                  v9 = 8LL * *(unsigned int *)(v14 + 40);
                  v76 = v18;
                  v21 = *(unsigned int *)(v14 + 40);
                  v22 = v6;
                  v75 = v17;
                  v23 = v14;
                  v24 = v9;
                  v78 = v16;
                  v74 = v15;
                  while ( 1 )
                  {
                    v6 = *(unsigned int *)(v20 + 40);
                    if ( v6 == v21 )
                    {
                      if ( !v24
                        || (LODWORD(v6) = memcmp(*(const void **)(v20 + 32), *(const void **)(v23 + 32), v24),
                            !(_DWORD)v6) )
                      {
                        v6 = *(_QWORD *)(v23 + 80);
                        if ( *(_QWORD *)(v20 + 80) == v6 )
                        {
                          v6 = *(_QWORD *)v23;
                          if ( *(_QWORD *)v20 == *(_QWORD *)v23 )
                          {
                            v6 = *(_QWORD *)(v23 + 24);
                            if ( *(_QWORD *)(v20 + 24) == v6 )
                            {
                              v6 = *(_QWORD *)(v23 + 88);
                              if ( *(_QWORD *)(v20 + 88) == v6 )
                                break;
                            }
                          }
                        }
                      }
                    }
                    v20 += 96;
                    if ( v22 == v20 )
                    {
                      v14 = v23;
                      v13 = v86;
                      v11 = v83;
                      v16 = v78;
                      v18 = v76;
                      v17 = v75;
                      v15 = v74;
                      goto LABEL_15;
                    }
                  }
                  LODWORD(a5) = v20;
                  v14 = v23;
                  v13 = v86;
                  v16 = v78;
                  v25 = *(_QWORD *)(v20 + 8) == 0;
                  v18 = v76;
                  v17 = v75;
                  v11 = v83;
                  v15 = v74;
                  if ( !v25 )
                    goto LABEL_15;
                  LODWORD(v6) = sub_19952F0(
                                  v74,
                                  v73,
                                  *(_QWORD *)(v14 + 8),
                                  0,
                                  *(_DWORD *)(v83 + 32),
                                  v74,
                                  *(_QWORD ***)(v83 + 40),
                                  *(_QWORD *)(v83 + 48));
                  v9 = v71;
                  v15 = v74;
                  v13 = v86;
                  if ( (_BYTE)v6 )
                    break;
                }
              }
            }
LABEL_10:
            v14 += 96;
            if ( v13 == v14 )
            {
              ++v89;
              v5 = v15;
              goto LABEL_81;
            }
          }
          v84 = v6;
          v26 = v14;
          v5 = v74;
          *(_BYTE *)(v73 + 728) &= *(_BYTE *)(v11 + 728);
          v27 = *(unsigned int *)(v11 + 64);
          v28 = *(_QWORD **)(v11 + 56);
          if ( &v28[10 * v27] != v28 )
          {
            v79 = v11;
            v29 = &v28[10 * v27];
            v30 = v26;
            v31 = v28;
            do
            {
              v31[9] += *(_QWORD *)(v30 + 8);
              v32 = *(_DWORD *)(v73 + 64);
              if ( v32 >= *(_DWORD *)(v73 + 68) )
              {
                sub_19957D0((unsigned __int64 *)(v73 + 56), 0);
                v32 = *(_DWORD *)(v73 + 64);
              }
              v33 = (_QWORD *)(*(_QWORD *)(v73 + 56) + 80LL * v32);
              if ( v33 )
              {
                *v33 = *v31;
                v33[1] = v31[1];
                sub_16CCCB0(v33 + 2, (__int64)(v33 + 7), (__int64)(v31 + 2));
                v33[9] = v31[9];
                v32 = *(_DWORD *)(v73 + 64);
              }
              *(_DWORD *)(v73 + 64) = v32 + 1;
              v34 = v31[9];
              if ( v34 > *(_QWORD *)(v73 + 720) )
                *(_QWORD *)(v73 + 720) = v34;
              if ( v34 < *(_QWORD *)(v73 + 712) )
                *(_QWORD *)(v73 + 712) = v34;
              v31 += 10;
            }
            while ( v29 != v31 );
            v11 = v79;
            v5 = v74;
          }
          v35 = *(unsigned int *)(v73 + 752);
          if ( *(_DWORD *)(v73 + 752) )
          {
            v87 = 0;
            v36 = 0;
            v80 = v11;
            do
            {
              while ( 1 )
              {
                v37 = (__int64 *)(*(_QWORD *)(v73 + 744) + 96 * v36);
                v38 = sub_1995490(
                        *(__int64 **)(v5 + 32),
                        *(_QWORD *)(v73 + 712),
                        *(_QWORD *)(v73 + 720),
                        *(_DWORD *)(v73 + 32),
                        *(_QWORD *)(v73 + 40),
                        *(_DWORD *)(v73 + 48),
                        (__int64)v37);
                a5 = v72;
                if ( !v38 )
                  break;
                if ( v35 == ++v36 )
                  goto LABEL_52;
              }
              --v35;
              sub_1994A60(v73, v37);
              v87 = v84;
            }
            while ( v35 != v36 );
LABEL_52:
            v11 = v80;
            if ( v87 )
              sub_1996C50(v73, -1108378657 * ((v73 - *(_QWORD *)(v5 + 368)) >> 6), v5 + 32128);
          }
          v39 = *(_QWORD *)(v5 + 368);
          v40 = *(_DWORD *)(v5 + 376);
          v41 = 1984LL * v40;
          v42 = v39 + v41 - 1984;
          if ( v11 != v42 )
          {
            sub_1997A20(v11, v42, v41, v39, a5, v26);
            v40 = *(_DWORD *)(v5 + 376);
            v39 = *(_QWORD *)(v5 + 368);
          }
          v43 = v40 - 1;
          *(_DWORD *)(v5 + 376) = v43;
          v44 = v39 + 1984 * v43;
          v45 = *(_QWORD *)(v44 + 1928);
          if ( v45 != *(_QWORD *)(v44 + 1920) )
            _libc_free(v45);
          v46 = *(_QWORD *)(v44 + 744);
          v47 = v46 + 96LL * *(unsigned int *)(v44 + 752);
          if ( v46 != v47 )
          {
            do
            {
              v47 -= 96LL;
              v48 = *(_QWORD *)(v47 + 32);
              if ( v48 != v47 + 48 )
                _libc_free(v48);
            }
            while ( v46 != v47 );
            v47 = *(_QWORD *)(v44 + 744);
          }
          if ( v47 != v44 + 760 )
            _libc_free(v47);
          v49 = *(_QWORD *)(v44 + 56);
          v50 = v49 + 80LL * *(unsigned int *)(v44 + 64);
          if ( v49 != v50 )
          {
            do
            {
              v50 -= 80LL;
              v51 = *(_QWORD *)(v50 + 32);
              if ( v51 != *(_QWORD *)(v50 + 24) )
                _libc_free(v51);
            }
            while ( v49 != v50 );
            v50 = *(_QWORD *)(v44 + 56);
          }
          if ( v50 != v44 + 72 )
            _libc_free(v50);
          v52 = *(unsigned int *)(v44 + 24);
          if ( (_DWORD)v52 )
          {
            v53 = *(unsigned __int64 **)(v44 + 8);
            v54 = &v53[6 * v52];
            do
            {
              if ( (unsigned __int64 *)*v53 != v53 + 2 )
                _libc_free(*v53);
              v53 += 6;
            }
            while ( v54 != v53 );
          }
          LODWORD(v6) = j___libc_free_0(*(_QWORD *)(v44 + 8));
          v56 = *(unsigned int *)(v5 + 376);
          v88 = *(_DWORD *)(v5 + 376);
          if ( *(_DWORD *)(v5 + 32144) )
          {
            v57 = *(_QWORD **)(v5 + 32136);
            a5 = 16LL * *(unsigned int *)(v5 + 32152);
            v58 = (_QWORD *)((char *)v57 + a5);
            if ( v57 != (_QWORD *)((char *)v57 + a5) )
            {
              while ( 1 )
              {
                v6 = *v57;
                if ( *v57 != -16 && v6 != -8 )
                  break;
                v57 += 2;
                if ( v58 == v57 )
                  goto LABEL_78;
              }
              if ( v57 != v58 )
              {
                v59 = v58;
                v60 = v57;
                v61 = v57[1];
                v62 = v59;
                v63 = v57 + 1;
                v85 = 1LL << v89;
                if ( (v61 & 1) != 0 )
                {
LABEL_94:
                  v64 = v61 >> 58;
                  if ( v61 >> 58 <= v89 )
                    goto LABEL_100;
                  if ( v56 < v64 && (v65 = ~(-1LL << v64), v66 = v65 & (v61 >> 1), _bittest64(&v66, v88)) )
                  {
                    v67 = 2 * ((v64 << 57) | v65 & (v85 | v66)) + 1;
                    v60[1] = v67;
                  }
                  else
                  {
                    v67 = 2 * ((v61 >> 58 << 57) | ~(-1LL << (v61 >> 58)) & ~v85 & (v61 >> 1)) + 1;
                    v60[1] = v67;
                  }
LABEL_98:
                  if ( (v67 & 1) != 0 )
                    goto LABEL_99;
                }
                else
                {
                  while ( 1 )
                  {
                    v64 = *(unsigned int *)(v61 + 16);
                    if ( v64 > v89 )
                      break;
LABEL_100:
                    v68 = v64;
                    if ( v56 <= v64 )
                      v68 = v56;
                    v60 += 2;
                    LODWORD(v6) = sub_13A5100(v63, v68, 0, v64, a5, v55);
                    if ( v60 == v62 )
                      goto LABEL_78;
                    while ( 1 )
                    {
                      v6 = *v60;
                      if ( *v60 != -16 && v6 != -8 )
                        break;
                      v60 += 2;
                      if ( v62 == v60 )
                        goto LABEL_78;
                    }
                    if ( v60 == v62 )
                      goto LABEL_78;
                    v61 = v60[1];
                    v63 = v60 + 1;
                    if ( (v61 & 1) != 0 )
                      goto LABEL_94;
                  }
                  if ( v56 < v64 )
                  {
                    v69 = *(_QWORD *)(*(_QWORD *)v61 + 8LL * (v88 >> 6));
                    if ( _bittest64(&v69, v88) )
                    {
                      *(_QWORD *)(*(_QWORD *)v61 + 8LL * ((unsigned int)v89 >> 6)) |= 1LL << v89;
                      v67 = v60[1];
                      goto LABEL_98;
                    }
                  }
                  *(_QWORD *)(8LL * ((unsigned int)v89 >> 6) + *(_QWORD *)v61) &= ~(1LL << v89);
                  v67 = v60[1];
                  if ( (v67 & 1) != 0 )
                  {
LABEL_99:
                    v64 = v67 >> 58;
                    goto LABEL_100;
                  }
                }
                v64 = *(unsigned int *)(v67 + 16);
                goto LABEL_100;
              }
            }
          }
LABEL_78:
          --v81;
          v9 = v89;
          if ( v81 == v89 )
            return v6;
        }
        v7 = *(_QWORD *)(v5 + 368);
      }
    }
  }
  return v6;
}
