// Function: sub_24F3E60
// Address: 0x24f3e60
//
void __fastcall sub_24F3E60(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r13
  _QWORD *v6; // rbx
  _QWORD *i; // r12
  __int64 v9; // r14
  __int64 v10; // r13
  int v11; // eax
  __int64 v12; // rdx
  unsigned __int8 *v13; // r9
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // r14
  __int64 v17; // rcx
  __int64 v18; // rdx
  unsigned int v19; // edx
  __int64 v20; // rcx
  unsigned __int8 *v21; // rdi
  unsigned __int8 *v22; // rax
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 *v25; // rax
  __int64 *v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  bool v33; // al
  __int64 v34; // r8
  bool v35; // di
  __int64 v36; // rcx
  __int64 v37; // rax
  __int64 *v38; // rax
  __int64 *v39; // rdx
  __int64 v40; // rcx
  __int64 v41; // rax
  __int64 v42; // rdx
  __int64 *v43; // rax
  __int64 *v44; // rax
  __int64 v45; // rax
  __int64 v46; // rax
  unsigned __int64 v47; // rdx
  __int64 v48; // rax
  unsigned __int64 v49; // rcx
  bool v50; // al
  __int64 v51; // rax
  __int64 v52; // rax
  __int64 v53; // rax
  unsigned __int8 *v54; // rax
  __int64 v55; // rbx
  __int64 v56; // rcx
  __int64 v57; // r13
  __int64 v58; // rdx
  __int64 v59; // rax
  __int64 v60; // r12
  __int64 v61; // rax
  bool v62; // cc
  _QWORD *v63; // rax
  __int64 v64; // rax
  _QWORD *v65; // rax
  __int64 v66; // rax
  _QWORD *v67; // rax
  unsigned __int64 v68; // rdx
  unsigned __int64 v69; // rcx
  __int64 v70; // rbx
  unsigned __int8 *v71; // rax
  __int64 v72; // [rsp+10h] [rbp-60h]
  unsigned __int8 *v73; // [rsp+10h] [rbp-60h]
  const void *v74; // [rsp+18h] [rbp-58h]
  __int64 v75; // [rsp+20h] [rbp-50h]
  bool v78; // [rsp+3Eh] [rbp-32h]
  char v79; // [rsp+3Fh] [rbp-31h]

  v5 = a2 + 72;
  *a1 = 0;
  *((_DWORD *)a1 + 4) = 0;
  *((_DWORD *)a1 + 16) = 0;
  *((_DWORD *)a1 + 24) = 0;
  *((_DWORD *)a1 + 32) = 0;
  *((_DWORD *)a1 + 44) = 0;
  *((_DWORD *)a1 + 56) = 0;
  *((_DWORD *)a1 + 64) = 0;
  a1[36] = 0;
  a1[39] = 0;
  a1[40] = 0;
  v6 = *(_QWORD **)(a2 + 80);
  if ( (_QWORD *)(a2 + 72) == v6 )
    return;
  if ( !v6 )
    goto LABEL_115;
  while ( 1 )
  {
    i = (_QWORD *)v6[4];
    if ( i != v6 + 3 )
      break;
    v6 = (_QWORD *)v6[1];
    if ( (_QWORD *)v5 == v6 )
      return;
    if ( !v6 )
      goto LABEL_115;
  }
  if ( (_QWORD *)v5 != v6 )
  {
    v78 = 0;
    v75 = 0;
    v79 = 0;
    v74 = a1 + 3;
    v9 = a2 + 72;
    v10 = a2;
    do
    {
      if ( !i )
        BUG();
      v11 = *((unsigned __int8 *)i - 24);
      if ( (unsigned __int8)v11 > 0x1Cu && (unsigned __int8)(v11 - 34) <= 0x33u )
      {
        a2 = 0x8000000000041LL;
        if ( _bittest64(&a2, (unsigned int)(v11 - 34)) )
        {
          v12 = *(i - 7);
          if ( v12 )
          {
            v13 = (unsigned __int8 *)(i - 3);
            if ( !*(_BYTE *)v12 )
            {
              v14 = *(_QWORD *)(v12 + 24);
              if ( v14 == i[7] && (a2 = (unsigned int)(*(_DWORD *)(v12 + 36) - 36), (unsigned int)a2 <= 2) )
              {
                v28 = *((unsigned int *)a1 + 44);
                if ( v28 + 1 > (unsigned __int64)*((unsigned int *)a1 + 45) )
                {
                  a2 = (__int64)(a1 + 23);
                  sub_C8D5F0((__int64)(a1 + 21), a1 + 23, v28 + 1, 8u, a5, (__int64)v13);
                  v28 = *((unsigned int *)a1 + 44);
                  v13 = (unsigned __int8 *)(i - 3);
                }
                *(_QWORD *)(a1[21] + 8 * v28) = v13;
                ++*((_DWORD *)a1 + 44);
              }
              else if ( (_BYTE)v11 == 85 && v14 == i[7] && (*(_BYTE *)(v12 + 33) & 0x20) != 0 )
              {
                switch ( *(_DWORD *)(v12 + 36) )
                {
                  case 0x1B:
                    v45 = *((unsigned int *)a1 + 24);
                    if ( v45 + 1 > (unsigned __int64)*((unsigned int *)a1 + 25) )
                    {
                      a2 = (__int64)(a1 + 13);
                      sub_C8D5F0((__int64)(a1 + 11), a1 + 13, v45 + 1, 8u, a5, (__int64)v13);
                      v45 = *((unsigned int *)a1 + 24);
                      v13 = (unsigned __int8 *)(i - 3);
                    }
                    *(_QWORD *)(a1[11] + 8 * v45) = v13;
                    ++*((_DWORD *)a1 + 24);
                    break;
                  case 0x27:
                  case 0x28:
                    v41 = *(_QWORD *)&v13[-32 * (*((_DWORD *)i - 5) & 0x7FFFFFF)];
                    if ( !v41 )
                      BUG();
                    v42 = *(_QWORD *)(v41 - 32);
                    if ( !v42 )
                      goto LABEL_116;
                    if ( *(_BYTE *)v42 )
                      goto LABEL_116;
                    a2 = *(_QWORD *)(v41 + 80);
                    if ( *(_QWORD *)(v42 + 24) != a2 )
                      goto LABEL_116;
                    if ( *(_DWORD *)(v42 + 36) != 48
                      || (v54 = sub_BD3990(
                                  *(unsigned __int8 **)(v41 + 32 * (3LL - (*(_DWORD *)(v41 + 4) & 0x7FFFFFF))),
                                  a2),
                          v13 = (unsigned __int8 *)(i - 3),
                          *v54 != 3)
                      || **((_BYTE **)v54 - 4) == 10 )
                    {
                      if ( *a1 )
                        sub_C64ED0("coroutine should have exactly one defining @llvm.coro.begin", 1u);
                      v72 = (__int64)v13;
                      v43 = (__int64 *)sub_BD5C60((__int64)v13);
                      i[6] = sub_A7A090(i + 6, v43, 0, 43);
                      v44 = (__int64 *)sub_BD5C60(v72);
                      i[6] = sub_A7A090(i + 6, v44, 0, 22);
                      a2 = sub_BD5C60(v72);
                      i[6] = sub_A7B980(i + 6, (__int64 *)a2, -1, 27);
                      *a1 = v72;
                    }
                    break;
                  case 0x2B:
                  case 0x2C:
                    v29 = *((unsigned int *)a1 + 4);
                    if ( v29 + 1 > (unsigned __int64)*((unsigned int *)a1 + 5) )
                    {
                      sub_C8D5F0((__int64)(a1 + 1), v74, v29 + 1, 8u, a5, (__int64)v13);
                      v29 = *((unsigned int *)a1 + 4);
                      v13 = (unsigned __int8 *)(i - 3);
                    }
                    *(_QWORD *)(a1[1] + 8 * v29) = v13;
                    v30 = (unsigned int)(*((_DWORD *)a1 + 4) + 1);
                    *((_DWORD *)a1 + 4) = v30;
                    v31 = *(i - 7);
                    if ( !v31 )
                      goto LABEL_116;
                    if ( *(_BYTE *)v31 )
                      goto LABEL_116;
                    a2 = i[7];
                    if ( *(_QWORD *)(v31 + 24) != a2 )
                      goto LABEL_116;
                    if ( *(_DWORD *)(v31 + 36) == 44 )
                    {
                      sub_24F3C30(v13, a2);
                      v30 = *((unsigned int *)a1 + 4);
                    }
                    v32 = *(_QWORD *)(a1[1] + 8 * v30 - 8);
                    v33 = sub_AD7A80(
                            *(_BYTE **)(v32 + 32 * (1LL - (*(_DWORD *)(v32 + 4) & 0x7FFFFFF))),
                            a2,
                            *(_DWORD *)(v32 + 4) & 0x7FFFFFF,
                            v32,
                            a5);
                    v35 = v78;
                    if ( v33 )
                      v35 = v33;
                    v36 = *(_QWORD *)(a1[1] + 8LL * *((unsigned int *)a1 + 4) - 8);
                    v78 = v35;
                    if ( sub_AD7A80(
                           *(_BYTE **)(v36 + 32 * (1LL - (*(_DWORD *)(v36 + 4) & 0x7FFFFFF))),
                           a2,
                           *(_DWORD *)(v36 + 4) & 0x7FFFFFF,
                           v36,
                           v34) )
                    {
                      break;
                    }
                    v37 = *(i - 7);
                    if ( !v37 || *(_BYTE *)v37 || *(_QWORD *)(v37 + 24) != i[7] )
                      goto LABEL_116;
                    if ( *(_DWORD *)(v37 + 36) == 43 && *((_DWORD *)a1 + 4) > 1u )
                    {
                      if ( !sub_AD7A80(
                              *(_BYTE **)(*(_QWORD *)a1[1]
                                        + 32 * (1LL - (*(_DWORD *)(*(_QWORD *)a1[1] + 4LL) & 0x7FFFFFF))),
                              a2,
                              *(_DWORD *)(*(_QWORD *)a1[1] + 4LL) & 0x7FFFFFF,
                              *(_QWORD *)a1[1],
                              a5) )
                        sub_C64ED0("Only one coro.end can be marked as fallthrough", 1u);
                      v38 = (__int64 *)a1[1];
                      v39 = &v38[*((unsigned int *)a1 + 4) - 1];
                      v40 = *v38;
                      a2 = *v39;
                      *v38 = *v39;
                      *v39 = v40;
                    }
                    break;
                  case 0x2E:
                    v53 = *(unsigned int *)(a3 + 8);
                    if ( v53 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
                    {
                      sub_C8D5F0(a3, (const void *)(a3 + 16), v53 + 1, 8u, a5, (__int64)v13);
                      v13 = (unsigned __int8 *)(i - 3);
                      v53 = *(unsigned int *)(a3 + 8);
                    }
                    a2 = a3;
                    *(_QWORD *)(*(_QWORD *)a3 + 8 * v53) = v13;
                    ++*(_DWORD *)(a3 + 8);
                    break;
                  case 0x39:
                    if ( !*(i - 1) )
                    {
                      a2 = a4;
                      v52 = *(unsigned int *)(a4 + 8);
                      if ( v52 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
                      {
                        a2 = a4 + 16;
                        sub_C8D5F0(a4, (const void *)(a4 + 16), v52 + 1, 8u, a5, (__int64)v13);
                        v13 = (unsigned __int8 *)(i - 3);
                        v52 = *(unsigned int *)(a4 + 8);
                      }
                      *(_QWORD *)(*(_QWORD *)a4 + 8 * v52) = v13;
                      ++*(_DWORD *)(a4 + 8);
                    }
                    break;
                  case 0x3A:
                    v51 = *((unsigned int *)a1 + 16);
                    if ( v51 + 1 > (unsigned __int64)*((unsigned int *)a1 + 17) )
                    {
                      a2 = (__int64)(a1 + 9);
                      sub_C8D5F0((__int64)(a1 + 7), a1 + 9, v51 + 1, 8u, a5, (__int64)v13);
                      v51 = *((unsigned int *)a1 + 16);
                      v13 = (unsigned __int8 *)(i - 3);
                    }
                    *(_QWORD *)(a1[7] + 8 * v51) = v13;
                    ++*((_DWORD *)a1 + 16);
                    break;
                  case 0x3C:
                    v48 = *((unsigned int *)a1 + 32);
                    v49 = *((unsigned int *)a1 + 33);
                    if ( v48 + 1 > v49 )
                    {
                      a2 = (__int64)(a1 + 17);
                      sub_C8D5F0((__int64)(a1 + 15), a1 + 17, v48 + 1, 8u, a5, (__int64)v13);
                      v48 = *((unsigned int *)a1 + 32);
                      v13 = (unsigned __int8 *)(i - 3);
                    }
                    *(_QWORD *)(a1[15] + 8 * v48) = v13;
                    ++*((_DWORD *)a1 + 32);
                    v50 = sub_AD7A80(
                            (_BYTE *)i[4 * (1LL - (*((_DWORD *)i - 5) & 0x7FFFFFF)) - 3],
                            a2,
                            *((_DWORD *)i - 5) & 0x7FFFFFF,
                            v49,
                            a5);
                    if ( v50 )
                    {
                      if ( v79 )
                        sub_C64ED0("Only one suspend point can be marked as final", 1u);
                      v79 = v50;
                      a2 = *((unsigned int *)a1 + 32) - 1LL;
                      v75 = a2;
                    }
                    break;
                  case 0x3D:
                    v73 = (unsigned __int8 *)(i - 3);
                    sub_24F3BC0((__int64)(i - 3), a2);
                    v46 = *((unsigned int *)a1 + 32);
                    v13 = (unsigned __int8 *)(i - 3);
                    v47 = v46 + 1;
                    if ( v46 + 1 > (unsigned __int64)*((unsigned int *)a1 + 33) )
                      goto LABEL_73;
                    goto LABEL_71;
                  case 0x3E:
                    v46 = *((unsigned int *)a1 + 32);
                    v47 = v46 + 1;
                    if ( v46 + 1 <= (unsigned __int64)*((unsigned int *)a1 + 33) )
                      goto LABEL_71;
                    v73 = (unsigned __int8 *)(i - 3);
LABEL_73:
                    a2 = (__int64)(a1 + 17);
                    sub_C8D5F0((__int64)(a1 + 15), a1 + 17, v47, 8u, a5, (__int64)v13);
                    v46 = *((unsigned int *)a1 + 32);
                    v13 = v73;
LABEL_71:
                    *(_QWORD *)(a1[15] + 8 * v46) = v13;
                    ++*((_DWORD *)a1 + 32);
                    break;
                  default:
                    break;
                }
              }
            }
          }
        }
      }
      for ( i = (_QWORD *)i[1]; i == v6 + 3; i = (_QWORD *)v6[4] )
      {
        v6 = (_QWORD *)v6[1];
        if ( (_QWORD *)v9 == v6 )
          goto LABEL_25;
        if ( !v6 )
          goto LABEL_115;
      }
    }
    while ( v6 != (_QWORD *)v9 );
LABEL_25:
    v15 = *a1;
    v16 = v10;
    if ( *a1 )
    {
      v17 = *(_QWORD *)(v15 - 32LL * (*(_DWORD *)(v15 + 4) & 0x7FFFFFF));
      v18 = *(_QWORD *)(v17 - 32);
      if ( !v18 || *(_BYTE *)v18 || *(_QWORD *)(v18 + 24) != *(_QWORD *)(v17 + 80) )
LABEL_116:
        BUG();
      v19 = *(_DWORD *)(v18 + 36);
      if ( v19 != 49 )
      {
        if ( v19 > 0x31 )
        {
          if ( v19 - 50 <= 1 )
          {
            *((_DWORD *)a1 + 70) = (v19 != 50) + 1;
            v70 = *(_QWORD *)(v15 - 32LL * (*(_DWORD *)(v15 + 4) & 0x7FFFFFF));
            sub_24F3850(v70, a2);
            a1[41] = (__int64)sub_BD3990(
                                *(unsigned __int8 **)(v70 + 32 * (3LL - (*(_DWORD *)(v70 + 4) & 0x7FFFFFF))),
                                a2);
            a1[42] = (__int64)sub_BD3990(
                                *(unsigned __int8 **)(v70 + 32 * (4LL - (*(_DWORD *)(v70 + 4) & 0x7FFFFFF))),
                                a2);
            v71 = sub_BD3990(*(unsigned __int8 **)(v70 + 32 * (5LL - (*(_DWORD *)(v70 + 4) & 0x7FFFFFF))), a2);
            *((_BYTE *)a1 + 360) = 0;
            a1[43] = (__int64)v71;
            a1[44] = 0;
            return;
          }
        }
        else if ( v19 == 48 )
        {
          *((_DWORD *)a1 + 70) = 0;
          *((_BYTE *)a1 + 364) = v79;
          *((_BYTE *)a1 + 365) = v78;
          v20 = *(_QWORD *)(v15 - 32LL * (*(_DWORD *)(v15 + 4) & 0x7FFFFFF));
          a1[41] = 0;
          v21 = *(unsigned __int8 **)(v20 + 32 * (1LL - (*(_DWORD *)(v20 + 4) & 0x7FFFFFF)));
          v22 = 0;
          if ( *v21 != 20 )
          {
            v22 = sub_BD3990(v21, v78);
            v79 = *((_BYTE *)a1 + 364);
          }
          a1[42] = (__int64)v22;
          a1[43] = 0;
          if ( v79 )
          {
            v23 = *((unsigned int *)a1 + 32);
            if ( v23 - 1 != v75 )
            {
              v24 = a1[15];
              v25 = (__int64 *)(v24 + 8 * v23 - 8);
              v26 = (__int64 *)(v24 + 8 * v75);
              v27 = *v26;
              *v26 = *v25;
              *v25 = v27;
            }
          }
          return;
        }
LABEL_115:
        BUG();
      }
      *((_DWORD *)a1 + 70) = 3;
      v55 = *(_QWORD *)(v15 - 32LL * (*(_DWORD *)(v15 + 4) & 0x7FFFFFF));
      sub_24F3B00(v55, a2);
      v57 = *(_QWORD *)(*(_QWORD *)(v55 + 40) + 72LL);
      v58 = *(_DWORD *)(v55 + 4) & 0x7FFFFFF;
      v59 = *(_QWORD *)(v55 + 32 * (2 - v58));
      if ( *(_DWORD *)(v59 + 32) <= 0x40u )
        v60 = *(_QWORD *)(v59 + 24);
      else
        v60 = **(_QWORD **)(v59 + 24);
      if ( (*(_BYTE *)(v57 + 2) & 1) != 0 )
        sub_B2C6D0(*(_QWORD *)(*(_QWORD *)(v55 + 40) + 72LL), a2, v58, v56);
      a1[41] = *(_QWORD *)(v57 + 96) + 40LL * (unsigned int)v60;
      v61 = *(_QWORD *)(v55 + 32 * (2LL - (*(_DWORD *)(v55 + 4) & 0x7FFFFFF)));
      v62 = *(_DWORD *)(v61 + 32) <= 0x40u;
      v63 = *(_QWORD **)(v61 + 24);
      if ( !v62 )
        v63 = (_QWORD *)*v63;
      *((_DWORD *)a1 + 85) = (_DWORD)v63;
      v64 = *(_QWORD *)(v55 - 32LL * (*(_DWORD *)(v55 + 4) & 0x7FFFFFF));
      v62 = *(_DWORD *)(v64 + 32) <= 0x40u;
      v65 = *(_QWORD **)(v64 + 24);
      if ( !v62 )
        v65 = (_QWORD *)*v65;
      a1[43] = (__int64)v65;
      v66 = *(_QWORD *)(v55 + 32 * (1LL - (*(_DWORD *)(v55 + 4) & 0x7FFFFFF)));
      v62 = *(_DWORD *)(v66 + 32) <= 0x40u;
      v67 = *(_QWORD **)(v66 + 24);
      if ( !v62 )
        v67 = (_QWORD *)*v67;
      v68 = 1;
      if ( v67 )
      {
        _BitScanReverse64(&v69, (unsigned __int64)v67);
        v68 = 0x8000000000000000LL >> ((unsigned __int8)v69 ^ 0x3Fu);
      }
      a1[44] = v68;
      a1[47] = (__int64)sub_BD3990(*(unsigned __int8 **)(v55 + 32 * (3LL - (*(_DWORD *)(v55 + 4) & 0x7FFFFFF))), a2);
      *((_DWORD *)a1 + 84) = (*(_WORD *)(v16 + 2) >> 4) & 0x3FF;
    }
  }
}
