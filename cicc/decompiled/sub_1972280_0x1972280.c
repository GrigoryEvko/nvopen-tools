// Function: sub_1972280
// Address: 0x1972280
//
__int64 __fastcall sub_1972280(
        __int64 *a1,
        __int64 a2,
        __m128i a3,
        __m128i a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  const char *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r14
  char v14; // al
  __int64 v15; // rcx
  int v16; // eax
  int v17; // edx
  double v18; // xmm4_8
  double v19; // xmm5_8
  unsigned int v21; // r15d
  __int64 v22; // r15
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r14
  __int64 v26; // rbx
  __int64 v27; // r15
  int v28; // r13d
  _QWORD *v29; // rax
  __int64 v30; // rdi
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // r13
  __int64 v37; // rbx
  unsigned __int64 v38; // rax
  __int64 v39; // rax
  __int64 v40; // r14
  unsigned __int64 v41; // rax
  __int64 v42; // r13
  unsigned __int64 v43; // rax
  __int64 v44; // rax
  __int64 v45; // rbx
  __int64 *v46; // rcx
  __int64 v47; // rdx
  __int64 v48; // r15
  __int64 v49; // rax
  char v50; // dl
  unsigned int v51; // edx
  bool v52; // al
  __int64 v53; // rbx
  __int64 i; // rcx
  __int64 *v55; // r8
  __int64 v56; // rax
  int v57; // eax
  __int64 j; // rdx
  unsigned __int64 v59; // r13
  __int64 v60; // rsi
  __int64 *v61; // r8
  __int64 **v62; // rbx
  __int64 v63; // [rsp+0h] [rbp-60h]
  __int64 v64; // [rsp+8h] [rbp-58h]
  __int64 *v65; // [rsp+10h] [rbp-50h]
  __int64 v66; // [rsp+10h] [rbp-50h]
  __int64 v67; // [rsp+18h] [rbp-48h]
  __int64 v68; // [rsp+18h] [rbp-48h]
  int v69; // [rsp+20h] [rbp-40h]
  int v70; // [rsp+20h] [rbp-40h]
  __int64 v71; // [rsp+28h] [rbp-38h]

  v11 = sub_1649960(*(_QWORD *)(**(_QWORD **)(a2 + 32) + 56LL));
  if ( v12 != 6
    || (*(_DWORD *)v11 != 1936549229 || *((_WORD *)v11 + 2) != 29797)
    && (*(_DWORD *)v11 != 1668113773 || *((_WORD *)v11 + 2) != 31088) )
  {
    v13 = *(_QWORD *)(**(_QWORD **)(a2 + 32) + 56LL) + 112LL;
    if ( (unsigned __int8)sub_1560180(v13, 34) || (v14 = sub_1560180(v13, 17)) != 0 )
      v14 = byte_4FB0700;
    v15 = a1[5];
    *((_BYTE *)a1 + 64) = v14;
    v16 = *(unsigned __int8 *)(*(_QWORD *)v15 + 73LL);
    *((_BYTE *)a1 + 264) = (v16 & 0x30) != 0;
    v17 = (int)*(unsigned __int8 *)(*(_QWORD *)v15 + 73LL) >> 6;
    *((_BYTE *)a1 + 265) = v17 != 0;
    LODWORD(v15) = *(unsigned __int8 *)(*(_QWORD *)v15 + 72LL);
    *((_BYTE *)a1 + 266) = (v15 & 0x30) != 0;
    if ( v17 | (((unsigned int)v15 | v16) >> 4) & 3 && (unsigned __int8)sub_1481F90((_QWORD *)a1[4], a2, a3, a4) )
      return sub_196FF90((__int64)a1, a3, a4, a5, a6, v18, v19, a9, a10);
    if ( (unsigned int)sub_14A2FC0(a1[6]) == 2 )
    {
      v22 = *a1;
      v23 = sub_1969100(**(_QWORD **)(*a1 + 32));
      v25 = v24;
      v26 = v23;
      if ( v23 != v24 )
      {
        v27 = v22 + 56;
        v28 = 0;
        do
        {
          v29 = sub_1648700(v26);
          v28 -= !sub_1377F70(v27, v29[5]) - 1;
          do
            v26 = *(_QWORD *)(v26 + 8);
          while ( v26 && (unsigned __int8)(*((_BYTE *)sub_1648700(v26) + 16) - 25) > 9u );
        }
        while ( v26 != v25 );
        if ( v28 == 1 )
        {
          v30 = *a1;
          v31 = *(_QWORD *)(*a1 + 32);
          if ( (unsigned int)((*(_QWORD *)(*a1 + 40) - v31) >> 3) == 1 )
          {
            v32 = *(_QWORD *)v31 + 40LL;
            v33 = *(_QWORD *)(*(_QWORD *)v31 + 48LL);
            if ( v32 == v33 )
              goto LABEL_28;
            v34 = 0;
            do
            {
              v33 = *(_QWORD *)(v33 + 8);
              ++v34;
            }
            while ( v32 != v33 );
            if ( v34 <= 19 )
            {
LABEL_28:
              v35 = sub_13FC520(v30);
              v36 = v35;
              if ( v35 )
              {
                v37 = *(_QWORD *)(v35 + 48);
                if ( v37 )
                  v37 -= 24;
                v38 = sub_157EBA0(v35);
                if ( v38 == v37 && *(_BYTE *)(v38 + 16) == 26 && (*(_DWORD *)(v38 + 20) & 0xFFFFFFF) != 3 )
                {
                  v39 = sub_157F0B0(v36);
                  v40 = v39;
                  if ( v39 )
                  {
                    v41 = sub_157EBA0(v39);
                    if ( *(_BYTE *)(v41 + 16) == 26 && (*(_DWORD *)(v41 + 20) & 0xFFFFFFF) != 1 )
                    {
                      v71 = *a1;
                      v42 = **(_QWORD **)(*a1 + 32);
                      v43 = sub_157EBA0(v42);
                      if ( *(_BYTE *)(v43 + 16) == 26 && (*(_DWORD *)(v43 + 20) & 0xFFFFFFF) == 3 )
                      {
                        v44 = sub_1969460(v43, v42);
                        v45 = v44;
                        if ( v44 )
                        {
                          if ( *(_BYTE *)(v44 + 16) == 50 )
                          {
                            if ( (*(_BYTE *)(v44 + 23) & 0x40) != 0 )
                              v46 = *(__int64 **)(v44 - 8);
                            else
                              v46 = (__int64 *)(v44 - 24LL * (*(_DWORD *)(v44 + 20) & 0xFFFFFFF));
                            v47 = *v46;
                            v48 = v46[3];
                            if ( (unsigned __int8)(*(_BYTE *)(*v46 + 16) - 35) > 0x11u )
                            {
                              if ( (unsigned __int8)(*(_BYTE *)(v48 + 16) - 35) > 0x11u )
                                return sub_196C0A0(a1, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
                              v47 = v46[3];
                              v48 = *v46;
                            }
                            if ( *(_QWORD *)(v47 - 48) == v48 )
                            {
                              v49 = *(_QWORD *)(v47 - 24);
                              if ( *(_BYTE *)(v49 + 16) == 13 )
                              {
                                v50 = *(_BYTE *)(v47 + 16);
                                if ( v50 == 37 )
                                {
                                  if ( *(_DWORD *)(v49 + 32) <= 0x40u )
                                  {
                                    v52 = *(_QWORD *)(v49 + 24) == 1;
                                  }
                                  else
                                  {
                                    v70 = *(_DWORD *)(v49 + 32);
                                    v52 = v70 - 1 == (unsigned int)sub_16A57B0(v49 + 24);
                                  }
LABEL_52:
                                  if ( v52 )
                                  {
                                    v53 = sub_19695C0(v48, v45, v42);
                                    if ( v53 )
                                    {
                                      for ( i = sub_157ED20(v42) + 24; v42 + 40 != i; i = *(_QWORD *)(i + 8) )
                                      {
                                        if ( !i )
                                          BUG();
                                        if ( *(_BYTE *)(i - 8) == 35 )
                                        {
                                          v55 = (*(_BYTE *)(i - 1) & 0x40) != 0
                                              ? *(__int64 **)(i - 32)
                                              : (__int64 *)(i - 24 - 24LL * (*(_DWORD *)(i - 4) & 0xFFFFFFF));
                                          v56 = v55[3];
                                          if ( *(_BYTE *)(v56 + 16) == 13 )
                                          {
                                            v21 = *(_DWORD *)(v56 + 32);
                                            if ( v21 <= 0x40 )
                                            {
                                              LOBYTE(v21) = *(_QWORD *)(v56 + 24) == 1;
                                            }
                                            else
                                            {
                                              v65 = v55;
                                              v67 = i;
                                              v57 = sub_16A57B0(v56 + 24);
                                              i = v67;
                                              v55 = v65;
                                              LOBYTE(v21) = v21 - 1 == v57;
                                            }
                                            if ( (_BYTE)v21 )
                                            {
                                              v68 = i - 24;
                                              v66 = sub_19695C0(*v55, i - 24, v42);
                                              if ( v66 )
                                              {
                                                for ( j = *(_QWORD *)(i - 16); j; j = *(_QWORD *)(v64 + 8) )
                                                {
                                                  v63 = i;
                                                  v64 = j;
                                                  if ( v42 != sub_1648700(j)[5] )
                                                  {
                                                    v59 = sub_157EBA0(v40);
                                                    if ( *(_BYTE *)(v59 + 16) == 26 )
                                                    {
                                                      v60 = sub_13FC520(v71);
                                                      if ( (*(_DWORD *)(v59 + 20) & 0xFFFFFFF) == 3 )
                                                        v61 = (__int64 *)sub_1969460(v59, v60);
                                                      else
                                                        v61 = 0;
                                                    }
                                                    else
                                                    {
                                                      sub_13FC520(v71);
                                                      v61 = 0;
                                                    }
                                                    if ( (*(_BYTE *)(v53 + 23) & 0x40) != 0 )
                                                      v62 = *(__int64 ***)(v53 - 8);
                                                    else
                                                      v62 = (__int64 **)(v53 - 24LL
                                                                             * (*(_DWORD *)(v53 + 20) & 0xFFFFFFF));
                                                    if ( *v62 == v61 || v62[3] == v61 )
                                                    {
                                                      sub_196A9C0(a1, v40, v68, v66, v61);
                                                      return v21;
                                                    }
                                                    return sub_196C0A0(
                                                             a1,
                                                             *(double *)a3.m128i_i64,
                                                             *(double *)a4.m128i_i64,
                                                             a5);
                                                  }
                                                  i = v63;
                                                }
                                              }
                                            }
                                          }
                                        }
                                      }
                                    }
                                  }
                                  return sub_196C0A0(a1, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
                                }
                                if ( v50 == 35 )
                                {
                                  v51 = *(_DWORD *)(v49 + 32);
                                  if ( v51 <= 0x40 )
                                  {
                                    v52 = 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v51) == *(_QWORD *)(v49 + 24);
                                  }
                                  else
                                  {
                                    v69 = *(_DWORD *)(v49 + 32);
                                    v52 = v69 == (unsigned int)sub_16A58F0(v49 + 24);
                                  }
                                  goto LABEL_52;
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    return sub_196C0A0(a1, *(double *)a3.m128i_i64, *(double *)a4.m128i_i64, a5);
  }
  return 0;
}
