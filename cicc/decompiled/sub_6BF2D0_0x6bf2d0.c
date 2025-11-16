// Function: sub_6BF2D0
// Address: 0x6bf2d0
//
__int64 __fastcall sub_6BF2D0(
        __int64 a1,
        __m128i *a2,
        _BYTE *a3,
        unsigned int a4,
        __int16 a5,
        int a6,
        _DWORD *a7,
        _DWORD *a8,
        __int64 *a9)
{
  __int64 v9; // r15
  __int64 *v11; // r12
  int v13; // ebx
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v17; // r13
  int v18; // eax
  __int64 v19; // rcx
  __int64 v20; // r8
  int v21; // r13d
  int v22; // eax
  __int64 v23; // rsi
  __int64 v24; // rdx
  __int64 v25; // rcx
  __int64 v26; // r8
  __int64 v27; // r9
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // r13
  __int64 v32; // rax
  __int64 v33; // rbx
  __int64 v34; // r13
  int v35; // r13d
  int v36; // eax
  __int64 v37; // rax
  __int64 v38; // r13
  __int64 v39; // rcx
  __int64 i; // rax
  char v41; // dl
  int v42; // eax
  unsigned int v43; // r13d
  _BYTE *v44; // rax
  __int64 v45; // rcx
  __int64 v46; // r8
  __int64 v47; // rcx
  __int64 v48; // r8
  __int64 v49; // r13
  __int64 *v50; // rax
  int v51; // r12d
  __int64 *v52; // rbx
  __int64 *v53; // r13
  __int64 j; // rax
  unsigned __int64 v55; // rax
  __int64 *v56; // rax
  __int64 k; // rax
  unsigned __int64 v58; // rdx
  __int64 m; // rax
  __int64 v60; // [rsp-10h] [rbp-1E0h]
  unsigned __int64 v62; // [rsp+0h] [rbp-1D0h]
  int v63; // [rsp+8h] [rbp-1C8h]
  int v64; // [rsp+Ch] [rbp-1C4h] BYREF
  unsigned __int8 v65; // [rsp+17h] [rbp-1B9h] BYREF
  unsigned int v66; // [rsp+18h] [rbp-1B8h] BYREF
  int v67; // [rsp+1Ch] [rbp-1B4h] BYREF
  int v68; // [rsp+20h] [rbp-1B0h] BYREF
  int v69; // [rsp+24h] [rbp-1ACh] BYREF
  unsigned int v70; // [rsp+28h] [rbp-1A8h] BYREF
  char v71[4]; // [rsp+2Ch] [rbp-1A4h] BYREF
  __int64 v72; // [rsp+30h] [rbp-1A0h] BYREF
  __int64 v73; // [rsp+38h] [rbp-198h] BYREF
  _BYTE v74[400]; // [rsp+40h] [rbp-190h] BYREF

  v9 = a1;
  v11 = (__int64 *)a2;
  v64 = a6;
  v67 = 0;
  v68 = 0;
  v69 = 0;
  v65 = 0;
  if ( !dword_4F077BC
    || qword_4F077A8 <= 0x9DCFu
    || !(unsigned int)sub_8D4C80(a1)
    && (!(unsigned int)sub_8D2E30(a1) || (v28 = sub_8D46C0(a1), !(unsigned int)sub_8D2310(v28)))
    || !(unsigned int)sub_8D3D10(a2->m128i_i64[0]) )
  {
LABEL_3:
    if ( (a2[1].m128i_i8[2] & 1) == 0 )
      goto LABEL_4;
    goto LABEL_36;
  }
  if ( (a2[1].m128i_i8[2] & 1) == 0 )
  {
    a3 = v74;
    v29 = sub_8D4890(a2->m128i_i64[0]);
    sub_6EA0A0(v29, v74);
    sub_82F1E0(v74, 0, a2);
    goto LABEL_3;
  }
LABEL_36:
  sub_68FA30(a1, a8, a2, (__int64)a3);
LABEL_4:
  v13 = v64;
  if ( !v64 )
  {
    v63 = sub_8D2600(a1);
    if ( dword_4F077C4 == 2 )
    {
      v13 = sub_8D32E0(a1);
      sub_6BEBB0(a1, a2, a4, &v69, &v68, &v64);
    }
    if ( v68 )
      goto LABEL_34;
    if ( v13 | v63 )
    {
      sub_6F69D0(a2, 15);
      if ( (unsigned int)sub_69A8F0(a2->m128i_i64, a1, a5, a7, &v65) )
      {
        v17 = a2->m128i_i64[0];
        v73 = a1;
        v18 = v64;
        v72 = v17;
        if ( v13 )
        {
          if ( v64 )
            goto LABEL_5;
          sub_68F0D0(a1, a2, v69, 1, a4, a7, &v73, &v72, &v67);
          v18 = v64;
        }
LABEL_13:
        if ( v18 )
          goto LABEL_5;
        if ( v67 )
          goto LABEL_6;
        if ( a2[1].m128i_i8[0] == 3 )
        {
          sub_6FC070(a1, a2, 1, 0, 0);
          goto LABEL_34;
        }
        if ( v63 )
        {
          sub_6F7220(a2, a1);
          goto LABEL_34;
        }
        if ( unk_4D041FC
          && a2[1].m128i_i8[1] == 1
          && !(unsigned int)sub_6ED0A0(a2)
          && (a1 == v17 || (unsigned int)sub_8D97D0(v17, a1, 32, v47, v48))
          && (!sub_6ED2B0(a2) || !(unsigned int)sub_8D2780(v17))
          && (*(_BYTE *)(qword_4D03C50 + 17LL) & 0x20) == 0
          && !(unsigned int)sub_6ECD10(a2) )
        {
LABEL_102:
          sub_691730(a1, a2->m128i_i64);
          goto LABEL_34;
        }
        if ( dword_4F077C0
          && (qword_4F077A8 <= 0x9C3Fu && a2[1].m128i_i8[1] == 1 && !(unsigned int)sub_6ED0A0(a2)
           || (unsigned int)sub_8D3A70(a1))
          && (a1 == v17 || (unsigned int)sub_8D97D0(v17, a1, 32, v19, v20)) )
        {
          goto LABEL_34;
        }
        if ( (unsigned int)sub_68B3F0(a2->m128i_i64, a1) )
        {
          v31 = a2[5].m128i_i64[1];
          v32 = sub_6F6F40(a2, 0);
          v33 = sub_691700(v32, a1, 0);
          sub_6E2DD0(a2, 1);
          a2[9].m128i_i64[0] = v33;
          a2->m128i_i64[0] = a1;
          a2[5].m128i_i64[1] = v31;
          goto LABEL_34;
        }
        if ( a2[1].m128i_i8[1] == 1
          && !(unsigned int)sub_6ED0A0(a2)
          && (dword_4F077C4 == 1 || unk_4D0436C)
          && !(unsigned int)sub_6ECD10(a2)
          && (unsigned int)sub_6EEB90(v17, a1) )
        {
          goto LABEL_102;
        }
        v70 = 0;
        if ( v13 )
        {
          v21 = v73;
          v22 = sub_6EB660(a2);
          v23 = 0;
          if ( !(unsigned int)sub_8E2F20(
                                v72,
                                0,
                                (*((_BYTE *)v11 + 19) & 0x10) != 0,
                                v22,
                                0,
                                v21,
                                (__int64)&v70,
                                171,
                                (__int64)&v66)
            || v70 && *((_BYTE *)v11 + 17) == 2 )
          {
            goto LABEL_29;
          }
        }
        else
        {
          v34 = a2[5].m128i_i64[1];
          sub_6FA3A0(a2);
          if ( dword_4F077C0 && qword_4F077A8 <= 0x9C3Fu )
            a2[6].m128i_i64[0] = v34;
          v35 = v73;
          if ( a2[1].m128i_i8[0] == 2 )
          {
            v36 = sub_6EB660(a2);
            v23 = 1;
            if ( !(unsigned int)sub_8E2F20(
                                  v72,
                                  1,
                                  (*((_BYTE *)v11 + 19) & 0x10) != 0,
                                  v36,
                                  (int)v11 + 144,
                                  v35,
                                  (__int64)&v70,
                                  171,
                                  (__int64)&v66) )
            {
              if ( qword_4D0495C )
              {
                if ( (v11[39] & 0xFF0000000008LL) == 0x70000000000LL && (v11[42] & 2) != 0 )
                {
                  if ( (unsigned int)sub_8D2E30(a1) )
                  {
                    v37 = sub_8D46C0(a1);
                    if ( (unsigned int)sub_8D2310(v37) )
                    {
                      v38 = *(_QWORD *)v11[43];
                      sub_69D070(0x269u, a8);
                      sub_6EAB60(v38, (*((_BYTE *)v11 + 18) & 0x40) != 0, 0, (_DWORD)a8, (_DWORD)a9, 0, (__int64)v11);
                      if ( *((_BYTE *)v11 + 16) )
                      {
                        for ( i = *v11; ; i = *(_QWORD *)(i + 160) )
                        {
                          v41 = *(_BYTE *)(i + 140);
                          if ( v41 != 12 )
                            break;
                        }
                        if ( v41 )
                        {
                          sub_6F5960(v11, 0, 0, v39, v60, 617);
                          sub_6FB850(a1, (_DWORD)v11, (_DWORD)a7, 0, 0, 0, 0, v70);
                        }
                      }
                      goto LABEL_34;
                    }
                  }
                }
              }
              goto LABEL_29;
            }
          }
          else
          {
            v42 = sub_6EB660(a2);
            v23 = 0;
            if ( !(unsigned int)sub_8E2F20(
                                  v72,
                                  0,
                                  (*((_BYTE *)v11 + 19) & 0x10) != 0,
                                  v42,
                                  0,
                                  v35,
                                  (__int64)&v70,
                                  171,
                                  (__int64)&v66) )
            {
LABEL_29:
              if ( dword_4F077C0 && (unsigned int)sub_8D3B10(a1) )
              {
                a1 = (__int64)v11;
                v23 = sub_832ED0(v11, v9);
                if ( v23 )
                {
                  sub_832FD0(v9, v23, v11);
                  goto LABEL_34;
                }
                v64 = 1;
              }
              else
              {
                v64 = 1;
                if ( !(unsigned int)sub_8D3A70(a1) )
                {
                  if ( (unsigned int)sub_6E5430(a1, v23, v24, v25, v26, v27) )
                    sub_6851C0(0xABu, a8);
                  goto LABEL_34;
                }
              }
              if ( (unsigned int)sub_6E5430(a1, v23, v24, v25, v26, v27) )
                sub_685360(0x77u, a7, v9);
LABEL_34:
              if ( !v64 )
                goto LABEL_6;
              goto LABEL_5;
            }
          }
        }
        v43 = v66;
        if ( !v66 )
          goto LABEL_142;
        if ( (unsigned int)sub_6E53E0(5, v66, a8) )
          sub_684B30(v43, a8);
        if ( v66 != 767 )
        {
LABEL_142:
          if ( !(unsigned int)sub_8D2780(a1) || (unsigned int)sub_8D29A0(a1) || *((_BYTE *)v11 + 16) != 1 )
            goto LABEL_155;
          v49 = v11[18];
          v62 = -1;
          if ( *(_BYTE *)(v49 + 24) == 1 )
          {
            v50 = v11;
            v51 = v13;
            v52 = (__int64 *)v49;
            v53 = v50;
            do
            {
              if ( *((_BYTE *)v52 + 56) != 5 || !(unsigned int)sub_8D2780(*v52) && !(unsigned int)sub_8D2E30(*v52) )
                break;
              for ( j = *v52; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
                ;
              v55 = *(_QWORD *)(j + 128);
              v52 = (__int64 *)v52[9];
              if ( v62 <= v55 )
                v55 = v62;
              v62 = v55;
            }
            while ( *((_BYTE *)v52 + 24) == 1 );
            v56 = v53;
            v49 = (__int64)v52;
            v13 = v51;
            v11 = v56;
          }
          if ( !(unsigned int)sub_6EFEE0(v49, v71) )
            goto LABEL_155;
          for ( k = *(_QWORD *)v49; *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
            ;
          v58 = *(_QWORD *)(k + 128);
          if ( v62 < v58 )
            goto LABEL_155;
          for ( m = a1; *(_BYTE *)(m + 140) == 12; m = *(_QWORD *)(m + 160) )
            ;
          if ( v58 > *(_QWORD *)(m + 128) )
          {
            if ( (unsigned int)sub_6E53E0(5, 69, a8) )
              sub_684B30(0x45u, a8);
          }
          else
          {
LABEL_155:
            if ( (unsigned int)sub_8D2E30(*v11) && !(unsigned int)sub_8D2E30(a1) )
            {
              if ( !v13 )
              {
                sub_6E6B60(v11, 0);
                goto LABEL_72;
              }
LABEL_96:
              sub_6FAB30(v11, a1, 0, 0, v70);
              goto LABEL_73;
            }
          }
        }
        if ( !v13 )
        {
LABEL_72:
          sub_6FCCE0(a1, (_DWORD)v11, (_DWORD)a8, 0, 0, 0, v70);
LABEL_73:
          if ( (unsigned int)qword_4F077B4 | dword_4F077BC )
          {
            v44 = (_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
            if ( (dword_4F04C44 != -1 || (v44[6] & 6) != 0 || v44[4] == 12)
              && (v44[12] & 0x10) == 0
              && *(_BYTE *)(qword_4D03C50 + 16LL) <= 3u
              && *((_BYTE *)v11 + 16) == 1 )
            {
              sub_6F4B70(v11);
            }
          }
          if ( v70
            && !dword_4D0488C
            && (!v13 || !dword_4F077BC || qword_4F077A8 <= 0x9E97u)
            && (dword_4D04964
             || (!(unsigned int)sub_8D2E30(a1) || !(unsigned int)sub_8D2E30(*v11))
             && *v11 != a1
             && !(unsigned int)sub_8D97D0(a1, *v11, 32, v45, v46))
            && (unsigned int)sub_6E91E0(28, a8) )
          {
            sub_6E6840(v11);
          }
          goto LABEL_34;
        }
        goto LABEL_96;
      }
    }
    else
    {
      sub_6F69D0(a2, 12);
      sub_6F6890(a2, 0);
      if ( (unsigned int)sub_69A8F0(a2->m128i_i64, a1, a5, a7, &v65) )
      {
        v17 = a2->m128i_i64[0];
        v73 = a1;
        v18 = v64;
        v72 = v17;
        goto LABEL_13;
      }
    }
    v30 = a2->m128i_i64[0];
    v73 = a1;
    v64 = 1;
    v72 = v30;
  }
LABEL_5:
  sub_6E6840(v11);
LABEL_6:
  v14 = v65;
  *(__int64 *)((char *)v11 + 68) = *(_QWORD *)a8;
  v15 = *a9;
  *((_BYTE *)v11 + 18) &= 0xD7u;
  *(__int64 *)((char *)v11 + 76) = v15;
  return sub_6E26D0(v14, v11);
}
