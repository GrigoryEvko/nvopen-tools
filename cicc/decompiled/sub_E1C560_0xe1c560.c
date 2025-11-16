// Function: sub_E1C560
// Address: 0xe1c560
//
__int64 __fastcall sub_E1C560(const void **a1, char a2)
{
  __int64 v4; // r8
  __int64 v5; // rdx
  unsigned __int8 *v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rax
  __int64 v9; // r12
  unsigned __int8 *v10; // rax
  unsigned __int64 v11; // r9
  unsigned __int64 v12; // r8
  unsigned __int64 v13; // rdx
  _QWORD *v14; // rax
  unsigned __int8 *v15; // rax
  unsigned __int64 v16; // rdi
  unsigned __int8 *v18; // rax
  int v19; // ecx
  __int64 v20; // rdi
  unsigned __int8 v21; // dl
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rbx
  __int64 v27; // rax
  char v28; // al
  __int64 v29; // rcx
  __int64 v30; // rdx
  __int64 v31; // rcx
  __int64 v32; // r8
  __int64 v33; // r9
  __int64 v34; // r14
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r9
  _BYTE *v39; // rax
  __int64 v40; // rdx
  __int64 v41; // rbx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // rax
  __int64 v46; // r14
  int v47; // ecx
  _BYTE *v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rbx
  __int64 v51; // rcx
  __int64 v52; // r8
  __int64 v53; // r9
  __int64 v54; // rdx
  __int64 v55; // r9
  _BYTE *v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rbx
  char v59; // dl
  __int64 v60; // rax
  char v61; // dl
  __int64 v62; // rax
  char v63; // si
  unsigned __int8 v64; // bl
  __int64 v65; // rdx
  __int64 v66; // rdi
  __int64 v67; // rdx
  char v68; // dl
  __int64 v69; // r10
  _BYTE *v70; // rax
  __int64 v71; // rdx
  __int64 v72; // rdx
  __int64 v73; // rdx
  __int64 v74; // rdx
  __int64 v75; // rdx
  __int64 v76; // rdx
  __int64 v77; // rdx
  __int64 v78; // rbx
  __int64 v79; // rdx
  __int64 v80; // rcx
  __int64 v81; // r8
  _BYTE *v82; // rax
  __int64 v83; // rdx
  __int64 v84; // rcx
  __int64 v85; // r8
  __int64 v86; // r9
  __int64 v87; // r14
  __int64 v88; // rax
  char v89; // al
  __int64 v90; // rdx
  void *v91; // [rsp+8h] [rbp-128h]
  __int64 v92; // [rsp+10h] [rbp-120h]
  char v93; // [rsp+1Fh] [rbp-111h]
  __int64 v94; // [rsp+20h] [rbp-110h]
  void *v95; // [rsp+28h] [rbp-108h]
  char v96; // [rsp+28h] [rbp-108h]
  int v97; // [rsp+28h] [rbp-108h]
  __int64 v98; // [rsp+28h] [rbp-108h]
  __int64 v99; // [rsp+38h] [rbp-F8h] BYREF
  __int64 v100; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v101; // [rsp+48h] [rbp-E8h]
  unsigned __int64 v102; // [rsp+50h] [rbp-E0h]
  char v103; // [rsp+58h] [rbp-D8h]
  __int64 v104[26]; // [rsp+60h] [rbp-D0h] BYREF

  sub_E0F2D0(v104, a1);
  v5 = (__int64)a1[1];
  v6 = (unsigned __int8 *)*a1;
  if ( *a1 == (const void *)v5 )
  {
LABEL_4:
    v7 = (__int64)&v100;
    v8 = ((_BYTE *)a1[91] - (_BYTE *)a1[90]) >> 3;
    v103 = 0;
    v100 = 0;
    v101 = 0;
    v102 = v8;
    v9 = sub_E1D370(a1, &v100);
    if ( !v9 )
      goto LABEL_11;
    v10 = (unsigned __int8 *)a1[90];
    v11 = v102;
    v12 = ((_BYTE *)a1[91] - v10) >> 3;
    if ( v102 < v12 )
    {
      v13 = v102;
      do
      {
        v7 = *(_QWORD *)&v10[8 * v13];
        v15 = (unsigned __int8 *)a1[83];
        v16 = *(_QWORD *)(v7 + 16);
        if ( v15 == a1[84] )
          goto LABEL_11;
        v14 = *(_QWORD **)v15;
        if ( !v14 || v16 >= (__int64)(v14[1] - *v14) >> 3 )
          goto LABEL_11;
        ++v13;
        *(_QWORD *)(v7 + 24) = *(_QWORD *)(*v14 + 8 * v16);
        v10 = (unsigned __int8 *)a1[90];
      }
      while ( v12 != v13 );
    }
    v7 = (__int64)a1[1];
    a1[91] = &v10[8 * v11];
    v18 = (unsigned __int8 *)*a1;
    if ( (const void *)v7 != *a1 )
    {
      v19 = *v18;
      if ( (unsigned __int8)(v19 - 46) > 0x31u || (v20 = 0x2000000800001LL, !_bittest64(&v20, (unsigned int)(v19 - 46))) )
      {
        if ( !a2 )
        {
          do
            *a1 = ++v18;
          while ( *(v18 - 1) && v18 != (unsigned __int8 *)v7 );
          goto LABEL_12;
        }
        v7 = 13;
        if ( !(unsigned __int8)sub_E0F5E0(a1, 0xDu, "Ua9enable_ifI") )
        {
          v46 = 0;
          goto LABEL_39;
        }
        v34 = ((_BYTE *)a1[3] - (_BYTE *)a1[2]) >> 3;
        while ( 1 )
        {
          v39 = *a1;
          if ( *a1 != a1[1] && *v39 == 69 )
            break;
          v99 = sub_E1F480(a1);
          if ( !v99 )
            goto LABEL_11;
          v7 = (__int64)&v99;
          sub_E18380((__int64)(a1 + 2), &v99, v35, v36, v37, v38);
        }
        *a1 = v39 + 1;
        v7 = 32;
        v95 = sub_E11E80(a1, v34, v30, v31, v32, v33);
        v41 = v40;
        v45 = sub_E0E790((__int64)(a1 + 102), 32, v40, v42, v43, v44);
        v46 = v45;
        if ( v45 )
        {
          v47 = *(unsigned __int8 *)(v45 + 10);
          v7 = 16394;
          *(_WORD *)(v45 + 8) = 16394;
          v31 = v47 & 0xFFFFFFF0;
          *(_QWORD *)(v45 + 24) = v41;
          *(_BYTE *)(v45 + 10) = v31 | 5;
          *(_QWORD *)v45 = &unk_49DF188;
          *(_QWORD *)(v45 + 16) = v95;
LABEL_39:
          v94 = 0;
          if ( (_BYTE)v100 || !BYTE1(v100) || (v94 = sub_E1AEA0((__int64)a1, v7, v30, v31, v32)) != 0 )
          {
            v48 = *a1;
            v49 = (__int64)a1[1];
            if ( *a1 != (const void *)v49 && *v48 == 118 )
            {
              v92 = 0;
              v56 = v48 + 1;
              *a1 = v56;
              v91 = 0;
LABEL_51:
              v58 = 0;
              if ( v56 == (_BYTE *)v49
                || *v56 != 81
                || (v59 = *((_BYTE *)a1 + 778),
                    *((_BYTE *)a1 + 778) = 1,
                    *a1 = v56 + 1,
                    v96 = v59,
                    v58 = sub_E18BB0((__int64)a1),
                    *((_BYTE *)a1 + 778) = v96,
                    v58) )
              {
                v7 = 72;
                v97 = HIDWORD(v100);
                v93 = v101;
                v60 = sub_E0E790((__int64)(a1 + 102), 72, (unsigned __int8)v101, v31, v32, v33);
                if ( v60 )
                {
                  *(_QWORD *)(v60 + 24) = v9;
                  *(_WORD *)(v60 + 8) = 19;
                  v61 = *(_BYTE *)(v60 + 10);
                  v9 = v60;
                  *(_QWORD *)(v60 + 16) = v94;
                  *(_QWORD *)(v60 + 48) = v46;
                  *(_QWORD *)(v60 + 32) = v91;
                  *(_BYTE *)(v60 + 10) = v61 & 0xF0 | 1;
                  *(_QWORD *)(v60 + 40) = v92;
                  *(_QWORD *)(v60 + 56) = v58;
                  *(_QWORD *)v60 = &unk_49DF548;
                  *(_DWORD *)(v60 + 64) = v97;
                  *(_BYTE *)(v60 + 68) = v93;
                  goto LABEL_12;
                }
              }
            }
            else
            {
              v50 = (_BYTE *)a1[3] - (_BYTE *)a1[2];
              while ( 1 )
              {
                v99 = sub_E1AEA0((__int64)a1, v7, v49, v31, v32);
                v54 = v99;
                if ( !v99 )
                  break;
                if ( (_BYTE *)a1[3] - (_BYTE *)a1[2] == v50 && v103 )
                {
                  v7 = 24;
                  v98 = v99;
                  v62 = sub_E0E790((__int64)(a1 + 102), 24, v99, v51, v52, v53);
                  if ( !v62 )
                    goto LABEL_11;
                  v63 = *(_BYTE *)(v62 + 10);
                  v54 = v98;
                  v51 = 16471;
                  *(_WORD *)(v62 + 8) = 16471;
                  *(_QWORD *)(v62 + 16) = v98;
                  *(_BYTE *)(v62 + 10) = v63 & 0xF0 | 5;
                  *(_QWORD *)v62 = &unk_49DF4E8;
                  v99 = v62;
                }
                v7 = (__int64)&v99;
                sub_E18380((__int64)(a1 + 2), &v99, v54, v51, v52, v53);
                if ( a1[1] != *a1 )
                {
                  if ( (unsigned __int8)(*(_BYTE *)*a1 - 46) > 0x31u )
                    continue;
                  v31 = 0x2000800800001LL;
                  if ( !_bittest64(&v31, (unsigned int)*(unsigned __int8 *)*a1 - 46) )
                    continue;
                }
                v7 = v50 >> 3;
                v91 = sub_E11E80(a1, v50 >> 3, v49, v31, v32, v55);
                v56 = *a1;
                v92 = v57;
                v49 = (__int64)a1[1];
                goto LABEL_51;
              }
            }
          }
          goto LABEL_11;
        }
        goto LABEL_11;
      }
    }
    goto LABEL_12;
  }
  v7 = *v6;
  if ( (_BYTE)v7 != 71 )
  {
    if ( (_BYTE)v7 == 84 )
    {
      if ( v5 - (_QWORD)v6 == 1 )
      {
LABEL_58:
        v7 = (__int64)(v6 + 1);
        v64 = 0;
        *a1 = v6 + 1;
        if ( (unsigned __int8 *)v5 != v6 + 1 )
          v64 = v6[1];
        if ( !(unsigned __int8)sub_E0DF70((__int64)a1) )
        {
          v7 = 1;
          v65 = sub_E1C560(a1, 1);
          if ( v65 )
          {
            v66 = (__int64)(a1 + 102);
            if ( v64 == 118 )
            {
              v7 = (__int64)"virtual thunk to ";
              v9 = sub_E0FEB0(v66, "virtual thunk to ", v65);
            }
            else
            {
              v7 = (__int64)"non-virtual thunk to ";
              v9 = sub_E0FEB0(v66, "non-virtual thunk to ", v65);
            }
            goto LABEL_12;
          }
        }
        goto LABEL_11;
      }
      v29 = v6[1];
      v7 = (unsigned __int8)(v29 - 65);
      switch ( (char)v29 )
      {
        case 'A':
          *a1 = v6 + 2;
          v90 = sub_E1F480(a1);
          if ( !v90 )
            goto LABEL_11;
          v7 = (__int64)"template parameter object for ";
          v9 = sub_E0FEB0((__int64)(a1 + 102), "template parameter object for ", v90);
          break;
        case 'C':
          *a1 = v6 + 2;
          v78 = sub_E1AEA0((__int64)a1, v7, v5, v29, v4);
          if ( !v78 )
            goto LABEL_11;
          v7 = 1;
          if ( !sub_E0DEF0((char **)a1, 1) )
            goto LABEL_11;
          v82 = *a1;
          if ( *a1 == a1[1] )
            goto LABEL_11;
          if ( *v82 != 95 )
            goto LABEL_11;
          *a1 = v82 + 1;
          v87 = sub_E1AEA0((__int64)a1, 1, v79, v80, v81);
          if ( !v87 )
            goto LABEL_11;
          v7 = 32;
          v88 = sub_E0E790((__int64)(a1 + 102), 32, v83, v84, v85, v86);
          v9 = v88;
          if ( v88 )
          {
            *(_WORD *)(v88 + 8) = 16406;
            v89 = *(_BYTE *)(v88 + 10);
            *(_QWORD *)(v9 + 16) = v87;
            *(_QWORD *)(v9 + 24) = v78;
            *(_BYTE *)(v9 + 10) = v89 & 0xF0 | 5;
            *(_QWORD *)v9 = &unk_49DF668;
          }
          break;
        case 'H':
          v7 = 0;
          *a1 = v6 + 2;
          v77 = sub_E1D370(a1, 0);
          if ( !v77 )
            goto LABEL_11;
          v7 = (__int64)"thread-local initialization routine for ";
          v9 = sub_E0FEB0((__int64)(a1 + 102), "thread-local initialization routine for ", v77);
          break;
        case 'I':
          *a1 = v6 + 2;
          v76 = sub_E1AEA0((__int64)a1, v7, v5, v29, v4);
          if ( !v76 )
            goto LABEL_11;
          v7 = (__int64)"typeinfo for ";
          v9 = sub_E0FEB0((__int64)(a1 + 102), "typeinfo for ", v76);
          break;
        case 'S':
          *a1 = v6 + 2;
          v75 = sub_E1AEA0((__int64)a1, v7, v5, v29, v4);
          if ( !v75 )
            goto LABEL_11;
          v7 = (__int64)"typeinfo name for ";
          v9 = sub_E0FEB0((__int64)(a1 + 102), "typeinfo name for ", v75);
          break;
        case 'T':
          *a1 = v6 + 2;
          v74 = sub_E1AEA0((__int64)a1, v7, v5, v29, v4);
          if ( !v74 )
            goto LABEL_11;
          v7 = (__int64)"VTT for ";
          v9 = sub_E0FEB0((__int64)(a1 + 102), "VTT for ", v74);
          break;
        case 'V':
          *a1 = v6 + 2;
          v73 = sub_E1AEA0((__int64)a1, v7, v5, v29, v4);
          if ( !v73 )
            goto LABEL_11;
          v7 = (__int64)"vtable for ";
          v9 = sub_E0FEB0((__int64)(a1 + 102), "vtable for ", v73);
          break;
        case 'W':
          v7 = 0;
          *a1 = v6 + 2;
          v72 = sub_E1D370(a1, 0);
          if ( !v72 )
            goto LABEL_11;
          v7 = (__int64)"thread-local wrapper routine for ";
          v9 = sub_E0FEB0((__int64)(a1 + 102), "thread-local wrapper routine for ", v72);
          break;
        case 'c':
          *a1 = v6 + 2;
          if ( (unsigned __int8)sub_E0DF70((__int64)a1) )
            goto LABEL_11;
          if ( (unsigned __int8)sub_E0DF70((__int64)a1) )
            goto LABEL_11;
          v7 = 1;
          v71 = sub_E1C560(a1, 1);
          if ( !v71 )
            goto LABEL_11;
          v7 = (__int64)"covariant return thunk to ";
          v9 = sub_E0FEB0((__int64)(a1 + 102), "covariant return thunk to ", v71);
          break;
        default:
          goto LABEL_58;
      }
      goto LABEL_12;
    }
    goto LABEL_4;
  }
  if ( v5 - (_QWORD)v6 == 1 )
    goto LABEL_11;
  v21 = v6[1];
  if ( v21 == 82 )
  {
    v7 = 0;
    *a1 = v6 + 2;
    if ( sub_E1D370(a1, 0) )
    {
      v7 = (__int64)&v100;
      v68 = sub_E0E050((char **)a1, &v100);
      v70 = *a1;
      if ( *a1 != a1[1] && *v70 == 95 )
      {
        *a1 = v70 + 1;
      }
      else if ( !v68 )
      {
        goto LABEL_11;
      }
      v7 = (__int64)"reference temporary for ";
      v9 = sub_E0FEB0((__int64)(a1 + 102), "reference temporary for ", v69);
      goto LABEL_12;
    }
LABEL_11:
    v9 = 0;
    goto LABEL_12;
  }
  if ( v21 == 86 )
  {
    v7 = 0;
    *a1 = v6 + 2;
    v67 = sub_E1D370(a1, 0);
    if ( v67 )
    {
      v7 = (__int64)"guard variable for ";
      v9 = sub_E0FEB0((__int64)(a1 + 102), "guard variable for ", v67);
      goto LABEL_12;
    }
    goto LABEL_11;
  }
  if ( v21 != 73 )
    goto LABEL_11;
  v7 = (__int64)&v100;
  v100 = 0;
  *a1 = v6 + 2;
  v9 = 0;
  if ( !(unsigned __int8)sub_E18460((__int64 *)a1, &v100) )
  {
    v26 = v100;
    if ( v100 )
    {
      v7 = 40;
      v27 = sub_E0E790((__int64)(a1 + 102), 40, v22, v23, v24, v25);
      v9 = v27;
      if ( v27 )
      {
        *(_WORD *)(v27 + 8) = 16405;
        v28 = *(_BYTE *)(v27 + 10);
        *(_QWORD *)(v9 + 16) = 23;
        *(_QWORD *)(v9 + 32) = v26;
        *(_BYTE *)(v9 + 10) = v28 & 0xF0 | 5;
        *(_QWORD *)v9 = &unk_49DF608;
        *(_QWORD *)(v9 + 24) = "initializer for module ";
      }
    }
  }
LABEL_12:
  sub_E0F090(v104, (const void *)v7);
  return v9;
}
