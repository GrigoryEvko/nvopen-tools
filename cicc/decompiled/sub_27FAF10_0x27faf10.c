// Function: sub_27FAF10
// Address: 0x27faf10
//
__int64 __fastcall sub_27FAF10(
        __int64 *a1,
        _QWORD *a2,
        __int64 a3,
        __int64 a4,
        __int64 *a5,
        __int64 **a6,
        __int64 a7,
        __int64 *a8,
        __int64 a9,
        _BYTE *a10,
        __int64 *a11,
        __int64 a12)
{
  __int64 v12; // r14
  __int64 v13; // r15
  __int64 v14; // rbx
  int v15; // ecx
  __int64 v16; // rsi
  int v17; // ecx
  unsigned int v18; // edx
  __int64 *v19; // rax
  __int64 v20; // rdi
  __int64 v21; // rax
  _QWORD *v22; // rbx
  bool v23; // zf
  __int64 v24; // r12
  unsigned __int8 v25; // r13
  __int64 v26; // rax
  __int64 v27; // r8
  __int64 v28; // rax
  __int64 v29; // rax
  __int64 v30; // r9
  __int64 v31; // r13
  unsigned __int8 **v32; // rcx
  int v33; // edx
  unsigned __int8 **v34; // r10
  unsigned __int64 v35; // rax
  __int64 v36; // rax
  int v37; // edx
  int v38; // r13d
  __int64 v39; // r14
  _BYTE *v40; // rbx
  __int64 v41; // r12
  __int64 v42; // r15
  __int64 v43; // rsi
  unsigned __int64 v44; // rax
  __int64 v45; // r13
  char v46; // al
  _QWORD *v47; // rdi
  _QWORD *v48; // rdx
  _QWORD *v49; // rax
  unsigned int v50; // r8d
  __int64 v51; // r9
  unsigned int v52; // edi
  __int64 *v53; // rax
  __int64 v54; // r10
  __int64 v55; // rax
  unsigned __int64 v56; // rdi
  __int64 v57; // rcx
  __int64 v58; // r8
  __int64 v59; // r9
  int v60; // eax
  __int64 v61; // r13
  __int64 v62; // rax
  __int64 v63; // rbx
  __int64 v64; // r15
  __int64 v65; // r14
  __int64 v66; // r12
  __int64 v67; // rsi
  __int64 *v68; // rax
  __int64 *v69; // rcx
  __int64 v70; // rdx
  int v72; // eax
  int v73; // r8d
  int v74; // edx
  __int64 v75; // rax
  __int64 v76; // [rsp+0h] [rbp-160h]
  char *v80; // [rsp+20h] [rbp-140h]
  __int64 v82; // [rsp+30h] [rbp-130h]
  _QWORD *v83; // [rsp+38h] [rbp-128h]
  __int64 v84; // [rsp+38h] [rbp-128h]
  __int64 v85; // [rsp+38h] [rbp-128h]
  unsigned __int8 v87; // [rsp+4Eh] [rbp-112h]
  unsigned __int8 v88; // [rsp+4Fh] [rbp-111h]
  unsigned __int64 v89; // [rsp+50h] [rbp-110h]
  __int64 v90; // [rsp+58h] [rbp-108h]
  char v91; // [rsp+58h] [rbp-108h]
  __int64 v92; // [rsp+58h] [rbp-108h]
  __int64 v93; // [rsp+60h] [rbp-100h]
  char *v94; // [rsp+68h] [rbp-F8h]
  unsigned __int8 **v95; // [rsp+70h] [rbp-F0h] BYREF
  __int64 v96; // [rsp+78h] [rbp-E8h]
  _BYTE v97[32]; // [rsp+80h] [rbp-E0h] BYREF
  char *v98; // [rsp+A0h] [rbp-C0h] BYREF
  int v99; // [rsp+A8h] [rbp-B8h]
  char v100; // [rsp+B0h] [rbp-B0h] BYREF

  sub_F6E9A0((__int64)&v98, a4, a1, a7, (__int64)a5, (__int64)a6);
  v80 = v98;
  v94 = &v98[8 * v99];
  if ( v98 != v94 )
  {
    v12 = a12;
    v87 = 0;
    if ( !a12 )
      v12 = a7;
    v13 = v12;
    while ( 1 )
    {
      v14 = *((_QWORD *)v94 - 1);
      v15 = *(_DWORD *)(a3 + 24);
      v16 = *(_QWORD *)(a3 + 8);
      v93 = v14;
      if ( !v15 )
        goto LABEL_84;
      v17 = v15 - 1;
      v18 = v17 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v19 = (__int64 *)(v16 + 16LL * v18);
      v20 = *v19;
      if ( v14 != *v19 )
        break;
LABEL_8:
      v21 = v19[1];
LABEL_9:
      if ( a7 == v21 )
      {
        v22 = (_QWORD *)(v14 + 48);
        if ( v93 + 48 != *(_QWORD *)(v93 + 56) )
        {
          do
          {
            while ( 1 )
            {
              v23 = (*v22 & 0xFFFFFFFFFFFFFFF8LL) == 0;
              v89 = *v22 & 0xFFFFFFFFFFFFFFF8LL;
              v24 = v89 - 24;
              v22 = (_QWORD *)v89;
              if ( v23 )
                v24 = 0;
              v25 = sub_F50EE0((unsigned __int8 *)v24, a5);
              if ( v25 )
              {
                sub_11C4E30((unsigned __int8 *)v24, 0, 0);
                sub_F54ED0((unsigned __int8 *)v24);
                v22 = *(_QWORD **)(v89 + 8);
LABEL_75:
                sub_27EC480((_QWORD *)v24, a9, a8, v57, v58, v59);
                v87 = v25;
                if ( v22 != *(_QWORD **)(v93 + 56) )
                  continue;
                goto LABEL_5;
              }
              v88 = sub_B46970((unsigned __int8 *)v24);
              if ( !v88 )
                break;
LABEL_43:
              if ( v22 == *(_QWORD **)(v93 + 56) )
                goto LABEL_5;
            }
            v82 = sub_3103290(a9);
            if ( *(_BYTE *)v24 != 63 )
            {
LABEL_78:
              v91 = 0;
              v39 = *(_QWORD *)(v24 + 16);
              goto LABEL_27;
            }
            v26 = 32LL * (*(_DWORD *)(v24 + 4) & 0x7FFFFFF);
            if ( (*(_BYTE *)(v24 + 7) & 0x40) != 0 )
            {
              v27 = *(_QWORD *)(v24 - 8);
              v28 = v27 + v26;
            }
            else
            {
              v27 = v24 - v26;
              v28 = v24;
            }
            v29 = v28 - v27;
            v95 = (unsigned __int8 **)v97;
            v30 = v29 >> 5;
            v96 = 0x400000000LL;
            v31 = v29 >> 5;
            if ( (unsigned __int64)v29 > 0x80 )
            {
              v76 = v29;
              v85 = v27;
              v92 = v29 >> 5;
              sub_C8D5F0((__int64)&v95, v97, v29 >> 5, 8u, v27, v30);
              v34 = v95;
              v33 = v96;
              LODWORD(v30) = v92;
              v27 = v85;
              v29 = v76;
              v32 = &v95[(unsigned int)v96];
            }
            else
            {
              v32 = (unsigned __int8 **)v97;
              v33 = 0;
              v34 = (unsigned __int8 **)v97;
            }
            if ( v29 > 0 )
            {
              v35 = 0;
              do
              {
                v32[v35 / 8] = *(unsigned __int8 **)(v27 + 4 * v35);
                v35 += 8LL;
                --v31;
              }
              while ( v31 );
              v34 = v95;
              v33 = v96;
            }
            LODWORD(v96) = v30 + v33;
            v36 = sub_DFCEF0(a6, (unsigned __int8 *)v24, v34, (unsigned int)(v30 + v33), 3);
            v38 = v37;
            if ( v95 != (unsigned __int8 **)v97 )
            {
              v90 = v36;
              _libc_free((unsigned __int64)v95);
              v36 = v90;
            }
            v91 = 0;
            v39 = *(_QWORD *)(v24 + 16);
            if ( !v38 && !v36 )
            {
              v61 = *(_QWORD *)(v24 + 40);
              v62 = v13 + 56;
              if ( v39 )
              {
                v84 = v24;
                v63 = v13;
                v64 = *(_QWORD *)(v24 + 16);
                v65 = v62;
                while ( 1 )
                {
                  v66 = *(_QWORD *)(v64 + 24);
                  v67 = *(_QWORD *)(v66 + 40);
                  if ( *(_BYTE *)(v63 + 84) )
                  {
                    v68 = *(__int64 **)(v63 + 64);
                    v69 = &v68[*(unsigned int *)(v63 + 76)];
                    if ( v68 != v69 )
                    {
                      while ( 1 )
                      {
                        v70 = *v68;
                        if ( v67 == *v68 )
                          break;
                        if ( v69 == ++v68 )
                          goto LABEL_96;
                      }
LABEL_94:
                      if ( v70 != v61 || (unsigned __int8)(*(_BYTE *)v66 - 61) > 1u )
                      {
                        v13 = v63;
                        v24 = v84;
                        v22 = (_QWORD *)v89;
                        goto LABEL_78;
                      }
                    }
                  }
                  else if ( sub_C8CA60(v65, v67) )
                  {
                    v70 = *(_QWORD *)(v66 + 40);
                    goto LABEL_94;
                  }
LABEL_96:
                  v64 = *(_QWORD *)(v64 + 8);
                  if ( !v64 )
                  {
                    v24 = v84;
                    v13 = v63;
                    v22 = (_QWORD *)v89;
                    v91 = 1;
                    v39 = *(_QWORD *)(v84 + 16);
                    goto LABEL_27;
                  }
                }
              }
LABEL_42:
              if ( (unsigned __int8)sub_27F1130((unsigned __int8 *)v24, a2, a4, a7, (__int64)a8, 1, a10, (__int64)a11) )
              {
                v25 = sub_27F8FB0(v24, a3, a4, a7, a9, a8, a11);
                if ( v25 )
                {
                  v87 = v88;
                  if ( !v88 )
                  {
                    v22 = *(_QWORD **)(v89 + 8);
                    sub_F54ED0((unsigned __int8 *)v24);
                    goto LABEL_75;
                  }
                }
              }
              goto LABEL_43;
            }
LABEL_27:
            if ( !v39 )
              goto LABEL_42;
            v83 = v22;
            v40 = (_BYTE *)v24;
            v41 = v13;
            while ( 1 )
            {
              v42 = *(_QWORD *)(v39 + 24);
              v43 = *(_QWORD *)(v42 + 40);
              if ( *(_BYTE *)v42 != 84 )
              {
                v45 = v41 + 56;
                goto LABEL_36;
              }
              v44 = *(_QWORD *)(v43 + 48) & 0xFFFFFFFFFFFFFFF8LL;
              if ( v44 == v43 + 48 )
                goto LABEL_115;
              if ( !v44 )
                BUG();
              if ( (unsigned int)*(unsigned __int8 *)(v44 - 24) - 30 > 0xA )
LABEL_115:
                BUG();
              if ( *(_BYTE *)(v44 - 24) == 39 )
                goto LABEL_59;
              if ( *v40 == 85 && *(_DWORD *)(v82 + 16) )
              {
                v50 = *(_DWORD *)(v82 + 24);
                v51 = *(_QWORD *)(v82 + 8);
                if ( v50 )
                {
                  v52 = (v50 - 1) & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
                  v53 = (__int64 *)(v51 + 16LL * v52);
                  v54 = *v53;
                  if ( v43 == *v53 )
                    goto LABEL_68;
                  v72 = 1;
                  while ( v54 != -4096 )
                  {
                    v74 = v72 + 1;
                    v75 = (v50 - 1) & (v52 + v72);
                    v52 = v75;
                    v53 = (__int64 *)(v51 + 16 * v75);
                    v54 = *v53;
                    if ( *v53 == v43 )
                      goto LABEL_68;
                    v72 = v74;
                  }
                }
                v53 = (__int64 *)(v51 + 16LL * v50);
LABEL_68:
                v55 = v53[1];
                v56 = v55 & 0xFFFFFFFFFFFFFFF8LL;
                if ( (v55 & 0xFFFFFFFFFFFFFFF8LL) == 0
                  || (v55 & 4) != 0 && !*(_DWORD *)(v56 + 8)
                  || ((v55 >> 2) & 1) != 0 && *(_DWORD *)(v56 + 8) != 1 )
                {
                  goto LABEL_59;
                }
              }
              v45 = v41 + 56;
              if ( a12 )
              {
                while ( 1 )
                {
                  v46 = sub_BD36B0(v42);
                  v43 = *(_QWORD *)(v42 + 40);
                  if ( !v46 || (*(_DWORD *)(v42 + 4) & 0x7FFFFFF) != 1 )
                    break;
                  if ( *(_BYTE *)(v41 + 84) )
                  {
                    v47 = *(_QWORD **)(v41 + 64);
                    v48 = &v47[*(unsigned int *)(v41 + 76)];
                    if ( v47 == v48 )
                      goto LABEL_40;
                    v49 = *(_QWORD **)(v41 + 64);
                    while ( v43 != *v49 )
                    {
                      if ( v48 == ++v49 )
                        goto LABEL_57;
                    }
                    v42 = *(_QWORD *)(*(_QWORD *)(v42 + 16) + 24LL);
                    if ( *(_BYTE *)v42 != 84 )
                    {
LABEL_61:
                      v43 = *(_QWORD *)(v42 + 40);
                      if ( !*(_BYTE *)(v41 + 84) )
                        goto LABEL_37;
                      goto LABEL_62;
                    }
                  }
                  else
                  {
                    if ( !sub_C8CA60(v41 + 56, v43) )
                      goto LABEL_61;
                    v42 = *(_QWORD *)(*(_QWORD *)(v42 + 16) + 24LL);
                    if ( *(_BYTE *)v42 != 84 )
                      goto LABEL_61;
                  }
                }
              }
LABEL_36:
              if ( *(_BYTE *)(v41 + 84) )
                break;
LABEL_37:
              if ( sub_C8CA60(v45, v43) )
              {
                if ( !v91 )
                  goto LABEL_59;
LABEL_39:
                v88 = v91;
              }
LABEL_40:
              v39 = *(_QWORD *)(v39 + 8);
              if ( !v39 )
              {
                v13 = v41;
                v24 = (__int64)v40;
                v22 = v83;
                goto LABEL_42;
              }
            }
LABEL_62:
            v47 = *(_QWORD **)(v41 + 64);
            v49 = &v47[*(unsigned int *)(v41 + 76)];
            if ( v47 == v49 )
              goto LABEL_40;
LABEL_57:
            while ( v43 != *v47 )
            {
              if ( v49 == ++v47 )
                goto LABEL_40;
            }
            if ( v91 )
              goto LABEL_39;
LABEL_59:
            v22 = v83;
            v13 = v41;
          }
          while ( v83 != *(_QWORD **)(v93 + 56) );
        }
      }
LABEL_5:
      v94 -= 8;
      if ( v80 == v94 )
        goto LABEL_100;
    }
    v60 = 1;
    while ( v20 != -4096 )
    {
      v73 = v60 + 1;
      v18 = v17 & (v60 + v18);
      v19 = (__int64 *)(v16 + 16LL * v18);
      v20 = *v19;
      if ( v14 == *v19 )
        goto LABEL_8;
      v60 = v73;
    }
LABEL_84:
    v21 = 0;
    goto LABEL_9;
  }
  v87 = 0;
LABEL_100:
  if ( byte_4F8F8E8[0] )
    nullsub_390();
  if ( v98 != &v100 )
    _libc_free((unsigned __int64)v98);
  return v87;
}
