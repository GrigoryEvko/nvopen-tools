// Function: sub_35E4090
// Address: 0x35e4090
//
__int64 __fastcall sub_35E4090(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v8; // rbx
  unsigned int v9; // eax
  __int64 v10; // rdx
  unsigned int v11; // eax
  __int64 v12; // rdx
  unsigned int v13; // eax
  __int64 v14; // rdx
  __int64 (*v15)(); // rax
  __int64 v16; // rdi
  __int64 (*v17)(); // rdx
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // r12
  __int64 v23; // r13
  _BYTE **v24; // r14
  _QWORD *v25; // rax
  _QWORD *v26; // rdx
  __int64 v27; // rdx
  unsigned int v28; // eax
  __int64 v29; // rdx
  unsigned int v30; // eax
  __int64 v31; // rdx
  unsigned int v32; // eax
  __int64 v33; // rdx
  char v35; // al
  __int64 v36; // rax
  _BYTE **v37; // rbx
  _BYTE *v38; // r13
  __int64 v39; // rdx
  unsigned int v40; // eax
  __int64 v41; // rdx
  __int64 v42; // r12
  __int64 (__fastcall *v43)(__int64, __int64, unsigned int, __int64); // rax
  unsigned int v44; // r9d
  __int64 v45; // rdi
  __int64 (*v46)(); // rax
  unsigned __int16 v47; // ax
  unsigned __int64 v48; // rcx
  __int64 v49; // rdx
  __int64 v50; // rcx
  unsigned __int64 v51; // rax
  unsigned __int64 v52; // rdx
  char v53; // al
  _BYTE **v54; // rdx
  __int64 *v55; // rbx
  __int64 *v56; // r12
  __int64 v57; // rdx
  _QWORD *v58; // rax
  _QWORD *v59; // rdx
  __int64 v60; // rdx
  _BYTE **v61; // r13
  __int64 v62; // rcx
  __int64 v63; // rdx
  __int64 v64; // rax
  char v65; // cl
  _QWORD **v66; // rax
  _QWORD **v67; // r13
  _QWORD *v68; // rdi
  _QWORD **v69; // rbx
  __int64 v70; // r12
  _QWORD **v71; // rax
  unsigned int v72; // eax
  __int64 v73; // rdx
  int v74; // eax
  __int64 v75; // rdx
  __int64 v76; // [rsp+0h] [rbp-100h]
  __int64 v77; // [rsp+8h] [rbp-F8h]
  unsigned __int8 v79; // [rsp+27h] [rbp-D9h]
  __int64 v80; // [rsp+28h] [rbp-D8h]
  __int64 v81; // [rsp+30h] [rbp-D0h]
  __int64 v82; // [rsp+38h] [rbp-C8h]
  unsigned int v83; // [rsp+40h] [rbp-C0h]
  unsigned __int64 v84; // [rsp+40h] [rbp-C0h]
  __int64 v85; // [rsp+40h] [rbp-C0h]
  __int64 v86; // [rsp+50h] [rbp-B0h]
  __int64 v87; // [rsp+58h] [rbp-A8h]
  __int64 v88; // [rsp+60h] [rbp-A0h]
  __int64 v89; // [rsp+68h] [rbp-98h]
  int v90; // [rsp+70h] [rbp-90h] BYREF
  __int64 v91; // [rsp+78h] [rbp-88h]
  __int64 v92; // [rsp+80h] [rbp-80h]
  __int64 v93; // [rsp+88h] [rbp-78h]
  unsigned __int64 v94; // [rsp+90h] [rbp-70h]
  __int64 v95; // [rsp+98h] [rbp-68h]
  __int64 v96; // [rsp+A0h] [rbp-60h]
  unsigned __int64 v97; // [rsp+A8h] [rbp-58h]
  __int64 v98; // [rsp+B0h] [rbp-50h] BYREF
  __int64 v99; // [rsp+B8h] [rbp-48h]
  __int64 v100; // [rsp+C0h] [rbp-40h]

  v8 = a2;
  ++*(_QWORD *)(a1 + 32);
  v86 = a1 + 32;
  if ( *(_BYTE *)(a1 + 60) )
    goto LABEL_6;
  v9 = 4 * (*(_DWORD *)(a1 + 52) - *(_DWORD *)(a1 + 56));
  v10 = *(unsigned int *)(a1 + 48);
  if ( v9 < 0x20 )
    v9 = 32;
  if ( (unsigned int)v10 <= v9 )
  {
    a2 = 0xFFFFFFFFLL;
    memset(*(void **)(a1 + 40), -1, 8 * v10);
LABEL_6:
    *(_QWORD *)(a1 + 52) = 0;
    goto LABEL_7;
  }
  sub_C8C990(v86, a2);
LABEL_7:
  ++*(_QWORD *)(a1 + 192);
  v77 = a1 + 192;
  if ( *(_BYTE *)(a1 + 220) )
  {
LABEL_12:
    *(_QWORD *)(a1 + 212) = 0;
    goto LABEL_13;
  }
  v11 = 4 * (*(_DWORD *)(a1 + 212) - *(_DWORD *)(a1 + 216));
  v12 = *(unsigned int *)(a1 + 208);
  if ( v11 < 0x20 )
    v11 = 32;
  if ( v11 >= (unsigned int)v12 )
  {
    a2 = 0xFFFFFFFFLL;
    memset(*(void **)(a1 + 200), -1, 8 * v12);
    goto LABEL_12;
  }
  sub_C8C990(v77, a2);
LABEL_13:
  ++*(_QWORD *)(a1 + 288);
  v76 = a1 + 288;
  if ( *(_BYTE *)(a1 + 316) )
  {
LABEL_18:
    *(_QWORD *)(a1 + 308) = 0;
    goto LABEL_19;
  }
  v13 = 4 * (*(_DWORD *)(a1 + 308) - *(_DWORD *)(a1 + 312));
  v14 = *(unsigned int *)(a1 + 304);
  if ( v13 < 0x20 )
    v13 = 32;
  if ( (unsigned int)v14 <= v13 )
  {
    memset(*(void **)(a1 + 296), -1, 8 * v14);
    goto LABEL_18;
  }
  sub_C8C990(v76, a2);
LABEL_19:
  v88 = sub_B2BEC0(v8);
  v15 = *(__int64 (**)())(*(_QWORD *)a3 + 16LL);
  if ( v15 == sub_23CE270 )
    BUG();
  v16 = ((__int64 (__fastcall *)(__int64, __int64))v15)(a3, v8);
  v17 = *(__int64 (**)())(*(_QWORD *)v16 + 144LL);
  v18 = 0;
  if ( v17 != sub_2C8F680 )
    v18 = ((__int64 (__fastcall *)(__int64))v17)(v16);
  *(_QWORD *)(a1 + 8) = v18;
  v19 = 0;
  v20 = sub_DFB1B0(a4);
  *(_DWORD *)(a1 + 24) = v20;
  v98 = v20;
  v99 = v21;
  v79 = 0;
  *(_QWORD *)(a1 + 16) = sub_B2BE50(v8);
  v81 = v8 + 72;
  v87 = *(_QWORD *)(v8 + 80);
  if ( v87 != v8 + 72 )
  {
    do
    {
      if ( !v87 )
        BUG();
      v22 = *(_QWORD *)(v87 + 32);
      v80 = v87 - 24;
      v89 = v87 + 24;
      if ( v22 != v87 + 24 )
      {
        while ( 1 )
        {
          v23 = v22 - 24;
          if ( !v22 )
            v23 = 0;
          v24 = (_BYTE **)v23;
          if ( *(_BYTE *)(a1 + 60) )
          {
            v25 = *(_QWORD **)(a1 + 40);
            v26 = &v25[*(unsigned int *)(a1 + 52)];
            if ( v25 == v26 )
              goto LABEL_55;
            while ( v23 != *v25 )
            {
              if ( v26 == ++v25 )
                goto LABEL_55;
            }
LABEL_32:
            v22 = *(_QWORD *)(v22 + 8);
            if ( v89 == v22 )
              break;
          }
          else
          {
            v19 = v23;
            if ( sub_C8CA60(v86, v23) )
              goto LABEL_32;
LABEL_55:
            v35 = *(_BYTE *)v23;
            if ( *(_BYTE *)v23 != 68 )
              goto LABEL_56;
            if ( (*(_BYTE *)(v23 + 7) & 0x40) != 0 )
              v54 = *(_BYTE ***)(v23 - 8);
            else
              v54 = (_BYTE **)(v23 - 32LL * (*(_DWORD *)(v23 + 4) & 0x7FFFFFF));
            if ( **v54 != 84
              || *(_BYTE *)(*(_QWORD *)(v23 + 8) + 8LL) != 12
              || *(_QWORD *)(a5 + 32) == *(_QWORD *)(a5 + 40) )
            {
              goto LABEL_32;
            }
            v55 = *(__int64 **)(a5 + 40);
            v85 = v22;
            v56 = *(__int64 **)(a5 + 32);
LABEL_84:
            v57 = *v56;
            if ( *(_BYTE *)(*v56 + 84) )
            {
              v58 = *(_QWORD **)(v57 + 64);
              v59 = &v58[*(unsigned int *)(v57 + 76)];
              if ( v58 == v59 )
                goto LABEL_125;
              while ( v80 != *v58 )
              {
                if ( v59 == ++v58 )
                  goto LABEL_125;
              }
LABEL_89:
              v19 = v88;
              v22 = v85;
              v90 = sub_2D5BAE0(*(_QWORD *)(a1 + 8), v88, *(__int64 **)(v23 + 8), 0);
              v91 = v60;
              if ( (*(_BYTE *)(v23 + 7) & 0x40) != 0 )
                v61 = *(_BYTE ***)(v23 - 8);
              else
                v61 = (_BYTE **)(v23 - 32LL * (*(_DWORD *)(v23 + 4) & 0x7FFFFFF));
              v38 = *v61;
              if ( (_WORD)v90 )
              {
                if ( (_WORD)v90 == 1 || (unsigned __int16)(v90 - 504) <= 7u )
LABEL_139:
                  BUG();
                v52 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v90 - 16];
              }
              else
              {
                v62 = sub_3007260((__int64)&v90);
                v64 = v63;
                v92 = v62;
                v52 = v62;
                v93 = v64;
              }
              if ( v52 > *(unsigned int *)(a1 + 24) )
                goto LABEL_32;
LABEL_76:
              v19 = (__int64)v38;
              v53 = sub_35E1200(a1, (__int64)v38, v52, a5);
              v22 = *(_QWORD *)(v22 + 8);
              v79 |= v53;
              if ( v89 == v22 )
                break;
            }
            else
            {
              v19 = v87 - 24;
              if ( sub_C8CA60(v57 + 56, v80) )
                goto LABEL_89;
LABEL_125:
              if ( v55 != ++v56 )
                goto LABEL_84;
              v24 = (_BYTE **)v23;
              v22 = v85;
              v35 = *(_BYTE *)v23;
LABEL_56:
              if ( v35 != 82 || sub_B532B0(*(_WORD *)(v23 + 2) & 0x3F) )
                goto LABEL_32;
              v36 = 4LL * (*(_DWORD *)(v23 + 4) & 0x7FFFFFF);
              v37 = (_BYTE **)(v23 - v36 * 8);
              if ( (*(_BYTE *)(v23 + 7) & 0x40) != 0 )
              {
                v37 = *(_BYTE ***)(v23 - 8);
                v24 = &v37[v36];
              }
              if ( v37 == v24 )
                goto LABEL_32;
              v82 = v22;
              do
              {
                v38 = *v37;
                if ( **v37 <= 0x1Cu )
                  goto LABEL_62;
                v39 = *((_QWORD *)v38 + 1);
                if ( *(_BYTE *)(v39 + 8) != 12 )
                  goto LABEL_62;
                v40 = sub_2D5BAE0(*(_QWORD *)(a1 + 8), v88, (__int64 *)v39, 0);
                v19 = *(_QWORD *)(a1 + 8);
                v42 = v41;
                if ( (_WORD)v40 )
                {
                  if ( *(_QWORD *)(v19 + 8LL * (unsigned __int16)v40 + 112) )
                    goto LABEL_62;
                }
                v83 = v40;
                sub_2FE6CC0((__int64)&v98, v19, *(_QWORD *)(a1 + 16), v40, v41);
                if ( (_BYTE)v98 != 1 )
                  goto LABEL_62;
                v43 = *(__int64 (__fastcall **)(__int64, __int64, unsigned int, __int64))(**(_QWORD **)(a1 + 8) + 592LL);
                if ( v43 == sub_2D56A50 )
                {
                  v19 = *(_QWORD *)(a1 + 8);
                  sub_2FE6CC0((__int64)&v98, v19, *(_QWORD *)(a1 + 16), v83, v42);
                  v44 = v83;
                  LOWORD(v98) = v99;
                  v99 = v100;
                }
                else
                {
                  v19 = *(_QWORD *)(a1 + 16);
                  v74 = v43(*(_QWORD *)(a1 + 8), v19, v83, v42);
                  v44 = v83;
                  LODWORD(v98) = v74;
                  v99 = v75;
                }
                v45 = *(_QWORD *)(a1 + 8);
                v46 = *(__int64 (**)())(*(_QWORD *)v45 + 1456LL);
                if ( v46 == sub_2D56680 )
                {
                  v47 = v98;
                  v48 = *(unsigned int *)(a1 + 24);
                  if ( (_WORD)v98 )
                    goto LABEL_120;
                }
                else
                {
                  v19 = v44;
                  if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD, __int64, _QWORD, __int64))v46)(
                         v45,
                         v44,
                         v42,
                         (unsigned int)v98,
                         v99) )
                  {
                    goto LABEL_62;
                  }
                  v47 = v98;
                  v48 = *(unsigned int *)(a1 + 24);
                  if ( (_WORD)v98 )
                  {
LABEL_120:
                    if ( v47 == 1 || (unsigned __int16)(v47 - 504) <= 7u )
                      goto LABEL_139;
                    v52 = *(_QWORD *)&byte_444C4A0[16 * v47 - 16];
                    if ( v52 > v48 )
                      goto LABEL_62;
                    goto LABEL_74;
                  }
                }
                v84 = v48;
                v94 = sub_3007260((__int64)&v98);
                v95 = v49;
                if ( v84 < v94 )
                  goto LABEL_62;
                v50 = sub_3007260((__int64)&v98);
                v51 = v52;
                v96 = v50;
                LODWORD(v52) = v50;
                v97 = v51;
LABEL_74:
                if ( (_DWORD)v52 )
                {
                  v22 = v82;
                  goto LABEL_76;
                }
LABEL_62:
                v37 += 4;
              }
              while ( v24 != v37 );
              v22 = *(_QWORD *)(v82 + 8);
              if ( v89 == v22 )
                break;
            }
          }
        }
      }
      v27 = *(unsigned int *)(a1 + 372);
      if ( (_DWORD)v27 != *(_DWORD *)(a1 + 376) )
      {
        v65 = *(_BYTE *)(a1 + 380);
        v66 = *(_QWORD ***)(a1 + 360);
        if ( !v65 )
          v27 = *(unsigned int *)(a1 + 368);
        v67 = &v66[v27];
        if ( v66 == v67 )
        {
LABEL_103:
          v70 = a1 + 352;
        }
        else
        {
          while ( 1 )
          {
            v68 = *v66;
            v69 = v66;
            if ( (unsigned __int64)*v66 < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( v67 == ++v66 )
              goto LABEL_103;
          }
          v70 = a1 + 352;
          if ( v66 != v67 )
          {
            do
            {
              sub_B43D60(v68);
              v71 = v69 + 1;
              if ( v69 + 1 == v67 )
                break;
              while ( 1 )
              {
                v68 = *v71;
                v69 = v71;
                if ( (unsigned __int64)*v71 < 0xFFFFFFFFFFFFFFFELL )
                  break;
                if ( v67 == ++v71 )
                  goto LABEL_108;
              }
            }
            while ( v67 != v71 );
LABEL_108:
            v65 = *(_BYTE *)(a1 + 380);
          }
        }
        ++*(_QWORD *)(a1 + 352);
        if ( !v65 )
        {
          v72 = 4 * (*(_DWORD *)(a1 + 372) - *(_DWORD *)(a1 + 376));
          v73 = *(unsigned int *)(a1 + 368);
          if ( v72 < 0x20 )
            v72 = 32;
          if ( v72 < (unsigned int)v73 )
          {
            sub_C8C990(v70, v19);
            goto LABEL_34;
          }
          v19 = 0xFFFFFFFFLL;
          memset(*(void **)(a1 + 360), -1, 8 * v73);
        }
        *(_QWORD *)(a1 + 372) = 0;
      }
LABEL_34:
      v87 = *(_QWORD *)(v87 + 8);
    }
    while ( v81 != v87 );
  }
  ++*(_QWORD *)(a1 + 32);
  if ( *(_BYTE *)(a1 + 60) )
    goto LABEL_40;
  v28 = 4 * (*(_DWORD *)(a1 + 52) - *(_DWORD *)(a1 + 56));
  v29 = *(unsigned int *)(a1 + 48);
  if ( v28 < 0x20 )
    v28 = 32;
  if ( (unsigned int)v29 <= v28 )
  {
    v19 = 0xFFFFFFFFLL;
    memset(*(void **)(a1 + 40), -1, 8 * v29);
LABEL_40:
    *(_QWORD *)(a1 + 52) = 0;
    goto LABEL_41;
  }
  sub_C8C990(v86, v19);
LABEL_41:
  ++*(_QWORD *)(a1 + 192);
  if ( *(_BYTE *)(a1 + 220) )
  {
LABEL_46:
    *(_QWORD *)(a1 + 212) = 0;
    goto LABEL_47;
  }
  v30 = 4 * (*(_DWORD *)(a1 + 212) - *(_DWORD *)(a1 + 216));
  v31 = *(unsigned int *)(a1 + 208);
  if ( v30 < 0x20 )
    v30 = 32;
  if ( (unsigned int)v31 <= v30 )
  {
    v19 = 0xFFFFFFFFLL;
    memset(*(void **)(a1 + 200), -1, 8 * v31);
    goto LABEL_46;
  }
  sub_C8C990(v77, v19);
LABEL_47:
  ++*(_QWORD *)(a1 + 288);
  if ( !*(_BYTE *)(a1 + 316) )
  {
    v32 = 4 * (*(_DWORD *)(a1 + 308) - *(_DWORD *)(a1 + 312));
    v33 = *(unsigned int *)(a1 + 304);
    if ( v32 < 0x20 )
      v32 = 32;
    if ( (unsigned int)v33 > v32 )
    {
      sub_C8C990(v76, v19);
      return v79;
    }
    memset(*(void **)(a1 + 296), -1, 8 * v33);
  }
  *(_QWORD *)(a1 + 308) = 0;
  return v79;
}
