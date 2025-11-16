// Function: sub_25D1D80
// Address: 0x25d1d80
//
void __fastcall sub_25D1D80(__int64 a1, __int64 a2, unsigned __int8 *a3, size_t a4, __int64 a5)
{
  __int64 v5; // r15
  __int64 v7; // r13
  __int64 v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // rax
  int v12; // r8d
  _BYTE *v13; // rdx
  int v14; // ecx
  int v15; // edx
  _BYTE *v16; // rax
  _QWORD *v17; // r13
  unsigned int v18; // edx
  __int64 v19; // rsi
  _QWORD *v20; // rbx
  unsigned __int64 v21; // rdi
  _QWORD *v22; // rdx
  __int64 v23; // rax
  _QWORD *v24; // r12
  _QWORD *v25; // rbx
  __int64 v26; // rax
  __int64 v27; // r13
  _QWORD *v28; // r9
  __int64 v29; // r14
  __int64 v30; // rax
  int v31; // eax
  __int64 v32; // rax
  __int64 v33; // rax
  void *v34; // rdi
  __int64 v35; // r13
  __int64 v36; // rdi
  _QWORD *v37; // r14
  _QWORD *v38; // rax
  _QWORD *v39; // r15
  __int64 v40; // rbx
  unsigned __int64 v41; // rax
  __int64 *v42; // rdx
  __int64 v43; // rax
  int v44; // edx
  __int64 v45; // rax
  __int64 *v46; // r13
  __int64 v47; // r12
  __int64 v48; // rax
  __int64 v49; // rdx
  void *v50; // rdx
  char *v51; // rax
  char *v52; // r13
  size_t v53; // rax
  void *v54; // rdi
  size_t v55; // rdx
  unsigned __int64 v56; // rax
  void *v57; // rdx
  __int64 v58; // rax
  __int64 v59; // rax
  __int64 v60; // rdx
  __int64 v61; // rdi
  signed __int64 v62; // rsi
  __int64 v63; // rax
  void *v64; // rdx
  __int64 v65; // r12
  char *v66; // r13
  size_t v67; // r8
  _QWORD *v68; // rax
  char *v69; // rax
  _BYTE *v70; // rax
  _WORD *v71; // rdx
  __int64 v72; // r8
  unsigned __int64 v73; // rdx
  unsigned __int8 *v74; // rsi
  size_t v75; // r13
  _BYTE *v76; // rax
  _BYTE *v77; // rdi
  unsigned __int64 v78; // rdx
  char *v79; // rax
  char *v80; // r13
  unsigned int v81; // esi
  unsigned int v82; // ecx
  __int64 v83; // rax
  const char *v84; // rax
  size_t v85; // rdx
  _QWORD *v88; // [rsp+18h] [rbp-898h]
  size_t v89; // [rsp+18h] [rbp-898h]
  __int64 v90; // [rsp+18h] [rbp-898h]
  __int64 v91; // [rsp+18h] [rbp-898h]
  __int64 v92; // [rsp+20h] [rbp-890h] BYREF
  _QWORD *v93; // [rsp+28h] [rbp-888h]
  __int64 v94; // [rsp+30h] [rbp-880h]
  unsigned int v95; // [rsp+38h] [rbp-878h]
  _QWORD v96[6]; // [rsp+40h] [rbp-870h] BYREF
  _BYTE *v97; // [rsp+70h] [rbp-840h] BYREF
  __int64 v98; // [rsp+78h] [rbp-838h]
  _BYTE v99[2096]; // [rsp+80h] [rbp-830h] BYREF

  v5 = a1;
  v7 = a5;
  v8 = *(_QWORD *)(a1 + 24);
  v97 = v99;
  v9 = *(_QWORD *)(a1 + 16);
  v10 = *(_QWORD *)(a1 + 8);
  v98 = 0x8000000000LL;
  v11 = *(_QWORD *)(a1 + 32);
  v96[4] = a5;
  v12 = *(_DWORD *)(a2 + 16);
  v96[0] = v8;
  v96[1] = a2;
  v96[2] = v10;
  v96[3] = v9;
  v96[5] = v11;
  v92 = 0;
  v93 = 0;
  v94 = 0;
  v95 = 0;
  if ( v12 )
  {
    v22 = *(_QWORD **)(a2 + 8);
    v23 = 2LL * *(unsigned int *)(a2 + 24);
    v24 = &v22[v23];
    if ( v22 != &v22[v23] )
    {
      while ( 1 )
      {
        v25 = v22;
        if ( *v22 <= 0xFFFFFFFFFFFFFFFDLL )
          break;
        v22 += 2;
        if ( v24 == v22 )
          goto LABEL_2;
      }
      if ( v24 != v22 )
      {
        v26 = v7;
        a1 = v22[1];
        v27 = a2;
        v28 = v96;
        v29 = v26;
        if ( !*(_BYTE *)(v8 + 336) )
          goto LABEL_32;
        while ( 1 )
        {
          if ( *(char *)(a1 + 12) < 0 )
            goto LABEL_32;
          while ( 1 )
          {
            v25 += 2;
            if ( v25 == v24 )
              goto LABEL_29;
            while ( *v25 > 0xFFFFFFFFFFFFFFFDLL )
            {
              v25 += 2;
              if ( v24 == v25 )
                goto LABEL_29;
            }
            if ( v24 == v25 )
            {
LABEL_29:
              v30 = v29;
              v13 = v97;
              a2 = v27;
              v14 = v98;
              v7 = v30;
              goto LABEL_5;
            }
            v8 = *(_QWORD *)(v5 + 24);
            a1 = v25[1];
            if ( *(_BYTE *)(v8 + 336) )
              break;
LABEL_32:
            v31 = *(_DWORD *)(a1 + 8);
            if ( !v31 )
            {
              a1 = *(_QWORD *)(a1 + 64);
              v31 = *(_DWORD *)(a1 + 8);
            }
            if ( v31 == 1 )
            {
              v88 = v28;
              sub_25D0CA0(
                a1,
                v8,
                qword_4FF14A8,
                v27,
                (__int64)&v97,
                (__int64)v28,
                v29,
                *(_QWORD *)(v5 + 32),
                (__int64)&v92);
              v28 = v88;
            }
          }
        }
      }
    }
  }
LABEL_2:
  v13 = v99;
  v14 = 0;
LABEL_5:
  v16 = &v13[16 * v14];
  while ( v14 )
  {
    a1 = *((_QWORD *)v16 - 1);
    --v14;
    v15 = *((_DWORD *)v16 - 4);
    v16 -= 16;
    LODWORD(v98) = v14;
    if ( a1 )
    {
      v8 = *(_QWORD *)(v5 + 24);
      sub_25D0CA0(a1, v8, v15, a2, (__int64)&v97, (__int64)v96, v7, *(_QWORD *)(v5 + 32), (__int64)&v92);
      v13 = v97;
      v14 = v98;
      goto LABEL_5;
    }
  }
  if ( !(_BYTE)qword_4FF0CC8 )
  {
LABEL_8:
    v17 = v93;
    v18 = v95;
    v19 = 32LL * v95;
    goto LABEL_9;
  }
  v32 = sub_C5F790(a1, v8);
  v33 = sub_904010(v32, "Missed imports into module ");
  v34 = *(void **)(v33 + 32);
  v35 = v33;
  if ( *(_QWORD *)(v33 + 24) - (_QWORD)v34 < a4 )
  {
    v35 = sub_CB6200(v33, a3, a4);
  }
  else if ( a4 )
  {
    memcpy(v34, a3, a4);
    *(_QWORD *)(v35 + 32) += a4;
  }
  sub_904010(v35, "\n");
  v36 = (unsigned int)v94;
  v17 = v93;
  v18 = v95;
  v19 = 32LL * v95;
  if ( (_DWORD)v94 )
  {
    v37 = (_QWORD *)((char *)v93 + v19);
    if ( v93 != (_QWORD *)((char *)v93 + v19) )
    {
      v38 = v93;
      while ( 1 )
      {
        v39 = v38;
        if ( *v38 <= 0xFFFFFFFFFFFFFFFDLL )
          break;
        v38 += 4;
        if ( v37 == v38 )
          goto LABEL_9;
      }
      if ( v37 != v38 )
      {
        v40 = v38[2];
        if ( !v40 )
          goto LABEL_52;
        while ( 1 )
        {
          do
          {
            v39 += 4;
            if ( v39 == v37 )
              goto LABEL_8;
            while ( *v39 > 0xFFFFFFFFFFFFFFFDLL )
            {
              v39 += 4;
              if ( v37 == v39 )
                goto LABEL_8;
            }
            if ( v37 == v39 )
              goto LABEL_8;
            v40 = v39[2];
          }
          while ( v40 );
LABEL_52:
          v41 = *(_QWORD *)v39[1] & 0xFFFFFFFFFFFFFFF8LL;
          v42 = *(__int64 **)(v41 + 24);
          if ( v42 != *(__int64 **)(v41 + 32) )
          {
            v43 = *v42;
            v44 = *(_DWORD *)(*v42 + 8);
            if ( !v44 )
            {
              v43 = *(_QWORD *)(v43 + 64);
              v44 = *(_DWORD *)(v43 + 8);
            }
            if ( v44 == 1 )
              v40 = v43;
          }
          v45 = sub_C5F790(v36, v19);
          v46 = (__int64 *)v39[1];
          v47 = v45;
          sub_CB59D0(v45, *(_QWORD *)(*v46 & 0xFFFFFFFFFFFFFFF8LL));
          v48 = *v46;
          if ( (*v46 & 1) != 0 )
            sub_BD5D20(*(_QWORD *)((v48 & 0xFFFFFFFFFFFFFFF8LL) + 8));
          else
            v49 = *(_QWORD *)((v48 & 0xFFFFFFFFFFFFFFF8LL) + 16);
          if ( v49 )
          {
            v71 = *(_WORD **)(v47 + 32);
            if ( *(_QWORD *)(v47 + 24) - (_QWORD)v71 <= 1u )
            {
              v72 = sub_CB6200(v47, (unsigned __int8 *)" (", 2u);
            }
            else
            {
              v72 = v47;
              *v71 = 10272;
              *(_QWORD *)(v47 + 32) += 2LL;
            }
            v73 = *v46 & 0xFFFFFFFFFFFFFFF8LL;
            v74 = *(unsigned __int8 **)(v73 + 8);
            if ( (*v46 & 1) != 0 )
            {
              v90 = v72;
              v84 = sub_BD5D20(*(_QWORD *)(v73 + 8));
              v72 = v90;
              v74 = (unsigned __int8 *)v84;
              v75 = v85;
            }
            else
            {
              v75 = *(_QWORD *)(v73 + 16);
            }
            v76 = *(_BYTE **)(v72 + 24);
            v77 = *(_BYTE **)(v72 + 32);
            if ( v75 > v76 - v77 )
            {
              v72 = sub_CB6200(v72, v74, v75);
              v76 = *(_BYTE **)(v72 + 24);
              v77 = *(_BYTE **)(v72 + 32);
            }
            else if ( v75 )
            {
              v91 = v72;
              memcpy(v77, v74, v75);
              v72 = v91;
              v77 = (_BYTE *)(*(_QWORD *)(v91 + 32) + v75);
              v76 = *(_BYTE **)(v91 + 24);
              *(_QWORD *)(v91 + 32) = v77;
            }
            if ( v77 == v76 )
            {
              sub_CB6200(v72, (unsigned __int8 *)")", 1u);
            }
            else
            {
              *v77 = 41;
              ++*(_QWORD *)(v72 + 32);
            }
          }
          v50 = *(void **)(v47 + 32);
          if ( *(_QWORD *)(v47 + 24) - (_QWORD)v50 <= 0xAu )
          {
            v47 = sub_CB6200(v47, ": Reason = ", 0xBu);
          }
          else
          {
            qmemcpy(v50, ": Reason = ", 11);
            *(_QWORD *)(v47 + 32) += 11LL;
          }
          v51 = sub_25CCE70(*(_DWORD *)(v39[1] + 12LL));
          v52 = v51;
          if ( !v51 )
            goto LABEL_86;
          v53 = strlen(v51);
          v54 = *(void **)(v47 + 32);
          v55 = v53;
          v56 = *(_QWORD *)(v47 + 24) - (_QWORD)v54;
          if ( v55 > v56 )
            break;
          if ( v55 )
          {
            v89 = v55;
            memcpy(v54, v52, v55);
            v57 = (void *)(*(_QWORD *)(v47 + 32) + v89);
            v58 = *(_QWORD *)(v47 + 24);
            *(_QWORD *)(v47 + 32) = v57;
            v54 = v57;
            v56 = v58 - (_QWORD)v57;
          }
          if ( v56 > 0xD )
          {
LABEL_67:
            qmemcpy(v54, ", Threshold = ", 14);
            *(_QWORD *)(v47 + 32) += 14LL;
            goto LABEL_68;
          }
LABEL_87:
          v47 = sub_CB6200(v47, ", Threshold = ", 0xEu);
LABEL_68:
          v59 = sub_CB59D0(v47, *((unsigned int *)v39 + 6));
          v60 = *(_QWORD *)(v59 + 32);
          v61 = v59;
          if ( (unsigned __int64)(*(_QWORD *)(v59 + 24) - v60) <= 8 )
          {
            v61 = sub_CB6200(v59, ", Size = ", 9u);
          }
          else
          {
            *(_BYTE *)(v60 + 8) = 32;
            *(_QWORD *)v60 = 0x3D20657A6953202CLL;
            *(_QWORD *)(v59 + 32) += 9LL;
          }
          v62 = -1;
          if ( v40 )
            v62 = *(int *)(v40 + 56);
          v63 = sub_CB59F0(v61, v62);
          v64 = *(void **)(v63 + 32);
          v65 = v63;
          if ( *(_QWORD *)(v63 + 24) - (_QWORD)v64 <= 0xEu )
          {
            v65 = sub_CB6200(v63, ", MaxHotness = ", 0xFu);
          }
          else
          {
            qmemcpy(v64, ", MaxHotness = ", 15);
            *(_QWORD *)(v63 + 32) += 15LL;
          }
          switch ( *(_BYTE *)(v39[1] + 8LL) )
          {
            case 0:
              v66 = "unknown";
              break;
            case 1:
              v66 = "cold";
              break;
            case 2:
              v66 = "none";
              break;
            case 3:
              v66 = "hot";
              break;
            case 4:
              v66 = "critical";
              break;
            default:
              BUG();
          }
          v67 = strlen(v66);
          v68 = *(_QWORD **)(v65 + 32);
          if ( *(_QWORD *)(v65 + 24) - (_QWORD)v68 < v67 )
          {
            v65 = sub_CB6200(v65, (unsigned __int8 *)v66, v67);
            v69 = *(char **)(v65 + 32);
            goto LABEL_78;
          }
          if ( (unsigned int)v67 >= 8 )
          {
            *v68 = *(_QWORD *)v66;
            *(_QWORD *)((char *)v68 + (unsigned int)v67 - 8) = *(_QWORD *)&v66[(unsigned int)v67 - 8];
            v78 = (unsigned __int64)(v68 + 1) & 0xFFFFFFFFFFFFFFF8LL;
            v79 = (char *)v68 - v78;
            v80 = (char *)(v66 - v79);
            if ( (((_DWORD)v67 + (_DWORD)v79) & 0xFFFFFFF8) >= 8 )
            {
              v81 = (v67 + (_DWORD)v79) & 0xFFFFFFF8;
              v82 = 0;
              do
              {
                v83 = v82;
                v82 += 8;
                *(_QWORD *)(v78 + v83) = *(_QWORD *)&v80[v83];
              }
              while ( v82 < v81 );
            }
LABEL_107:
            v68 = *(_QWORD **)(v65 + 32);
            goto LABEL_108;
          }
          if ( (v67 & 4) != 0 )
          {
            *(_DWORD *)v68 = *(_DWORD *)v66;
            *(_DWORD *)((char *)v68 + (unsigned int)v67 - 4) = *(_DWORD *)&v66[(unsigned int)v67 - 4];
            v68 = *(_QWORD **)(v65 + 32);
            goto LABEL_108;
          }
          if ( (_DWORD)v67 )
          {
            *(_BYTE *)v68 = *v66;
            if ( (v67 & 2) != 0 )
            {
              *(_WORD *)((char *)v68 + (unsigned int)v67 - 2) = *(_WORD *)&v66[(unsigned int)v67 - 2];
              v68 = *(_QWORD **)(v65 + 32);
              goto LABEL_108;
            }
            goto LABEL_107;
          }
LABEL_108:
          v69 = (char *)v68 + v67;
          *(_QWORD *)(v65 + 32) = v69;
LABEL_78:
          if ( *(_QWORD *)(v65 + 24) - (_QWORD)v69 <= 0xCu )
          {
            v65 = sub_CB6200(v65, ", Attempts = ", 0xDu);
          }
          else
          {
            qmemcpy(v69, ", Attempts = ", 13);
            *(_QWORD *)(v65 + 32) += 13LL;
          }
          v19 = *(unsigned int *)(v39[1] + 16LL);
          v36 = sub_CB59D0(v65, v19);
          v70 = *(_BYTE **)(v36 + 32);
          if ( *(_BYTE **)(v36 + 24) == v70 )
          {
            v19 = (__int64)"\n";
            sub_CB6200(v36, (unsigned __int8 *)"\n", 1u);
          }
          else
          {
            *v70 = 10;
            ++*(_QWORD *)(v36 + 32);
          }
        }
        v47 = sub_CB6200(v47, (unsigned __int8 *)v52, v55);
LABEL_86:
        v54 = *(void **)(v47 + 32);
        if ( *(_QWORD *)(v47 + 24) - (_QWORD)v54 > 0xDu )
          goto LABEL_67;
        goto LABEL_87;
      }
    }
  }
LABEL_9:
  if ( v18 )
  {
    v20 = (_QWORD *)((char *)v17 + v19);
    do
    {
      while ( 1 )
      {
        if ( *v17 <= 0xFFFFFFFFFFFFFFFDLL )
        {
          v21 = v17[1];
          if ( v21 )
            break;
        }
        v17 += 4;
        if ( v20 == v17 )
          goto LABEL_15;
      }
      v17 += 4;
      j_j___libc_free_0(v21);
    }
    while ( v20 != v17 );
LABEL_15:
    v17 = v93;
    v19 = 32LL * v95;
  }
  sub_C7D6A0((__int64)v17, v19, 8);
  if ( v97 != v99 )
    _libc_free((unsigned __int64)v97);
}
