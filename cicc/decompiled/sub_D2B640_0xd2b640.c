// Function: sub_D2B640
// Address: 0xd2b640
//
__int64 __fastcall sub_D2B640(__int64 a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v6; // rax
  char *v7; // rsi
  __int64 v8; // rsi
  unsigned __int64 v9; // r13
  __int64 v10; // r12
  __int64 v11; // r15
  unsigned __int8 *v12; // rax
  size_t v13; // rdx
  void *v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // r13
  _QWORD *v20; // rax
  _BYTE *v21; // rdx
  _BYTE *v22; // rbx
  _BYTE *v23; // r15
  __int64 v24; // r13
  _DWORD *v25; // rdx
  __int64 v26; // r14
  _DWORD *v27; // rdx
  char *v28; // rsi
  __int64 v29; // rax
  _DWORD *v30; // rdx
  const char *v31; // rax
  size_t v32; // rdx
  _BYTE *v33; // rdi
  unsigned __int8 *v34; // rsi
  _BYTE *v35; // rax
  int v36; // ecx
  __int64 *v37; // rsi
  __int64 i; // r15
  unsigned int v39; // edx
  __int64 *v40; // rax
  __int64 v41; // r11
  int v42; // eax
  unsigned int v43; // r9d
  __int64 v44; // r8
  __int64 v46; // rax
  __int64 v47; // r12
  void *v48; // rdx
  __int64 v49; // rdi
  __int64 v50; // rax
  void *v51; // rdx
  __int64 *v52; // r14
  __int64 v53; // r13
  __int64 v54; // rax
  __int64 v55; // rax
  __int64 *v56; // rbx
  __int64 *v57; // r13
  __int64 v58; // rdx
  __int64 v59; // r15
  __int64 v60; // r14
  const char *v61; // rax
  size_t v62; // rdx
  _BYTE *v63; // rdi
  unsigned __int8 *v64; // rsi
  _BYTE *v65; // rax
  size_t v66; // r15
  _BYTE *v67; // rax
  unsigned int v68; // edx
  __int64 v69; // rsi
  unsigned int v70; // ecx
  __int64 *v71; // rax
  __int64 v72; // r8
  int v73; // ecx
  int v74; // eax
  __int64 v75; // rdi
  unsigned int v76; // r8d
  unsigned int v77; // r10d
  __int64 *v78; // rax
  __int64 v79; // r11
  int v80; // eax
  int v81; // eax
  int v82; // ebx
  int v83; // eax
  int v84; // eax
  int v85; // ebx
  int v86; // r9d
  __int64 v88; // [rsp+10h] [rbp-60h]
  __int64 v89; // [rsp+18h] [rbp-58h]
  __int64 v90; // [rsp+18h] [rbp-58h]
  __int64 v92; // [rsp+28h] [rbp-48h]
  __int64 v93; // [rsp+30h] [rbp-40h]
  __int64 *v94; // [rsp+30h] [rbp-40h]
  size_t v95; // [rsp+38h] [rbp-38h]
  __int64 *v96; // [rsp+38h] [rbp-38h]
  size_t v97; // [rsp+38h] [rbp-38h]

  v92 = sub_BC0510(a4, &unk_4F86C48, a3);
  v89 = v92 + 8;
  v5 = sub_904010(*a2, "Printing the call graph for module: ");
  v6 = sub_CB6200(v5, *(unsigned __int8 **)(a3 + 168), *(_QWORD *)(a3 + 176));
  v7 = "\n\n";
  sub_904010(v6, "\n\n");
  v88 = a3 + 24;
  v93 = *(_QWORD *)(a3 + 32);
  if ( v93 != a3 + 24 )
  {
    do
    {
      v8 = v93 - 56;
      if ( !v93 )
        v8 = 0;
      v9 = sub_D29010(v89, v8);
      v10 = *a2;
      v11 = sub_904010(*a2, "  Edges in function: ");
      v12 = (unsigned __int8 *)sub_BD5D20(*(_QWORD *)(v9 + 8));
      v14 = *(void **)(v11 + 32);
      if ( *(_QWORD *)(v11 + 24) - (_QWORD)v14 < v13 )
      {
        v11 = sub_CB6200(v11, v12, v13);
      }
      else if ( v13 )
      {
        v97 = v13;
        memcpy(v14, v12, v13);
        *(_QWORD *)(v11 + 32) += v97;
      }
      sub_904010(v11, "\n");
      if ( *(_BYTE *)(v9 + 104) )
        v19 = v9 + 24;
      else
        v19 = sub_D29180(v9, (__int64)"\n", v15, v16, v17, v18);
      v20 = sub_D23BF0(v19);
      v22 = v21;
      v23 = v20;
      v24 = sub_D23C30(v19);
      while ( (_BYTE *)v24 != v23 )
      {
LABEL_13:
        v25 = *(_DWORD **)(v10 + 32);
        if ( *(_QWORD *)(v10 + 24) - (_QWORD)v25 <= 3u )
        {
          v46 = sub_CB6200(v10, (unsigned __int8 *)"    ", 4u);
          v27 = *(_DWORD **)(v46 + 32);
          v26 = v46;
        }
        else
        {
          *v25 = 538976288;
          v26 = v10;
          v27 = (_DWORD *)(*(_QWORD *)(v10 + 32) + 4LL);
          *(_QWORD *)(v10 + 32) = v27;
        }
        v28 = "ref ";
        if ( (*v23 & 4) != 0 )
          v28 = "call";
        if ( *(_QWORD *)(v26 + 24) - (_QWORD)v27 > 3u )
        {
          *v27 = *(_DWORD *)v28;
          v30 = (_DWORD *)(*(_QWORD *)(v26 + 32) + 4LL);
          *(_QWORD *)(v26 + 32) = v30;
        }
        else
        {
          v29 = sub_CB6200(v26, (unsigned __int8 *)v28, 4u);
          v30 = *(_DWORD **)(v29 + 32);
          v26 = v29;
        }
        if ( *(_QWORD *)(v26 + 24) - (_QWORD)v30 <= 3u )
        {
          v26 = sub_CB6200(v26, (unsigned __int8 *)" -> ", 4u);
        }
        else
        {
          *v30 = 540945696;
          *(_QWORD *)(v26 + 32) += 4LL;
        }
        v31 = sub_BD5D20(*(_QWORD *)((*(_QWORD *)v23 & 0xFFFFFFFFFFFFFFF8LL) + 8));
        v33 = *(_BYTE **)(v26 + 32);
        v34 = (unsigned __int8 *)v31;
        v35 = *(_BYTE **)(v26 + 24);
        if ( v35 - v33 < v32 )
        {
          v26 = sub_CB6200(v26, v34, v32);
          v35 = *(_BYTE **)(v26 + 24);
          v33 = *(_BYTE **)(v26 + 32);
        }
        else if ( v32 )
        {
          v95 = v32;
          memcpy(v33, v34, v32);
          v35 = *(_BYTE **)(v26 + 24);
          v33 = (_BYTE *)(v95 + *(_QWORD *)(v26 + 32));
          *(_QWORD *)(v26 + 32) = v33;
        }
        if ( v35 == v33 )
        {
          sub_CB6200(v26, (unsigned __int8 *)"\n", 1u);
        }
        else
        {
          *v33 = 10;
          ++*(_QWORD *)(v26 + 32);
        }
        v23 += 8;
        if ( v22 != v23 )
        {
          while ( (*(_QWORD *)v23 & 0xFFFFFFFFFFFFFFF8LL) == 0 || !*(_QWORD *)(*(_QWORD *)v23 & 0xFFFFFFFFFFFFFFF8LL) )
          {
            v23 += 8;
            if ( v23 == v22 )
            {
              if ( (_BYTE *)v24 != v23 )
                goto LABEL_13;
              goto LABEL_30;
            }
          }
        }
      }
LABEL_30:
      v7 = "\n";
      sub_904010(v10, "\n");
      v93 = *(_QWORD *)(v93 + 8);
    }
    while ( v88 != v93 );
  }
  sub_D2AD40(v89, (__int64 *)v7);
  v36 = *(_DWORD *)(v92 + 448);
  if ( v36 )
  {
    v37 = *(__int64 **)(v92 + 440);
    for ( i = *v37; i; i = v37[v42] )
    {
      if ( *(_DWORD *)(i + 16) )
      {
LABEL_46:
        v47 = *a2;
        v48 = *(void **)(*a2 + 32);
        if ( *(_QWORD *)(*a2 + 24) - (_QWORD)v48 <= 0xDu )
        {
          v49 = sub_CB6200(*a2, "  RefSCC with ", 0xEu);
        }
        else
        {
          v49 = *a2;
          qmemcpy(v48, "  RefSCC with ", 14);
          *(_QWORD *)(v47 + 32) += 14LL;
        }
        v50 = sub_CB59F0(v49, *(unsigned int *)(i + 16));
        v51 = *(void **)(v50 + 32);
        if ( *(_QWORD *)(v50 + 24) - (_QWORD)v51 <= 0xBu )
        {
          sub_CB6200(v50, " call SCCs:\n", 0xCu);
        }
        else
        {
          qmemcpy(v51, " call SCCs:\n", 12);
          *(_QWORD *)(v50 + 32) += 12LL;
        }
        v52 = *(__int64 **)(i + 8);
        v96 = v52;
        v94 = &v52[*(unsigned int *)(i + 16)];
        if ( v52 != v94 )
        {
          v90 = i;
          do
          {
            v53 = *v96;
            v54 = sub_904010(v47, "    SCC with ");
            v55 = sub_CB59F0(v54, *(int *)(v53 + 16));
            sub_904010(v55, " functions:\n");
            v56 = *(__int64 **)(v53 + 8);
            v57 = &v56[*(unsigned int *)(v53 + 16)];
            while ( v57 != v56 )
            {
              while ( 1 )
              {
                v58 = *(_QWORD *)(v47 + 32);
                v59 = *v56;
                if ( (unsigned __int64)(*(_QWORD *)(v47 + 24) - v58) <= 5 )
                {
                  v60 = sub_CB6200(v47, (unsigned __int8 *)"      ", 6u);
                }
                else
                {
                  *(_DWORD *)v58 = 538976288;
                  v60 = v47;
                  *(_WORD *)(v58 + 4) = 8224;
                  *(_QWORD *)(v47 + 32) += 6LL;
                }
                v61 = sub_BD5D20(*(_QWORD *)(v59 + 8));
                v63 = *(_BYTE **)(v60 + 32);
                v64 = (unsigned __int8 *)v61;
                v65 = *(_BYTE **)(v60 + 24);
                v66 = v62;
                if ( v65 - v63 < v62 )
                {
                  v60 = sub_CB6200(v60, v64, v62);
                  v65 = *(_BYTE **)(v60 + 24);
                  v63 = *(_BYTE **)(v60 + 32);
                }
                else if ( v62 )
                {
                  memcpy(v63, v64, v62);
                  v65 = *(_BYTE **)(v60 + 24);
                  v63 = (_BYTE *)(v66 + *(_QWORD *)(v60 + 32));
                  *(_QWORD *)(v60 + 32) = v63;
                }
                if ( v65 == v63 )
                  break;
                ++v56;
                *v63 = 10;
                ++*(_QWORD *)(v60 + 32);
                if ( v57 == v56 )
                  goto LABEL_62;
              }
              ++v56;
              sub_CB6200(v60, (unsigned __int8 *)"\n", 1u);
            }
LABEL_62:
            ++v96;
          }
          while ( v94 != v96 );
          i = v90;
        }
        v67 = *(_BYTE **)(v47 + 32);
        if ( *(_BYTE **)(v47 + 24) == v67 )
        {
          sub_CB6200(v47, (unsigned __int8 *)"\n", 1u);
        }
        else
        {
          *v67 = 10;
          ++*(_QWORD *)(v47 + 32);
        }
        v68 = *(_DWORD *)(v92 + 608);
        v69 = *(_QWORD *)(v92 + 592);
        if ( v68 )
        {
          v70 = (v68 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
          v71 = (__int64 *)(v69 + 16LL * v70);
          v72 = *v71;
          if ( *v71 == i )
          {
LABEL_68:
            v73 = *(_DWORD *)(v92 + 448);
            v74 = *((_DWORD *)v71 + 2) + 1;
            if ( v74 == v73 )
              break;
            v75 = *(_QWORD *)(v92 + 440);
            i = *(_QWORD *)(v75 + 8LL * v74);
            if ( !i )
              break;
            v76 = v68 - 1;
LABEL_74:
            if ( *(_DWORD *)(i + 16) )
              goto LABEL_46;
            if ( v68 )
            {
              v77 = v76 & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
              v78 = (__int64 *)(v69 + 16LL * v77);
              v79 = *v78;
              if ( i != *v78 )
              {
                v81 = 1;
                while ( v79 != -4096 )
                {
                  v82 = v81 + 1;
                  v77 = v76 & (v81 + v77);
                  v78 = (__int64 *)(v69 + 16LL * v77);
                  v79 = *v78;
                  if ( i == *v78 )
                    goto LABEL_72;
                  v81 = v82;
                }
                goto LABEL_76;
              }
            }
            else
            {
LABEL_76:
              v78 = (__int64 *)(v69 + 16LL * v68);
            }
LABEL_72:
            v80 = *((_DWORD *)v78 + 2) + 1;
            if ( v73 == v80 )
              break;
            i = *(_QWORD *)(v75 + 8LL * v80);
            if ( !i )
              break;
            goto LABEL_74;
          }
          v83 = 1;
          while ( v72 != -4096 )
          {
            v86 = v83 + 1;
            v70 = (v68 - 1) & (v83 + v70);
            v71 = (__int64 *)(v69 + 16LL * v70);
            v72 = *v71;
            if ( *v71 == i )
              goto LABEL_68;
            v83 = v86;
          }
        }
        v71 = (__int64 *)(v69 + 16LL * v68);
        goto LABEL_68;
      }
      v43 = *(_DWORD *)(v92 + 608);
      v44 = *(_QWORD *)(v92 + 592);
      if ( v43 )
      {
        v39 = (v43 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
        v40 = (__int64 *)(v44 + 16LL * v39);
        v41 = *v40;
        if ( *v40 == i )
        {
LABEL_35:
          v42 = *((_DWORD *)v40 + 2) + 1;
          if ( v42 == v36 )
            break;
          continue;
        }
        v84 = 1;
        while ( v41 != -4096 )
        {
          v85 = v84 + 1;
          v39 = (v43 - 1) & (v84 + v39);
          v40 = (__int64 *)(v44 + 16LL * v39);
          v41 = *v40;
          if ( *v40 == i )
            goto LABEL_35;
          v84 = v85;
        }
      }
      v42 = *(_DWORD *)(v44 + 16LL * v43 + 8) + 1;
      if ( v42 == v36 )
        break;
    }
  }
  *(_BYTE *)(a1 + 28) = 1;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 16) = 2;
  *(_DWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_QWORD *)(a1 + 64) = 2;
  *(_DWORD *)(a1 + 72) = 0;
  *(_BYTE *)(a1 + 76) = 1;
  sub_AE6EC0(a1, (__int64)&unk_4F82400);
  return a1;
}
