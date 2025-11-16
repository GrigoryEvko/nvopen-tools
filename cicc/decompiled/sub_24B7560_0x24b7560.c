// Function: sub_24B7560
// Address: 0x24b7560
//
__int64 __fastcall sub_24B7560(__int64 a1, __int64 **a2, void **a3, __int64 a4, void **a5, __int64 a6)
{
  unsigned __int8 *v8; // rdi
  __int64 v9; // rcx
  size_t v10; // rsi
  __int64 v11; // rdx
  char *v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned __int64 v19; // rax
  int v20; // eax
  __int64 v21; // r14
  unsigned __int64 v22; // rbx
  char v23; // r15
  unsigned __int8 **v24; // rdi
  __int64 v25; // r15
  _BYTE *v26; // rax
  _WORD *v27; // rdx
  unsigned int v28; // eax
  size_t v29; // rdx
  size_t v30; // r12
  unsigned int v31; // ebx
  __int64 (__fastcall **v32)(); // rax
  void *v33; // rax
  __int64 v34; // rax
  void *v35; // rax
  __int64 v36; // rax
  void *v37; // rax
  __int64 v38; // rax
  __int64 v39; // rax
  const char *v40; // rsi
  unsigned __int8 **v42; // rsi
  __int64 v43; // r13
  __int64 v44; // rax
  __int64 v45; // r13
  size_t v46; // rdx
  unsigned __int8 *v47; // rsi
  __int64 v48; // rdi
  __int64 v49; // rax
  void *v50; // rax
  __int64 v51; // rax
  __int64 v52; // rax
  size_t v53; // rdx
  unsigned __int64 v54; // rax
  __int64 v55; // r13
  int v56; // r15d
  unsigned int v57; // ebx
  _QWORD *v58; // rdi
  __int64 v59; // rax
  _QWORD *v60; // rdi
  unsigned __int64 v61; // r12
  size_t v62; // r14
  bool v63; // zf
  int v64; // r14d
  char *v65; // rdx
  void *v66; // rax
  __int64 v67; // rax
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 v70; // rax
  __int64 v71; // rax
  unsigned int v72; // ebx
  __int64 i; // rax
  _QWORD *v74; // rdi
  __int64 v75; // rax
  _QWORD *v76; // rdi
  unsigned __int64 v77; // r12
  size_t v78; // r15
  char *v79; // rdx
  int v80; // r15d
  __int64 v81; // rax
  __int64 v82; // rax
  __int64 v83; // r12
  __int64 v86; // [rsp+20h] [rbp-1F0h]
  __int64 v88; // [rsp+48h] [rbp-1C8h]
  unsigned __int64 v89; // [rsp+68h] [rbp-1A8h]
  int v90; // [rsp+70h] [rbp-1A0h]
  unsigned int v91; // [rsp+9Ch] [rbp-174h] BYREF
  __int64 *v92; // [rsp+A0h] [rbp-170h] BYREF
  __int64 v93; // [rsp+A8h] [rbp-168h]
  __int64 v94; // [rsp+B0h] [rbp-160h] BYREF
  unsigned __int8 *v95; // [rsp+C0h] [rbp-150h]
  size_t v96; // [rsp+C8h] [rbp-148h]
  _BYTE v97[16]; // [rsp+D0h] [rbp-140h] BYREF
  unsigned __int8 *v98; // [rsp+E0h] [rbp-130h] BYREF
  size_t v99; // [rsp+E8h] [rbp-128h]
  _BYTE v100[16]; // [rsp+F0h] [rbp-120h] BYREF
  __int64 v101[2]; // [rsp+100h] [rbp-110h] BYREF
  _QWORD v102[2]; // [rsp+110h] [rbp-100h] BYREF
  unsigned __int8 *v103; // [rsp+120h] [rbp-F0h] BYREF
  size_t v104; // [rsp+128h] [rbp-E8h]
  _QWORD v105[2]; // [rsp+130h] [rbp-E0h] BYREF
  unsigned __int8 *v106; // [rsp+140h] [rbp-D0h] BYREF
  size_t n; // [rsp+148h] [rbp-C8h]
  __int64 v108; // [rsp+150h] [rbp-C0h] BYREF
  _BYTE *v109; // [rsp+158h] [rbp-B8h]
  _BYTE *v110; // [rsp+160h] [rbp-B0h]
  __int64 v111; // [rsp+168h] [rbp-A8h]
  unsigned __int8 **v112; // [rsp+170h] [rbp-A0h]
  unsigned __int8 **v113; // [rsp+180h] [rbp-90h] BYREF
  __int64 (__fastcall **v114)(); // [rsp+188h] [rbp-88h]
  char *v115; // [rsp+198h] [rbp-78h]
  char *v116; // [rsp+1A0h] [rbp-70h]

  if ( !*(_QWORD *)(a6 + 8) )
  {
    sub_CA0F50((__int64 *)&v103, a3);
    LOWORD(v116) = 260;
    v113 = &v103;
    sub_C67360((__int64 *)&v106, (__int64)&v113, &v91);
    v8 = *(unsigned __int8 **)a6;
    if ( v106 == (unsigned __int8 *)&v108 )
    {
      v53 = n;
      if ( n )
      {
        if ( n == 1 )
          *v8 = v108;
        else
          memcpy(v8, &v108, n);
        v53 = n;
        v8 = *(unsigned __int8 **)a6;
      }
      *(_QWORD *)(a6 + 8) = v53;
      v8[v53] = 0;
      v8 = v106;
      goto LABEL_6;
    }
    v9 = v108;
    v10 = n;
    if ( v8 == (unsigned __int8 *)(a6 + 16) )
    {
      *(_QWORD *)a6 = v106;
      *(_QWORD *)(a6 + 8) = v10;
      *(_QWORD *)(a6 + 16) = v9;
    }
    else
    {
      v11 = *(_QWORD *)(a6 + 16);
      *(_QWORD *)a6 = v106;
      *(_QWORD *)(a6 + 8) = v10;
      *(_QWORD *)(a6 + 16) = v9;
      if ( v8 )
      {
        v106 = v8;
        v108 = v11;
LABEL_6:
        n = 0;
        *v8 = 0;
        if ( v106 != (unsigned __int8 *)&v108 )
          j_j___libc_free_0((unsigned __int64)v106);
        if ( v103 != (unsigned __int8 *)v105 )
          j_j___libc_free_0((unsigned __int64)v103);
        goto LABEL_10;
      }
    }
    v106 = (unsigned __int8 *)&v108;
    v8 = (unsigned __int8 *)&v108;
    goto LABEL_6;
  }
  LOWORD(v116) = 260;
  v113 = (unsigned __int8 **)a6;
  v28 = sub_C83360((__int64)&v113, (int *)&v91, 0, 2, 1, 0x1B6u);
  v30 = v29;
  n = v29;
  v31 = v28;
  LODWORD(v106) = v28;
  v32 = sub_2241E50();
  LODWORD(v113) = 17;
  v114 = v32;
  if ( (*(unsigned __int8 (__fastcall **)(size_t, _QWORD, unsigned __int8 ***))(*(_QWORD *)v30 + 48LL))(v30, v31, &v113)
    || (*((unsigned __int8 (__fastcall **)(__int64 (__fastcall **)(), unsigned __int8 **, _QWORD))*v114 + 7))(
         v114,
         &v106,
         (unsigned int)v113) )
  {
    v33 = sub_CB72A0();
    v34 = sub_904010((__int64)v33, "file exists, overwriting");
    sub_904010(v34, "\n");
  }
  else
  {
    if ( (_DWORD)v106 )
    {
      v35 = sub_CB72A0();
      v36 = sub_904010((__int64)v35, "error writing into file");
      sub_904010(v36, "\n");
      sub_24A46A0((__int64 *)a1, byte_3F871B3);
      return a1;
    }
    v50 = sub_CB72A0();
    v51 = sub_904010((__int64)v50, "writing to the newly created file ");
    v52 = sub_CB6200(v51, *(unsigned __int8 **)a6, *(_QWORD *)(a6 + 8));
    sub_904010(v52, "\n");
  }
LABEL_10:
  sub_CB6EE0((__int64)&v113, v91, 1, 0, 0);
  if ( v91 == -1 )
  {
    v37 = sub_CB72A0();
    v38 = sub_904010((__int64)v37, "error opening file '");
    v39 = sub_CB6200(v38, *(unsigned __int8 **)a6, *(_QWORD *)(a6 + 8));
    v40 = "' for writing!\n";
    sub_904010(v39, "' for writing!\n");
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)a1 = a1 + 16;
    *(_BYTE *)(a1 + 16) = 0;
    goto LABEL_58;
  }
  sub_CA0F50((__int64 *)&v92, a5);
  v12 = (char *)sub_BD5D20(**a2);
  v103 = (unsigned __int8 *)v105;
  sub_24A2F70((__int64 *)&v103, v12, (__int64)&v12[v13]);
  if ( v93 )
  {
    v42 = (unsigned __int8 **)&v92;
    v43 = sub_904010((__int64)&v113, "digraph \"");
  }
  else
  {
    if ( !v104 )
    {
      sub_904010((__int64)&v113, "digraph unnamed {\n");
      goto LABEL_14;
    }
    v42 = &v103;
    v43 = sub_904010((__int64)&v113, "digraph \"");
  }
  sub_C67200((__int64 *)&v106, (__int64)v42);
  v44 = sub_CB6200(v43, v106, n);
  sub_904010(v44, "\" {\n");
  if ( v106 != (unsigned __int8 *)&v108 )
  {
    j_j___libc_free_0((unsigned __int64)v106);
    if ( !v93 )
      goto LABEL_15;
LABEL_63:
    v45 = sub_904010((__int64)&v113, "\tlabel=\"");
    sub_C67200((__int64 *)&v106, (__int64)&v92);
    v46 = n;
    v47 = v106;
    v48 = v45;
    goto LABEL_64;
  }
LABEL_14:
  if ( v93 )
    goto LABEL_63;
LABEL_15:
  if ( !v104 )
    goto LABEL_16;
  v83 = sub_904010((__int64)&v113, "\tlabel=\"");
  sub_C67200((__int64 *)&v106, (__int64)&v103);
  v46 = n;
  v47 = v106;
  v48 = v83;
LABEL_64:
  v49 = sub_CB6200(v48, v47, v46);
  sub_904010(v49, "\";\n");
  if ( v106 != (unsigned __int8 *)&v108 )
    j_j___libc_free_0((unsigned __int64)v106);
LABEL_16:
  n = 0;
  LOBYTE(v108) = 0;
  v106 = (unsigned __int8 *)&v108;
  sub_CB6200((__int64)&v113, (unsigned __int8 *)&v108, 0);
  if ( v106 != (unsigned __int8 *)&v108 )
    j_j___libc_free_0((unsigned __int64)v106);
  if ( v115 == v116 )
    sub_CB6200((__int64)&v113, (unsigned __int8 *)"\n", 1u);
  else
    *v116++ = 10;
  if ( v103 != (unsigned __int8 *)v105 )
    j_j___libc_free_0((unsigned __int64)v103);
  v14 = **a2;
  v86 = v14 + 72;
  v88 = *(_QWORD *)(v14 + 80);
  if ( v14 + 72 == v88 )
    goto LABEL_107;
  do
  {
    v96 = 0;
    v97[0] = 0;
    v15 = v88 - 24;
    if ( !v88 )
      v15 = 0;
    v89 = v15;
    v16 = v15;
    v95 = v97;
    v17 = sub_904010((__int64)&v113, "\tNode");
    v18 = sub_CB5A80(v17, v16);
    sub_904010(v18, " [shape=");
    sub_904010((__int64)&v113, "record,");
    if ( v96 )
    {
      v71 = sub_CB6200((__int64)&v113, v95, v96);
      sub_904010(v71, ",");
    }
    sub_904010((__int64)&v113, "label=");
    sub_904010((__int64)&v113, "\"{");
    sub_24A8B90((__int64)&v103, v89, (__int64)*a2);
    sub_C67200((__int64 *)&v106, (__int64)&v103);
    sub_CB6200((__int64)&v113, v106, n);
    if ( v106 != (unsigned __int8 *)&v108 )
      j_j___libc_free_0((unsigned __int64)v106);
    if ( v103 != (unsigned __int8 *)v105 )
      j_j___libc_free_0((unsigned __int64)v103);
    v99 = 0;
    v98 = v100;
    v111 = 0x100000000LL;
    v100[0] = 0;
    n = 0;
    v106 = (unsigned __int8 *)&unk_49DD210;
    v108 = 0;
    v112 = &v98;
    v109 = 0;
    v110 = 0;
    sub_CB5980((__int64)&v106, 0, 0, 0);
    v19 = *(_QWORD *)(v89 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v89 + 48 == v19 )
      goto LABEL_75;
    if ( !v19 )
      goto LABEL_117;
    if ( (unsigned int)*(unsigned __int8 *)(v19 - 24) - 30 > 0xA )
      goto LABEL_75;
    v20 = sub_B46E30(v19 - 24);
    if ( !v20 )
      goto LABEL_75;
    v21 = (unsigned int)(v20 - 1);
    v22 = 0;
    v23 = 0;
    do
    {
      v101[0] = (__int64)v102;
      sub_24A2F70(v101, byte_3F871B3, (__int64)byte_3F871B3);
      if ( v101[1] )
      {
        v27 = v110;
        if ( (_DWORD)v22 )
        {
          if ( v110 != v109 )
          {
            *v110 = 124;
            v27 = v110 + 1;
            v110 = v27;
            if ( (unsigned __int64)(v109 - (_BYTE *)v27) <= 1 )
              goto LABEL_51;
            goto LABEL_37;
          }
          sub_CB6200((__int64)&v106, (unsigned __int8 *)"|", 1u);
          v27 = v110;
        }
        if ( (unsigned __int64)(v109 - (_BYTE *)v27) <= 1 )
        {
LABEL_51:
          v24 = (unsigned __int8 **)sub_CB6200((__int64)&v106, "<s", 2u);
          goto LABEL_38;
        }
LABEL_37:
        v24 = &v106;
        *v27 = 29500;
        v110 += 2;
LABEL_38:
        v25 = sub_CB59D0((__int64)v24, v22);
        v26 = *(_BYTE **)(v25 + 32);
        if ( *(_BYTE **)(v25 + 24) == v26 )
        {
          v25 = sub_CB6200(v25, (unsigned __int8 *)">", 1u);
        }
        else
        {
          *v26 = 62;
          ++*(_QWORD *)(v25 + 32);
        }
        sub_C67200((__int64 *)&v103, (__int64)v101);
        sub_CB6200(v25, v103, v104);
        if ( v103 != (unsigned __int8 *)v105 )
          j_j___libc_free_0((unsigned __int64)v103);
        if ( (_QWORD *)v101[0] != v102 )
          j_j___libc_free_0(v101[0]);
        v23 = 1;
LABEL_45:
        if ( v21 == v22 )
          goto LABEL_74;
        goto LABEL_46;
      }
      if ( (_QWORD *)v101[0] == v102 )
        goto LABEL_45;
      j_j___libc_free_0(v101[0]);
      if ( v21 == v22 )
      {
LABEL_74:
        if ( !v23 )
          goto LABEL_75;
        goto LABEL_116;
      }
LABEL_46:
      ++v22;
    }
    while ( v22 != 64 );
    if ( !v23 )
      goto LABEL_75;
    sub_904010((__int64)&v106, "|<s64>truncated...");
LABEL_116:
    sub_904010((__int64)&v113, "|");
    v69 = sub_904010((__int64)&v113, "{");
    v70 = sub_CB6200(v69, v98, v99);
    sub_904010(v70, "}");
LABEL_75:
    sub_904010((__int64)&v113, "}\"");
    sub_904010((__int64)&v113, "];\n");
    v54 = *(_QWORD *)(v89 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v89 + 48 != v54 )
    {
      if ( v54 )
      {
        v55 = v54 - 24;
        if ( (unsigned int)*(unsigned __int8 *)(v54 - 24) - 30 <= 0xA )
        {
          v56 = sub_B46E30(v55);
          if ( v56 )
          {
            v90 = v56;
            v57 = 0;
            while ( 1 )
            {
              v61 = sub_B46EC0(v55, v57);
              if ( v61 )
              {
                v103 = (unsigned __int8 *)v105;
                sub_24A2F70((__int64 *)&v103, byte_3F871B3, (__int64)byte_3F871B3);
                v62 = v104;
                if ( v103 != (unsigned __int8 *)v105 )
                  j_j___libc_free_0((unsigned __int64)v103);
                v63 = v62 == 0;
                v64 = -1;
                if ( !v63 )
                  v64 = v57;
                v103 = (unsigned __int8 *)v105;
                sub_24A2F70((__int64 *)&v103, byte_3F871B3, (__int64)byte_3F871B3);
                v65 = v116;
                if ( (unsigned __int64)(v115 - v116) > 4 )
                {
                  *(_DWORD *)v116 = 1685016073;
                  v58 = &v113;
                  v65[4] = 101;
                  v116 += 5;
                }
                else
                {
                  v58 = (_QWORD *)sub_CB6200((__int64)&v113, "\tNode", 5u);
                }
                sub_CB5A80((__int64)v58, v89);
                if ( v64 != -1 )
                {
                  v59 = sub_904010((__int64)&v113, ":s");
                  sub_CB59F0(v59, v64);
                }
                if ( (unsigned __int64)(v115 - v116) <= 7 )
                {
                  v60 = (_QWORD *)sub_CB6200((__int64)&v113, " -> Node", 8u);
                }
                else
                {
                  v60 = &v113;
                  *(_QWORD *)v116 = 0x65646F4E203E2D20LL;
                  v116 += 8;
                }
                sub_CB5A80((__int64)v60, v61);
                if ( v104 )
                {
                  v67 = sub_904010((__int64)&v113, "[");
                  v68 = sub_CB6200(v67, v103, v104);
                  sub_904010(v68, "]");
                }
                if ( (unsigned __int64)(v115 - v116) <= 1 )
                {
                  sub_CB6200((__int64)&v113, (unsigned __int8 *)";\n", 2u);
                }
                else
                {
                  *(_WORD *)v116 = 2619;
                  v116 += 2;
                }
                if ( v103 != (unsigned __int8 *)v105 )
                  j_j___libc_free_0((unsigned __int64)v103);
              }
              if ( ++v57 == v56 )
                break;
              if ( v57 == 64 )
              {
                v72 = 64;
                for ( i = sub_B46EC0(v55, 0x40u); ; i = sub_B46EC0(v55, v72) )
                {
                  v77 = i;
                  if ( i )
                  {
                    v103 = (unsigned __int8 *)v105;
                    sub_24A2F70((__int64 *)&v103, byte_3F871B3, (__int64)byte_3F871B3);
                    v78 = v104;
                    if ( v103 != (unsigned __int8 *)v105 )
                      j_j___libc_free_0((unsigned __int64)v103);
                    v103 = (unsigned __int8 *)v105;
                    sub_24A2F70((__int64 *)&v103, byte_3F871B3, (__int64)byte_3F871B3);
                    v79 = v116;
                    v80 = v78 == 0 ? -1 : 0x40;
                    if ( (unsigned __int64)(v115 - v116) > 4 )
                    {
                      *(_DWORD *)v116 = 1685016073;
                      v74 = &v113;
                      v79[4] = 101;
                      v116 += 5;
                    }
                    else
                    {
                      v74 = (_QWORD *)sub_CB6200((__int64)&v113, "\tNode", 5u);
                    }
                    sub_CB5A80((__int64)v74, v89);
                    if ( v80 != -1 )
                    {
                      v75 = sub_904010((__int64)&v113, ":s");
                      sub_CB59F0(v75, 64);
                    }
                    if ( (unsigned __int64)(v115 - v116) <= 7 )
                    {
                      v76 = (_QWORD *)sub_CB6200((__int64)&v113, " -> Node", 8u);
                    }
                    else
                    {
                      v76 = &v113;
                      *(_QWORD *)v116 = 0x65646F4E203E2D20LL;
                      v116 += 8;
                    }
                    sub_CB5A80((__int64)v76, v77);
                    if ( v104 )
                    {
                      v81 = sub_904010((__int64)&v113, "[");
                      v82 = sub_CB6200(v81, v103, v104);
                      sub_904010(v82, "]");
                    }
                    if ( (unsigned __int64)(v115 - v116) <= 1 )
                    {
                      sub_CB6200((__int64)&v113, (unsigned __int8 *)";\n", 2u);
                    }
                    else
                    {
                      *(_WORD *)v116 = 2619;
                      v116 += 2;
                    }
                    if ( v103 != (unsigned __int8 *)v105 )
                      j_j___libc_free_0((unsigned __int64)v103);
                  }
                  if ( v90 == ++v72 )
                    break;
                }
                goto LABEL_102;
              }
            }
          }
        }
        goto LABEL_102;
      }
LABEL_117:
      BUG();
    }
LABEL_102:
    v106 = (unsigned __int8 *)&unk_49DD210;
    sub_CB5840((__int64)&v106);
    if ( v98 != v100 )
      j_j___libc_free_0((unsigned __int64)v98);
    if ( v95 != v97 )
      j_j___libc_free_0((unsigned __int64)v95);
    v88 = *(_QWORD *)(v88 + 8);
  }
  while ( v86 != v88 );
LABEL_107:
  sub_904010((__int64)&v113, "}\n");
  if ( v92 != &v94 )
    j_j___libc_free_0((unsigned __int64)v92);
  v66 = sub_CB72A0();
  v40 = " done. \n";
  sub_904010((__int64)v66, " done. \n");
  *(_QWORD *)a1 = a1 + 16;
  if ( *(_QWORD *)a6 == a6 + 16 )
  {
    *(__m128i *)(a1 + 16) = _mm_loadu_si128((const __m128i *)(a6 + 16));
  }
  else
  {
    *(_QWORD *)a1 = *(_QWORD *)a6;
    *(_QWORD *)(a1 + 16) = *(_QWORD *)(a6 + 16);
  }
  *(_QWORD *)(a1 + 8) = *(_QWORD *)(a6 + 8);
  *(_QWORD *)a6 = a6 + 16;
  *(_QWORD *)(a6 + 8) = 0;
  *(_BYTE *)(a6 + 16) = 0;
LABEL_58:
  sub_CB5B00((int *)&v113, (__int64)v40);
  return a1;
}
