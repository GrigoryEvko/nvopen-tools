// Function: sub_22E8150
// Address: 0x22e8150
//
void __fastcall sub_22E8150(__int64 a1, __int64 *a2)
{
  __int64 v3; // rdi
  __int64 v4; // rdx
  __int64 v5; // rax
  __int64 v6; // r9
  _QWORD *v7; // rdx
  __int64 v8; // rdi
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rdx
  unsigned __int64 v13; // rbx
  __int64 v14; // r14
  unsigned int v15; // r15d
  unsigned int v16; // esi
  unsigned int v17; // r12d
  unsigned __int64 v18; // r12
  __int64 v19; // rdi
  __m128i *v20; // rdx
  __m128i si128; // xmm0
  __int64 v22; // rdx
  __m128i v23; // xmm0
  __int64 v24; // rax
  __int64 v25; // r8
  _WORD *v26; // rdx
  _BYTE *v27; // rsi
  __int64 v28; // r12
  __int64 v29; // rcx
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rdi
  _WORD *v33; // rdx
  unsigned __int64 v34; // rax
  __int64 v35; // rdi
  __int64 v36; // rdx
  unsigned __int64 v37; // rdx
  _QWORD *v38; // r12
  __int64 v39; // rax
  __int64 v40; // r14
  unsigned int v41; // r15d
  unsigned int v42; // esi
  unsigned int v43; // ebx
  unsigned int v44; // r15d
  unsigned __int64 v45; // rax
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // r9
  unsigned __int64 v49; // r12
  unsigned int v50; // ebx
  unsigned int v51; // r15d
  unsigned __int64 v52; // rax
  __int64 v53; // rcx
  __int64 v54; // r8
  __int64 v55; // r9
  unsigned __int64 v56; // r12
  int v57; // eax
  unsigned int v58; // esi
  __int64 v59; // rdi
  _WORD *v60; // rdx
  __int64 v61; // rdi
  _BYTE *v62; // rax
  __int64 v63; // rdi
  _BYTE *v64; // rax
  __int64 v65; // rdi
  _BYTE *v66; // rax
  __int64 v67; // rax
  int v68; // [rsp+24h] [rbp-ECh]
  int v69; // [rsp+24h] [rbp-ECh]
  unsigned int v71; // [rsp+30h] [rbp-E0h]
  __int64 v72; // [rsp+30h] [rbp-E0h]
  unsigned int v73; // [rsp+30h] [rbp-E0h]
  _QWORD *v74; // [rsp+38h] [rbp-D8h]
  _QWORD *v75; // [rsp+38h] [rbp-D8h]
  _QWORD *v76; // [rsp+38h] [rbp-D8h]
  _QWORD *v77; // [rsp+38h] [rbp-D8h]
  unsigned __int8 *v78; // [rsp+80h] [rbp-90h] BYREF
  size_t v79; // [rsp+88h] [rbp-88h]
  _BYTE v80[16]; // [rsp+90h] [rbp-80h] BYREF
  unsigned __int8 *v81; // [rsp+A0h] [rbp-70h] BYREF
  size_t v82; // [rsp+A8h] [rbp-68h]
  _QWORD v83[12]; // [rsp+B0h] [rbp-60h] BYREF

  v3 = *(_QWORD *)a1;
  v4 = *(_QWORD *)(v3 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v3 + 24) - v4) <= 4 )
  {
    v3 = sub_CB6200(v3, "\tNode", 5u);
  }
  else
  {
    *(_DWORD *)v4 = 1685016073;
    *(_BYTE *)(v4 + 4) = 101;
    *(_QWORD *)(v3 + 32) += 5LL;
  }
  v5 = sub_CB5A80(v3, (unsigned __int64)a2);
  v7 = *(_QWORD **)(v5 + 32);
  if ( *(_QWORD *)(v5 + 24) - (_QWORD)v7 <= 7u )
  {
    sub_CB6200(v5, " [shape=", 8u);
  }
  else
  {
    *v7 = 0x3D65706168735B20LL;
    *(_QWORD *)(v5 + 32) += 8LL;
  }
  v8 = *(_QWORD *)a1;
  v9 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  v10 = *(_QWORD *)(*(_QWORD *)a1 + 24LL) - v9;
  if ( *(_BYTE *)(a1 + 16) )
  {
    if ( v10 <= 4 )
    {
      sub_CB6200(v8, (unsigned __int8 *)"none,", 5u);
    }
    else
    {
      *(_DWORD *)v9 = 1701736302;
      *(_BYTE *)(v9 + 4) = 44;
      *(_QWORD *)(v8 + 32) += 5LL;
    }
  }
  else if ( v10 <= 6 )
  {
    sub_CB6200(v8, (unsigned __int8 *)"record,", 7u);
  }
  else
  {
    *(_DWORD *)v9 = 1868785010;
    *(_WORD *)(v9 + 4) = 25714;
    *(_BYTE *)(v9 + 6) = 44;
    *(_QWORD *)(v8 + 32) += 7LL;
  }
  v11 = *(_QWORD *)a1;
  v12 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  if ( (unsigned __int64)(*(_QWORD *)(*(_QWORD *)a1 + 24LL) - v12) <= 5 )
  {
    sub_CB6200(v11, "label=", 6u);
  }
  else
  {
    *(_DWORD *)v12 = 1700946284;
    *(_WORD *)(v12 + 4) = 15724;
    *(_QWORD *)(v11 + 32) += 6LL;
  }
  if ( *(_BYTE *)(a1 + 16) )
  {
    v74 = (_QWORD *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 48);
    v13 = *v74 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v74 == (_QWORD *)v13 )
    {
      v14 = 0;
    }
    else
    {
      if ( !v13 )
        goto LABEL_23;
      v14 = v13 - 24;
      if ( (unsigned int)*(unsigned __int8 *)(v13 - 24) - 30 >= 0xB )
        v14 = 0;
    }
    v15 = 0;
    do
    {
      v17 = v15;
      if ( v74 != (_QWORD *)v13 )
      {
        if ( !v13 )
          goto LABEL_23;
        if ( (unsigned int)*(unsigned __int8 *)(v13 - 24) - 30 <= 0xA )
        {
          if ( (unsigned int)sub_B46E30(v13 - 24) == v15 )
            break;
          goto LABEL_20;
        }
      }
      if ( !v15 )
        break;
LABEL_20:
      v16 = v15++;
    }
    while ( *(_QWORD *)(a2[1] + 32) == sub_B46EC0(v14, v16) );
    if ( v74 == (_QWORD *)v13 )
      goto LABEL_132;
    if ( !v13 )
      goto LABEL_23;
    if ( (unsigned int)*(unsigned __int8 *)(v13 - 24) - 30 <= 0xA )
    {
      v68 = sub_B46E30(v13 - 24);
      goto LABEL_33;
    }
LABEL_132:
    v68 = 0;
LABEL_33:
    if ( v68 != v17 )
    {
      v71 = 0;
      while ( 1 )
      {
        ++v17;
        if ( v74 == (_QWORD *)v13 )
          goto LABEL_41;
        if ( !v13 )
          goto LABEL_23;
        if ( (unsigned int)*(unsigned __int8 *)(v13 - 24) - 30 > 0xA )
        {
LABEL_41:
          if ( !v17 )
            goto LABEL_42;
        }
        else if ( v17 == (unsigned int)sub_B46E30(v13 - 24) )
        {
          goto LABEL_42;
        }
        if ( *(_QWORD *)(a2[1] + 32) == sub_B46EC0(v14, v17) )
          continue;
LABEL_42:
        ++v71;
        if ( v68 == v17 )
        {
          v18 = v71;
          goto LABEL_45;
        }
        if ( v71 == 64 )
        {
          v18 = 65;
          goto LABEL_45;
        }
      }
    }
    v18 = 1;
LABEL_45:
    v19 = *(_QWORD *)a1;
    v20 = *(__m128i **)(*(_QWORD *)a1 + 32LL);
    if ( *(_QWORD *)(*(_QWORD *)a1 + 24LL) - (_QWORD)v20 <= 0x30u )
    {
      v67 = sub_CB6200(v19, "<<table border=\"0\" cellborder=\"1\" cellspacing=\"0\"", 0x31u);
      v22 = *(_QWORD *)(v67 + 32);
      v19 = v67;
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CB60);
      v20[3].m128i_i8[0] = 34;
      *v20 = si128;
      v20[1] = _mm_load_si128((const __m128i *)&xmmword_3F8CB70);
      v20[2] = _mm_load_si128((const __m128i *)&xmmword_3F8CB80);
      v22 = *(_QWORD *)(v19 + 32) + 49LL;
      *(_QWORD *)(v19 + 32) = v22;
    }
    if ( (unsigned __int64)(*(_QWORD *)(v19 + 24) - v22) <= 0x2E )
    {
      v19 = sub_CB6200(v19, " cellpadding=\"0\"><tr><td align=\"text\" colspan=\"", 0x2Fu);
    }
    else
    {
      v23 = _mm_load_si128((const __m128i *)&xmmword_3F8CB90);
      qmemcpy((void *)(v22 + 32), "text\" colspan=\"", 15);
      *(__m128i *)v22 = v23;
      *(__m128i *)(v22 + 16) = _mm_load_si128((const __m128i *)&xmmword_3F8CBA0);
      *(_QWORD *)(v19 + 32) += 47LL;
    }
    v24 = sub_CB59D0(v19, v18);
    v26 = *(_WORD **)(v24 + 32);
    if ( *(_QWORD *)(v24 + 24) - (_QWORD)v26 <= 1u )
    {
      sub_CB6200(v24, "\">", 2u);
    }
    else
    {
      v6 = 15906;
      *v26 = 15906;
      *(_QWORD *)(v24 + 32) += 2LL;
    }
  }
  else
  {
    v59 = *(_QWORD *)a1;
    v60 = *(_WORD **)(*(_QWORD *)a1 + 32LL);
    if ( *(_QWORD *)(*(_QWORD *)a1 + 24LL) - (_QWORD)v60 <= 1u )
    {
      sub_CB6200(v59, (unsigned __int8 *)"\"{", 2u);
    }
    else
    {
      v25 = 31522;
      *v60 = 31522;
      *(_QWORD *)(v59 + 32) += 2LL;
    }
  }
  v27 = (_BYTE *)(a1 + 17);
  v28 = *(_QWORD *)a1;
  v29 = *(_QWORD *)(**(_QWORD **)(a1 + 8) + 32LL);
  if ( *(_BYTE *)(a1 + 16) )
  {
    sub_22E5F10((__int64 *)&v81, v27, a2, v29, v25, v6);
    v30 = sub_CB6200(v28, v81, v82);
    v31 = *(_QWORD *)(v30 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(v30 + 24) - v31) <= 4 )
    {
      sub_CB6200(v30, "</td>", 5u);
    }
    else
    {
      *(_DWORD *)v31 = 1685335868;
      *(_BYTE *)(v31 + 4) = 62;
      *(_QWORD *)(v30 + 32) += 5LL;
    }
    if ( v81 != (unsigned __int8 *)v83 )
      j_j___libc_free_0((unsigned __int64)v81);
  }
  else
  {
    sub_22E5F10((__int64 *)&v78, v27, a2, v29, v25, v6);
    sub_C67200((__int64 *)&v81, (__int64)&v78);
    sub_CB6200(v28, v81, v82);
    if ( v81 != (unsigned __int8 *)v83 )
      j_j___libc_free_0((unsigned __int64)v81);
    if ( v78 != v80 )
      j_j___libc_free_0((unsigned __int64)v78);
  }
  v83[4] = &v78;
  v80[0] = 0;
  v78 = v80;
  v83[3] = 0x100000000LL;
  v79 = 0;
  v82 = 0;
  memset(v83, 0, 24);
  v81 = (unsigned __int8 *)&unk_49DD210;
  sub_CB5980((__int64)&v81, 0, 0, 0);
  if ( (unsigned __int8)sub_22E7820(a1, (__int64)&v81, a2) )
  {
    if ( *(_BYTE *)(a1 + 16) )
      goto LABEL_58;
    v61 = *(_QWORD *)a1;
    v62 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
    if ( *(_BYTE **)(*(_QWORD *)a1 + 24LL) == v62 )
    {
      sub_CB6200(v61, (unsigned __int8 *)"|", 1u);
    }
    else
    {
      *v62 = 124;
      ++*(_QWORD *)(v61 + 32);
    }
    v63 = *(_QWORD *)a1;
    if ( *(_BYTE *)(a1 + 16) )
    {
LABEL_58:
      sub_CB6200(*(_QWORD *)a1, v78, v79);
    }
    else
    {
      v64 = *(_BYTE **)(v63 + 32);
      if ( *(_BYTE **)(v63 + 24) == v64 )
      {
        v63 = sub_CB6200(v63, (unsigned __int8 *)"{", 1u);
      }
      else
      {
        *v64 = 123;
        ++*(_QWORD *)(v63 + 32);
      }
      v65 = sub_CB6200(v63, v78, v79);
      v66 = *(_BYTE **)(v65 + 32);
      if ( *(_BYTE **)(v65 + 24) == v66 )
      {
        sub_CB6200(v65, (unsigned __int8 *)"}", 1u);
      }
      else
      {
        *v66 = 125;
        ++*(_QWORD *)(v65 + 32);
      }
    }
  }
  v32 = *(_QWORD *)a1;
  v33 = *(_WORD **)(*(_QWORD *)a1 + 32LL);
  v34 = *(_QWORD *)(*(_QWORD *)a1 + 24LL) - (_QWORD)v33;
  if ( *(_BYTE *)(a1 + 16) )
  {
    if ( v34 <= 0xD )
    {
      sub_CB6200(v32, "</tr></table>>", 0xEu);
    }
    else
    {
      qmemcpy(v33, "</tr></table>>", 14);
      *(_QWORD *)(v32 + 32) += 14LL;
    }
  }
  else if ( v34 <= 1 )
  {
    sub_CB6200(v32, (unsigned __int8 *)"}\"", 2u);
  }
  else
  {
    *v33 = 8829;
    *(_QWORD *)(v32 + 32) += 2LL;
  }
  v35 = *(_QWORD *)a1;
  v36 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  if ( (unsigned __int64)(*(_QWORD *)(*(_QWORD *)a1 + 24LL) - v36) <= 2 )
  {
    sub_CB6200(v35, (unsigned __int8 *)"];\n", 3u);
  }
  else
  {
    *(_BYTE *)(v36 + 2) = 10;
    *(_WORD *)v36 = 15197;
    *(_QWORD *)(v35 + 32) += 3LL;
  }
  v75 = (_QWORD *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 48);
  v37 = *v75 & 0xFFFFFFFFFFFFFFF8LL;
  v38 = (_QWORD *)v37;
  if ( v75 == (_QWORD *)v37 )
  {
    v40 = 0;
  }
  else
  {
    if ( !v37 )
      goto LABEL_23;
    v39 = 0;
    if ( (unsigned int)*(unsigned __int8 *)(v37 - 24) - 30 < 0xB )
      v39 = v37 - 24;
    v40 = v39;
  }
  v41 = 0;
  v72 = v37 - 24;
  do
  {
    v43 = v41;
    if ( v75 != v38 )
    {
      if ( !v38 )
        goto LABEL_23;
      if ( (unsigned int)*((unsigned __int8 *)v38 - 24) - 30 <= 0xA )
      {
        if ( v41 == (unsigned int)sub_B46E30(v72) )
          break;
        goto LABEL_72;
      }
    }
    if ( !v41 )
      break;
LABEL_72:
    v42 = v41++;
  }
  while ( *(_QWORD *)(a2[1] + 32) == sub_B46EC0(v40, v42) );
  if ( v75 == v38 )
    goto LABEL_116;
  if ( !v38 )
LABEL_23:
    BUG();
  if ( (unsigned int)*((unsigned __int8 *)v38 - 24) - 30 <= 0xA )
  {
    v69 = sub_B46E30((__int64)(v38 - 3));
    goto LABEL_81;
  }
LABEL_116:
  v69 = 0;
LABEL_81:
  v73 = 0;
  v44 = v43;
  if ( v43 == v69 )
    goto LABEL_93;
  while ( 2 )
  {
    v45 = sub_B46EC0(v40, v44);
    sub_22DDF00((_QWORD *)a2[1], v45);
    sub_22E7E60((__int64 *)a1, a2, v73, v46, v47, v48, (__int64)a2, v40, v44);
    v76 = (_QWORD *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 48);
    v49 = *v76 & 0xFFFFFFFFFFFFFFF8LL;
    v50 = v44;
    while ( 2 )
    {
      ++v50;
      if ( v76 != (_QWORD *)v49 )
      {
        if ( !v49 )
          goto LABEL_23;
        if ( (unsigned int)*(unsigned __int8 *)(v49 - 24) - 30 <= 0xA )
        {
          if ( v50 == (unsigned int)sub_B46E30(v49 - 24) )
            goto LABEL_91;
LABEL_85:
          if ( *(_QWORD *)(a2[1] + 32) != sub_B46EC0(v40, v50) )
            goto LABEL_91;
          continue;
        }
      }
      break;
    }
    if ( v50 )
      goto LABEL_85;
LABEL_91:
    ++v73;
    v44 = v50;
    if ( v73 != 64 )
    {
      if ( v69 == v50 )
        goto LABEL_93;
      continue;
    }
    break;
  }
  while ( v69 != v50 )
  {
    v51 = v50 + 1;
    v52 = sub_B46EC0(v40, v50);
    sub_22DDF00((_QWORD *)a2[1], v52);
    sub_22E7E60((__int64 *)a1, a2, 64, v53, v54, v55, (__int64)a2, v40, v50);
    v77 = (_QWORD *)((*a2 & 0xFFFFFFFFFFFFFFF8LL) + 48);
    v56 = *v77 & 0xFFFFFFFFFFFFFFF8LL;
    do
    {
      v50 = v51;
      if ( v77 == (_QWORD *)v56 )
        goto LABEL_105;
      if ( !v56 )
        goto LABEL_23;
      if ( (unsigned int)*(unsigned __int8 *)(v56 - 24) - 30 > 0xA )
LABEL_105:
        v57 = 0;
      else
        v57 = sub_B46E30(v56 - 24);
      if ( v51 == v57 )
        break;
      v58 = v51++;
    }
    while ( *(_QWORD *)(a2[1] + 32) == sub_B46EC0(v40, v58) );
  }
LABEL_93:
  v81 = (unsigned __int8 *)&unk_49DD210;
  sub_CB5840((__int64)&v81);
  if ( v78 != v80 )
    j_j___libc_free_0((unsigned __int64)v78);
}
