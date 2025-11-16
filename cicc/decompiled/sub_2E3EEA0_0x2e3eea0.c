// Function: sub_2E3EEA0
// Address: 0x2e3eea0
//
void __fastcall sub_2E3EEA0(__int64 a1, unsigned __int64 a2)
{
  __int64 v2; // r13
  __int64 *v5; // r15
  __int64 v6; // rdi
  __int64 v7; // rdx
  __int64 v8; // rax
  _QWORD *v9; // rdx
  __int64 v10; // rdi
  __int64 v11; // rdx
  unsigned __int64 v12; // rax
  size_t v13; // rdx
  __int64 v14; // rdi
  __int64 v15; // rdx
  __int64 v16; // rdi
  __m128i *v17; // r9
  unsigned __int64 v18; // r8
  __int64 v19; // rax
  __int64 v20; // rdx
  int v21; // esi
  int v22; // ecx
  unsigned int v23; // r14d
  __m128i si128; // xmm0
  __int64 v25; // rdx
  __m128i v26; // xmm0
  __int64 v27; // rax
  _WORD *v28; // rdx
  __int64 v29; // r15
  __int64 *v30; // rcx
  __int64 v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rdi
  _WORD *v34; // rdx
  unsigned __int64 v35; // rax
  __int64 v36; // rdi
  __int64 v37; // rdx
  unsigned __int64 *v38; // r15
  unsigned int v39; // r14d
  unsigned __int64 *v40; // r13
  unsigned __int64 v41; // r15
  unsigned int v42; // eax
  __int64 v43; // rdi
  _BYTE *v44; // rax
  void *v45; // rdx
  _BYTE *v46; // rax
  __int64 v47; // r14
  unsigned __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rdi
  _BYTE *v51; // rax
  __int64 v52; // rdi
  _BYTE *v53; // rax
  __int64 v54; // rdi
  _BYTE *v55; // rax
  __int64 i; // [rsp+10h] [rbp-C0h]
  unsigned int v57; // [rsp+18h] [rbp-B8h]
  unsigned __int8 *v58; // [rsp+20h] [rbp-B0h] BYREF
  size_t v59; // [rsp+28h] [rbp-A8h]
  _BYTE v60[16]; // [rsp+30h] [rbp-A0h] BYREF
  unsigned __int8 *v61; // [rsp+40h] [rbp-90h] BYREF
  size_t v62; // [rsp+48h] [rbp-88h]
  _BYTE v63[16]; // [rsp+50h] [rbp-80h] BYREF
  unsigned __int8 *v64; // [rsp+60h] [rbp-70h] BYREF
  size_t v65; // [rsp+68h] [rbp-68h]
  _BYTE *v66; // [rsp+70h] [rbp-60h] BYREF
  __int64 v67; // [rsp+78h] [rbp-58h]
  _BYTE *v68; // [rsp+80h] [rbp-50h]
  __int64 v69; // [rsp+88h] [rbp-48h]
  unsigned __int8 **v70; // [rsp+90h] [rbp-40h]

  v2 = a1 + 24;
  v5 = **(__int64 ***)(a1 + 8);
  v58 = v60;
  v59 = 0;
  v60[0] = 0;
  v57 = qword_4F8DE48[8];
  if ( LODWORD(qword_4F8DE48[8]) )
  {
    if ( !*(_QWORD *)(a1 + 32) )
    {
      v47 = *(_QWORD *)(sub_2E3A060(v5) + 328);
      for ( i = sub_2E3A060(v5) + 320; i != v47; v47 = *(_QWORD *)(v47 + 8) )
      {
        v48 = sub_2E39EA0(v5, v47);
        if ( v48 < *(_QWORD *)(a1 + 32) )
          v48 = *(_QWORD *)(a1 + 32);
        *(_QWORD *)(a1 + 32) = v48;
      }
    }
    v41 = sub_2E39EA0(v5, a2);
    v42 = sub_F02DD0(v57, 0x64u);
    v64 = *(unsigned __int8 **)(a1 + 32);
    if ( sub_1098D20((unsigned __int64 *)&v64, v42) <= v41 )
    {
      v69 = 0x100000000LL;
      v65 = 0;
      v64 = (unsigned __int8 *)&unk_49DD210;
      v70 = &v58;
      v66 = 0;
      v67 = 0;
      v68 = 0;
      sub_CB5980((__int64)&v64, 0, 0, 0);
      v45 = v68;
      if ( (unsigned __int64)(v67 - (_QWORD)v68) <= 0xA )
      {
        sub_CB6200((__int64)&v64, (unsigned __int8 *)"color=\"red\"", 0xBu);
        v46 = v68;
      }
      else
      {
        v68[10] = 34;
        qmemcpy(v45, "color=\"red", 10);
        v46 = v68 + 11;
        v68 += 11;
      }
      if ( v66 != v46 )
        sub_CB5AE0((__int64 *)&v64);
      v64 = (unsigned __int8 *)&unk_49DD210;
      sub_CB5840((__int64)&v64);
    }
  }
  v6 = *(_QWORD *)a1;
  v7 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  if ( (unsigned __int64)(*(_QWORD *)(*(_QWORD *)a1 + 24LL) - v7) <= 4 )
  {
    v6 = sub_CB6200(v6, "\tNode", 5u);
  }
  else
  {
    *(_DWORD *)v7 = 1685016073;
    *(_BYTE *)(v7 + 4) = 101;
    *(_QWORD *)(v6 + 32) += 5LL;
  }
  v8 = sub_CB5A80(v6, a2);
  v9 = *(_QWORD **)(v8 + 32);
  if ( *(_QWORD *)(v8 + 24) - (_QWORD)v9 <= 7u )
  {
    sub_CB6200(v8, " [shape=", 8u);
  }
  else
  {
    *v9 = 0x3D65706168735B20LL;
    *(_QWORD *)(v8 + 32) += 8LL;
  }
  v10 = *(_QWORD *)a1;
  v11 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  v12 = *(_QWORD *)(*(_QWORD *)a1 + 24LL) - v11;
  if ( *(_BYTE *)(a1 + 16) )
  {
    if ( v12 <= 4 )
    {
      sub_CB6200(v10, (unsigned __int8 *)"none,", 5u);
    }
    else
    {
      *(_DWORD *)v11 = 1701736302;
      *(_BYTE *)(v11 + 4) = 44;
      *(_QWORD *)(v10 + 32) += 5LL;
    }
LABEL_9:
    v13 = v59;
    if ( !v59 )
      goto LABEL_10;
LABEL_48:
    v43 = sub_CB6200(*(_QWORD *)a1, v58, v13);
    v44 = *(_BYTE **)(v43 + 32);
    if ( *(_BYTE **)(v43 + 24) == v44 )
    {
      sub_CB6200(v43, (unsigned __int8 *)",", 1u);
    }
    else
    {
      *v44 = 44;
      ++*(_QWORD *)(v43 + 32);
    }
    goto LABEL_10;
  }
  if ( v12 <= 6 )
  {
    sub_CB6200(v10, (unsigned __int8 *)"record,", 7u);
    goto LABEL_9;
  }
  *(_DWORD *)v11 = 1868785010;
  *(_WORD *)(v11 + 4) = 25714;
  *(_BYTE *)(v11 + 6) = 44;
  v13 = v59;
  *(_QWORD *)(v10 + 32) += 7LL;
  if ( v13 )
    goto LABEL_48;
LABEL_10:
  v14 = *(_QWORD *)a1;
  v15 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  if ( (unsigned __int64)(*(_QWORD *)(*(_QWORD *)a1 + 24LL) - v15) <= 5 )
  {
    sub_CB6200(v14, "label=", 6u);
  }
  else
  {
    *(_DWORD *)v15 = 1700946284;
    *(_WORD *)(v15 + 4) = 15724;
    *(_QWORD *)(v14 + 32) += 6LL;
  }
  v16 = *(_QWORD *)a1;
  v17 = *(__m128i **)(*(_QWORD *)a1 + 32LL);
  v18 = *(_QWORD *)(*(_QWORD *)a1 + 24LL) - (_QWORD)v17;
  if ( *(_BYTE *)(a1 + 16) )
  {
    v19 = *(_QWORD *)(a2 + 112);
    v20 = v19 + 8LL * *(unsigned int *)(a2 + 120);
    if ( v19 == v20 )
    {
      v23 = 1;
    }
    else
    {
      v21 = 0;
      do
      {
        v19 += 8;
        ++v21;
      }
      while ( v19 != v20 && v21 != 64 );
      v22 = 1;
      if ( v21 )
        v22 = v21;
      v23 = (v19 != v20) + v22;
    }
    if ( v18 <= 0x30 )
    {
      v49 = sub_CB6200(v16, "<<table border=\"0\" cellborder=\"1\" cellspacing=\"0\"", 0x31u);
      v25 = *(_QWORD *)(v49 + 32);
      v16 = v49;
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CB60);
      v17[3].m128i_i8[0] = 34;
      *v17 = si128;
      v17[1] = _mm_load_si128((const __m128i *)&xmmword_3F8CB70);
      v17[2] = _mm_load_si128((const __m128i *)&xmmword_3F8CB80);
      v25 = *(_QWORD *)(v16 + 32) + 49LL;
      *(_QWORD *)(v16 + 32) = v25;
    }
    if ( (unsigned __int64)(*(_QWORD *)(v16 + 24) - v25) <= 0x2E )
    {
      v16 = sub_CB6200(v16, " cellpadding=\"0\"><tr><td align=\"text\" colspan=\"", 0x2Fu);
    }
    else
    {
      v26 = _mm_load_si128((const __m128i *)&xmmword_3F8CB90);
      qmemcpy((void *)(v25 + 32), "text\" colspan=\"", 15);
      *(__m128i *)v25 = v26;
      *(__m128i *)(v25 + 16) = _mm_load_si128((const __m128i *)&xmmword_3F8CBA0);
      *(_QWORD *)(v16 + 32) += 47LL;
    }
    v27 = sub_CB59D0(v16, v23);
    v28 = *(_WORD **)(v27 + 32);
    if ( *(_QWORD *)(v27 + 24) - (_QWORD)v28 <= 1u )
    {
      sub_CB6200(v27, "\">", 2u);
    }
    else
    {
      *v28 = 15906;
      *(_QWORD *)(v27 + 32) += 2LL;
    }
  }
  else if ( v18 <= 1 )
  {
    sub_CB6200(v16, (unsigned __int8 *)"\"{", 2u);
  }
  else
  {
    v17->m128i_i16[0] = 31522;
    *(_QWORD *)(v16 + 32) += 2LL;
  }
  v29 = *(_QWORD *)a1;
  v30 = **(__int64 ***)(a1 + 8);
  if ( *(_BYTE *)(a1 + 16) )
  {
    sub_2E3AFD0((__int64)&v64, v2, a2, v30);
    v31 = sub_CB6200(v29, v64, v65);
    v32 = *(_QWORD *)(v31 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(v31 + 24) - v32) <= 4 )
    {
      sub_CB6200(v31, "</td>", 5u);
    }
    else
    {
      *(_DWORD *)v32 = 1685335868;
      *(_BYTE *)(v32 + 4) = 62;
      *(_QWORD *)(v31 + 32) += 5LL;
    }
    if ( v64 != (unsigned __int8 *)&v66 )
      j_j___libc_free_0((unsigned __int64)v64);
  }
  else
  {
    sub_2E3AFD0((__int64)&v61, v2, a2, v30);
    sub_C67200((__int64 *)&v64, (__int64)&v61);
    sub_CB6200(v29, v64, v65);
    if ( v64 != (unsigned __int8 *)&v66 )
      j_j___libc_free_0((unsigned __int64)v64);
    if ( v61 != v63 )
      j_j___libc_free_0((unsigned __int64)v61);
  }
  v70 = &v61;
  v63[0] = 0;
  v61 = v63;
  v69 = 0x100000000LL;
  v62 = 0;
  v65 = 0;
  v66 = 0;
  v67 = 0;
  v68 = 0;
  v64 = (unsigned __int8 *)&unk_49DD210;
  sub_CB5980((__int64)&v64, 0, 0, 0);
  if ( (unsigned __int8)sub_2E3E650(a1, (__int64)&v64, a2) )
  {
    if ( *(_BYTE *)(a1 + 16) )
      goto LABEL_33;
    v50 = *(_QWORD *)a1;
    v51 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
    if ( *(_BYTE **)(*(_QWORD *)a1 + 24LL) == v51 )
    {
      sub_CB6200(v50, (unsigned __int8 *)"|", 1u);
    }
    else
    {
      *v51 = 124;
      ++*(_QWORD *)(v50 + 32);
    }
    v52 = *(_QWORD *)a1;
    if ( *(_BYTE *)(a1 + 16) )
    {
LABEL_33:
      sub_CB6200(*(_QWORD *)a1, v61, v62);
    }
    else
    {
      v53 = *(_BYTE **)(v52 + 32);
      if ( *(_BYTE **)(v52 + 24) == v53 )
      {
        v52 = sub_CB6200(v52, (unsigned __int8 *)"{", 1u);
      }
      else
      {
        *v53 = 123;
        ++*(_QWORD *)(v52 + 32);
      }
      v54 = sub_CB6200(v52, v61, v62);
      v55 = *(_BYTE **)(v54 + 32);
      if ( *(_BYTE **)(v54 + 24) == v55 )
      {
        sub_CB6200(v54, (unsigned __int8 *)"}", 1u);
      }
      else
      {
        *v55 = 125;
        ++*(_QWORD *)(v54 + 32);
      }
    }
  }
  v33 = *(_QWORD *)a1;
  v34 = *(_WORD **)(*(_QWORD *)a1 + 32LL);
  v35 = *(_QWORD *)(*(_QWORD *)a1 + 24LL) - (_QWORD)v34;
  if ( *(_BYTE *)(a1 + 16) )
  {
    if ( v35 <= 0xD )
    {
      sub_CB6200(v33, "</tr></table>>", 0xEu);
    }
    else
    {
      qmemcpy(v34, "</tr></table>>", 14);
      *(_QWORD *)(v33 + 32) += 14LL;
    }
  }
  else if ( v35 <= 1 )
  {
    sub_CB6200(v33, (unsigned __int8 *)"}\"", 2u);
  }
  else
  {
    *v34 = 8829;
    *(_QWORD *)(v33 + 32) += 2LL;
  }
  v36 = *(_QWORD *)a1;
  v37 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  if ( (unsigned __int64)(*(_QWORD *)(*(_QWORD *)a1 + 24LL) - v37) <= 2 )
  {
    sub_CB6200(v36, (unsigned __int8 *)"];\n", 3u);
  }
  else
  {
    *(_BYTE *)(v37 + 2) = 10;
    *(_WORD *)v37 = 15197;
    *(_QWORD *)(v36 + 32) += 3LL;
  }
  v38 = *(unsigned __int64 **)(a2 + 112);
  v39 = 0;
  v40 = &v38[*(unsigned int *)(a2 + 120)];
  if ( v40 != v38 )
  {
    while ( 1 )
    {
      sub_2E3EAA0(a1, a2, v39++, v38++);
      if ( v38 == v40 )
        break;
      if ( v39 == 64 )
      {
        while ( v38 != v40 )
          sub_2E3EAA0(a1, a2, 64, v38++);
        break;
      }
    }
  }
  v64 = (unsigned __int8 *)&unk_49DD210;
  sub_CB5840((__int64)&v64);
  if ( v61 != v63 )
    j_j___libc_free_0((unsigned __int64)v61);
  if ( v58 != v60 )
    j_j___libc_free_0((unsigned __int64)v58);
}
