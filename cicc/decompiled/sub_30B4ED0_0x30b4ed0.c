// Function: sub_30B4ED0
// Address: 0x30b4ed0
//
void __fastcall sub_30B4ED0(__int64 a1, unsigned __int64 a2)
{
  __int64 v4; // rdi
  __int64 v5; // rdx
  __int64 v6; // rax
  _QWORD *v7; // rdx
  __int64 v8; // rdi
  __int64 v9; // rdx
  unsigned __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rdx
  __int64 v13; // rdi
  __m128i *v14; // r8
  unsigned __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rax
  unsigned int v19; // edx
  unsigned __int64 v20; // r12
  __m128i si128; // xmm0
  __int64 v22; // rdx
  __int64 v23; // rax
  __m128i v24; // xmm0
  __int64 v25; // rax
  _WORD *v26; // rdx
  __int64 v27; // r15
  __int64 v28; // rcx
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rdi
  _WORD *v32; // rdx
  unsigned __int64 v33; // rax
  __int64 v34; // rdi
  __int64 v35; // rdx
  __int64 **v36; // r15
  int v37; // r14d
  __int64 *v38; // rcx
  int v39; // edx
  __int64 v40; // rdi
  _BYTE *v41; // rax
  __int64 v42; // rdi
  _BYTE *v43; // rax
  __int64 v44; // rdi
  _BYTE *v45; // rax
  __int64 **v46; // [rsp+10h] [rbp-D0h]
  unsigned __int64 v47; // [rsp+28h] [rbp-B8h]
  unsigned __int8 *v48; // [rsp+50h] [rbp-90h] BYREF
  size_t v49; // [rsp+58h] [rbp-88h]
  _BYTE v50[16]; // [rsp+60h] [rbp-80h] BYREF
  unsigned __int8 *v51; // [rsp+70h] [rbp-70h] BYREF
  size_t v52; // [rsp+78h] [rbp-68h]
  _QWORD v53[12]; // [rsp+80h] [rbp-60h] BYREF

  v4 = *(_QWORD *)a1;
  v5 = *(_QWORD *)(v4 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v4 + 24) - v5) <= 4 )
  {
    v4 = sub_CB6200(v4, "\tNode", 5u);
  }
  else
  {
    *(_DWORD *)v5 = 1685016073;
    *(_BYTE *)(v5 + 4) = 101;
    *(_QWORD *)(v4 + 32) += 5LL;
  }
  v6 = sub_CB5A80(v4, a2);
  v7 = *(_QWORD **)(v6 + 32);
  if ( *(_QWORD *)(v6 + 24) - (_QWORD)v7 <= 7u )
  {
    sub_CB6200(v6, " [shape=", 8u);
  }
  else
  {
    *v7 = 0x3D65706168735B20LL;
    *(_QWORD *)(v6 + 32) += 8LL;
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
  v13 = *(_QWORD *)a1;
  v14 = *(__m128i **)(*(_QWORD *)a1 + 32LL);
  v15 = *(_QWORD *)(*(_QWORD *)a1 + 24LL) - (_QWORD)v14;
  if ( *(_BYTE *)(a1 + 16) )
  {
    v16 = *(_QWORD *)(a2 + 40);
    v17 = v16 + 8LL * *(unsigned int *)(a2 + 48);
    if ( v16 == v17 )
    {
      v20 = 1;
    }
    else
    {
      v18 = v16 + 8;
      v19 = 0;
      do
      {
        ++v19;
        if ( v17 == v18 )
        {
          v20 = v19;
          goto LABEL_18;
        }
        v18 += 8;
      }
      while ( v19 != 64 );
      v20 = 65;
    }
LABEL_18:
    if ( v15 <= 0x30 )
    {
      v13 = sub_CB6200(v13, "<<table border=\"0\" cellborder=\"1\" cellspacing=\"0\"", 0x31u);
      v22 = *(_QWORD *)(v13 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(v13 + 24) - v22) > 0x2E )
        goto LABEL_20;
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CB60);
      v14[3].m128i_i8[0] = 34;
      *v14 = si128;
      v14[1] = _mm_load_si128((const __m128i *)&xmmword_3F8CB70);
      v14[2] = _mm_load_si128((const __m128i *)&xmmword_3F8CB80);
      v22 = *(_QWORD *)(v13 + 32) + 49LL;
      v23 = *(_QWORD *)(v13 + 24);
      *(_QWORD *)(v13 + 32) = v22;
      if ( (unsigned __int64)(v23 - v22) > 0x2E )
      {
LABEL_20:
        v24 = _mm_load_si128((const __m128i *)&xmmword_3F8CB90);
        qmemcpy((void *)(v22 + 32), "text\" colspan=\"", 15);
        *(__m128i *)v22 = v24;
        *(__m128i *)(v22 + 16) = _mm_load_si128((const __m128i *)&xmmword_3F8CBA0);
        *(_QWORD *)(v13 + 32) += 47LL;
        goto LABEL_21;
      }
    }
    v13 = sub_CB6200(v13, " cellpadding=\"0\"><tr><td align=\"text\" colspan=\"", 0x2Fu);
LABEL_21:
    v25 = sub_CB59D0(v13, v20);
    v26 = *(_WORD **)(v25 + 32);
    if ( *(_QWORD *)(v25 + 24) - (_QWORD)v26 <= 1u )
    {
      sub_CB6200(v25, "\">", 2u);
    }
    else
    {
      *v26 = 15906;
      *(_QWORD *)(v25 + 32) += 2LL;
    }
    goto LABEL_23;
  }
  if ( v15 <= 1 )
  {
    sub_CB6200(v13, (unsigned __int8 *)"\"{", 2u);
  }
  else
  {
    v14->m128i_i16[0] = 31522;
    *(_QWORD *)(v13 + 32) += 2LL;
  }
LABEL_23:
  v27 = *(_QWORD *)a1;
  v28 = **(_QWORD **)(a1 + 8);
  if ( *(_BYTE *)(a1 + 16) )
  {
    sub_30B38C0((__int64 *)&v51, (_BYTE *)(a1 + 17), a2, v28);
    v29 = sub_CB6200(v27, v51, v52);
    v30 = *(_QWORD *)(v29 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(v29 + 24) - v30) <= 4 )
    {
      sub_CB6200(v29, "</td>", 5u);
    }
    else
    {
      *(_DWORD *)v30 = 1685335868;
      *(_BYTE *)(v30 + 4) = 62;
      *(_QWORD *)(v29 + 32) += 5LL;
    }
    if ( v51 != (unsigned __int8 *)v53 )
      j_j___libc_free_0((unsigned __int64)v51);
  }
  else
  {
    sub_30B38C0((__int64 *)&v48, (_BYTE *)(a1 + 17), a2, v28);
    sub_C67200((__int64 *)&v51, (__int64)&v48);
    sub_CB6200(v27, v51, v52);
    if ( v51 != (unsigned __int8 *)v53 )
      j_j___libc_free_0((unsigned __int64)v51);
    if ( v48 != v50 )
      j_j___libc_free_0((unsigned __int64)v48);
  }
  v50[0] = 0;
  v48 = v50;
  v53[3] = 0x100000000LL;
  v49 = 0;
  v52 = 0;
  memset(v53, 0, 24);
  v51 = (unsigned __int8 *)&unk_49DD210;
  v53[4] = &v48;
  sub_CB5980((__int64)&v51, 0, 0, 0);
  if ( (unsigned __int8)sub_30B47D0(a1, (__int64)&v51, a2) )
  {
    if ( *(_BYTE *)(a1 + 16) )
      goto LABEL_30;
    v40 = *(_QWORD *)a1;
    v41 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
    if ( *(_BYTE **)(*(_QWORD *)a1 + 24LL) == v41 )
    {
      sub_CB6200(v40, (unsigned __int8 *)"|", 1u);
    }
    else
    {
      *v41 = 124;
      ++*(_QWORD *)(v40 + 32);
    }
    v42 = *(_QWORD *)a1;
    if ( *(_BYTE *)(a1 + 16) )
    {
LABEL_30:
      sub_CB6200(*(_QWORD *)a1, v48, v49);
    }
    else
    {
      v43 = *(_BYTE **)(v42 + 32);
      if ( *(_BYTE **)(v42 + 24) == v43 )
      {
        v42 = sub_CB6200(v42, (unsigned __int8 *)"{", 1u);
      }
      else
      {
        *v43 = 123;
        ++*(_QWORD *)(v42 + 32);
      }
      v44 = sub_CB6200(v42, v48, v49);
      v45 = *(_BYTE **)(v44 + 32);
      if ( *(_BYTE **)(v44 + 24) == v45 )
      {
        sub_CB6200(v44, (unsigned __int8 *)"}", 1u);
      }
      else
      {
        *v45 = 125;
        ++*(_QWORD *)(v44 + 32);
      }
    }
  }
  v31 = *(_QWORD *)a1;
  v32 = *(_WORD **)(*(_QWORD *)a1 + 32LL);
  v33 = *(_QWORD *)(*(_QWORD *)a1 + 24LL) - (_QWORD)v32;
  if ( *(_BYTE *)(a1 + 16) )
  {
    if ( v33 <= 0xD )
    {
      sub_CB6200(v31, "</tr></table>>", 0xEu);
    }
    else
    {
      qmemcpy(v32, "</tr></table>>", 14);
      *(_QWORD *)(v31 + 32) += 14LL;
    }
  }
  else if ( v33 <= 1 )
  {
    sub_CB6200(v31, (unsigned __int8 *)"}\"", 2u);
  }
  else
  {
    *v32 = 8829;
    *(_QWORD *)(v31 + 32) += 2LL;
  }
  v34 = *(_QWORD *)a1;
  v35 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  if ( (unsigned __int64)(*(_QWORD *)(*(_QWORD *)a1 + 24LL) - v35) <= 2 )
  {
    sub_CB6200(v34, (unsigned __int8 *)"];\n", 3u);
  }
  else
  {
    *(_BYTE *)(v35 + 2) = 10;
    *(_WORD *)v35 = 15197;
    *(_QWORD *)(v34 + 32) += 3LL;
  }
  v36 = *(__int64 ***)(a2 + 40);
  v46 = &v36[*(unsigned int *)(a2 + 48)];
  if ( v46 != v36 )
  {
    v47 = a2;
    v37 = 0;
    do
    {
      if ( sub_30B3230((char *)(a1 + 17), **v36, **(_QWORD **)(a1 + 8)) )
      {
        ++v36;
        ++v37;
        if ( v46 == v36 )
          goto LABEL_42;
      }
      else
      {
        v38 = (__int64 *)v36;
        v39 = v37;
        ++v36;
        ++v37;
        sub_30B4C10((_QWORD **)a1, v47, v39, v38, sub_30B3080);
        if ( v46 == v36 )
          goto LABEL_42;
      }
    }
    while ( v37 != 64 );
    do
    {
      if ( !sub_30B3230((char *)(a1 + 17), **v36, **(_QWORD **)(a1 + 8)) )
        sub_30B4C10((_QWORD **)a1, v47, 64, (__int64 *)v36, sub_30B3080);
      ++v36;
    }
    while ( v46 != v36 );
  }
LABEL_42:
  v51 = (unsigned __int8 *)&unk_49DD210;
  sub_CB5840((__int64)&v51);
  if ( v48 != v50 )
    j_j___libc_free_0((unsigned __int64)v48);
}
