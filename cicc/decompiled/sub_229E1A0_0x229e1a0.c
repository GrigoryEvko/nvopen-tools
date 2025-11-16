// Function: sub_229E1A0
// Address: 0x229e1a0
//
void __fastcall sub_229E1A0(_BYTE *a1, unsigned __int64 a2)
{
  __int64 v4; // rdi
  __int64 v5; // rdx
  __int64 v6; // rax
  __int64 v7; // r9
  _QWORD *v8; // rdx
  __int64 v9; // rdi
  __int64 v10; // rdx
  unsigned __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rdx
  __int64 v14; // rdi
  __m128i *v15; // r8
  unsigned __int64 v16; // rcx
  __int64 v17; // rax
  __int64 v18; // rdx
  int v19; // esi
  int v20; // r12d
  unsigned int v21; // r12d
  __m128i si128; // xmm0
  __int64 v23; // rdx
  __m128i v24; // xmm0
  __int64 v25; // rax
  _WORD *v26; // rdx
  __int64 v27; // r12
  unsigned __int8 *v28; // rsi
  __int64 v29; // rax
  __int64 v30; // rdx
  unsigned __int8 *v31; // rdi
  __int64 v32; // rdi
  _WORD *v33; // rdx
  unsigned __int64 v34; // rax
  __int64 v35; // rdi
  __int64 v36; // rdx
  unsigned __int64 *v37; // r15
  unsigned __int64 *v38; // r12
  int v39; // r13d
  __int64 v40; // rdi
  unsigned __int64 v41; // r8
  _QWORD *v42; // rdx
  __int64 v43; // rdi
  _WORD *v44; // rdx
  unsigned __int64 v45; // r8
  __int64 v46; // rdi
  __int64 v47; // rdx
  __int64 v48; // rax
  __int64 v49; // rax
  __int64 v50; // rdi
  _BYTE *v51; // rax
  __int64 v52; // rdi
  _BYTE *v53; // rax
  __int64 v54; // rdi
  _QWORD *v55; // rdx
  __int64 v56; // rdi
  _WORD *v57; // rdx
  unsigned __int64 v58; // r13
  __int64 v59; // rdi
  __int64 v60; // rdx
  __int64 v61; // rdi
  _BYTE *v62; // rax
  __int64 v63; // rdi
  _BYTE *v64; // rax
  __int64 v65; // rax
  __int64 v66; // rdi
  _BYTE *v67; // rax
  __int64 v68; // rdi
  _BYTE *v69; // rax
  __int64 v70; // rdi
  _BYTE *v71; // rax
  __int64 v72; // rax
  __m128i v73; // xmm0
  __int64 v74; // rax
  __m128i v75; // xmm0
  unsigned __int64 v76; // [rsp+18h] [rbp-E8h]
  unsigned __int64 v77; // [rsp+28h] [rbp-D8h]
  unsigned __int64 v78; // [rsp+28h] [rbp-D8h]
  unsigned __int8 *v79; // [rsp+50h] [rbp-B0h] BYREF
  size_t v80; // [rsp+58h] [rbp-A8h]
  _BYTE v81[16]; // [rsp+60h] [rbp-A0h] BYREF
  unsigned __int64 v82; // [rsp+70h] [rbp-90h] BYREF
  unsigned __int8 *v83; // [rsp+78h] [rbp-88h]
  _QWORD v84[2]; // [rsp+80h] [rbp-80h] BYREF
  unsigned __int64 v85; // [rsp+90h] [rbp-70h] BYREF
  unsigned __int8 *v86; // [rsp+98h] [rbp-68h]
  _QWORD v87[12]; // [rsp+A0h] [rbp-60h] BYREF

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
  v8 = *(_QWORD **)(v6 + 32);
  if ( *(_QWORD *)(v6 + 24) - (_QWORD)v8 <= 7u )
  {
    sub_CB6200(v6, " [shape=", 8u);
  }
  else
  {
    *v8 = 0x3D65706168735B20LL;
    *(_QWORD *)(v6 + 32) += 8LL;
  }
  v9 = *(_QWORD *)a1;
  v10 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  v11 = *(_QWORD *)(*(_QWORD *)a1 + 24LL) - v10;
  if ( a1[16] )
  {
    if ( v11 <= 4 )
    {
      sub_CB6200(v9, (unsigned __int8 *)"none,", 5u);
    }
    else
    {
      *(_DWORD *)v10 = 1701736302;
      *(_BYTE *)(v10 + 4) = 44;
      *(_QWORD *)(v9 + 32) += 5LL;
    }
  }
  else if ( v11 <= 6 )
  {
    sub_CB6200(v9, (unsigned __int8 *)"record,", 7u);
  }
  else
  {
    *(_DWORD *)v10 = 1868785010;
    *(_WORD *)(v10 + 4) = 25714;
    *(_BYTE *)(v10 + 6) = 44;
    *(_QWORD *)(v9 + 32) += 7LL;
  }
  v12 = *(_QWORD *)a1;
  v13 = *(_QWORD *)(*(_QWORD *)a1 + 32LL);
  if ( (unsigned __int64)(*(_QWORD *)(*(_QWORD *)a1 + 24LL) - v13) <= 5 )
  {
    sub_CB6200(v12, "label=", 6u);
  }
  else
  {
    *(_DWORD *)v13 = 1700946284;
    *(_WORD *)(v13 + 4) = 15724;
    *(_QWORD *)(v12 + 32) += 6LL;
  }
  v14 = *(_QWORD *)a1;
  v15 = *(__m128i **)(*(_QWORD *)a1 + 32LL);
  v16 = *(_QWORD *)(*(_QWORD *)a1 + 24LL) - (_QWORD)v15;
  if ( a1[16] )
  {
    v17 = *(_QWORD *)(a2 + 24);
    v18 = v17 + 8LL * *(unsigned int *)(a2 + 32);
    if ( v17 == v18 )
    {
      v21 = 1;
    }
    else
    {
      v19 = 0;
      do
      {
        v17 += 8;
        ++v19;
      }
      while ( v17 != v18 && v19 != 64 );
      v20 = 1;
      if ( v19 )
        v20 = v19;
      v21 = (v17 != v18) + v20;
    }
    if ( v16 <= 0x30 )
    {
      v65 = sub_CB6200(v14, "<<table border=\"0\" cellborder=\"1\" cellspacing=\"0\"", 0x31u);
      v23 = *(_QWORD *)(v65 + 32);
      v14 = v65;
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CB60);
      v15[3].m128i_i8[0] = 34;
      *v15 = si128;
      v15[1] = _mm_load_si128((const __m128i *)&xmmword_3F8CB70);
      v15[2] = _mm_load_si128((const __m128i *)&xmmword_3F8CB80);
      v23 = *(_QWORD *)(v14 + 32) + 49LL;
      *(_QWORD *)(v14 + 32) = v23;
    }
    if ( (unsigned __int64)(*(_QWORD *)(v14 + 24) - v23) <= 0x2E )
    {
      v14 = sub_CB6200(v14, " cellpadding=\"0\"><tr><td align=\"text\" colspan=\"", 0x2Fu);
    }
    else
    {
      v24 = _mm_load_si128((const __m128i *)&xmmword_3F8CB90);
      qmemcpy((void *)(v23 + 32), "text\" colspan=\"", 15);
      *(__m128i *)v23 = v24;
      *(__m128i *)(v23 + 16) = _mm_load_si128((const __m128i *)&xmmword_3F8CBA0);
      *(_QWORD *)(v14 + 32) += 47LL;
    }
    v25 = sub_CB59D0(v14, v21);
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
  }
  else if ( v16 <= 1 )
  {
    sub_CB6200(v14, (unsigned __int8 *)"\"{", 2u);
  }
  else
  {
    v15->m128i_i16[0] = 31522;
    *(_QWORD *)(v14 + 32) += 2LL;
  }
  v27 = *(_QWORD *)a1;
  v28 = *(unsigned __int8 **)a2;
  if ( a1[16] )
  {
    if ( v28 )
    {
      if ( a1[17] )
        sub_11F3900((__int64)&v85, v28);
      else
        sub_11F8430(
          (__int64 *)&v85,
          (__int64)v28,
          0,
          0,
          0,
          v7,
          (void (__fastcall *)(__int64, __int64 *, unsigned int *, __int64))sub_11F32A0,
          (__int64)sub_11F32F0);
    }
    else
    {
      v82 = 24;
      v85 = (unsigned __int64)v87;
      v72 = sub_22409D0((__int64)&v85, &v82, 0);
      v73 = _mm_load_si128((const __m128i *)&xmmword_4289C20);
      v85 = v72;
      v87[0] = v82;
      *(_QWORD *)(v72 + 16) = 0x65646F6E20746F6FLL;
      *(__m128i *)v72 = v73;
      v86 = (unsigned __int8 *)v82;
      *(_BYTE *)(v85 + v82) = 0;
    }
    v29 = sub_CB6200(v27, (unsigned __int8 *)v85, (size_t)v86);
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
    v31 = (unsigned __int8 *)v85;
    if ( (_QWORD *)v85 != v87 )
LABEL_33:
      j_j___libc_free_0((unsigned __int64)v31);
  }
  else
  {
    if ( v28 )
    {
      if ( a1[17] )
        sub_11F3900((__int64)&v82, v28);
      else
        sub_11F8430(
          (__int64 *)&v82,
          (__int64)v28,
          0,
          0,
          0,
          v7,
          (void (__fastcall *)(__int64, __int64 *, unsigned int *, __int64))sub_11F32A0,
          (__int64)sub_11F32F0);
      sub_C67200((__int64 *)&v85, (__int64)&v82);
    }
    else
    {
      v85 = 24;
      v82 = (unsigned __int64)v84;
      v74 = sub_22409D0((__int64)&v82, &v85, 0);
      v75 = _mm_load_si128((const __m128i *)&xmmword_4289C20);
      v82 = v74;
      v84[0] = v85;
      *(_QWORD *)(v74 + 16) = 0x65646F6E20746F6FLL;
      *(__m128i *)v74 = v75;
      v83 = (unsigned __int8 *)v85;
      *(_BYTE *)(v82 + v85) = 0;
      sub_C67200((__int64 *)&v85, (__int64)&v82);
    }
    sub_CB6200(v27, (unsigned __int8 *)v85, (size_t)v86);
    if ( (_QWORD *)v85 != v87 )
      j_j___libc_free_0(v85);
    v31 = (unsigned __int8 *)v82;
    if ( (_QWORD *)v82 != v84 )
      goto LABEL_33;
  }
  v79 = v81;
  v87[3] = 0x100000000LL;
  v80 = 0;
  v81[0] = 0;
  v85 = (unsigned __int64)&unk_49DD210;
  v86 = 0;
  memset(v87, 0, 24);
  v87[4] = &v79;
  sub_CB5980((__int64)&v85, 0, 0, 0);
  if ( (unsigned __int8)sub_229DD50((__int64)a1, (__int64)&v85, a2) )
  {
    if ( a1[16] )
      goto LABEL_36;
    v66 = *(_QWORD *)a1;
    v67 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
    if ( *(_BYTE **)(*(_QWORD *)a1 + 24LL) == v67 )
    {
      sub_CB6200(v66, (unsigned __int8 *)"|", 1u);
    }
    else
    {
      *v67 = 124;
      ++*(_QWORD *)(v66 + 32);
    }
    v68 = *(_QWORD *)a1;
    if ( a1[16] )
    {
LABEL_36:
      sub_CB6200(*(_QWORD *)a1, v79, v80);
    }
    else
    {
      v69 = *(_BYTE **)(v68 + 32);
      if ( *(_BYTE **)(v68 + 24) == v69 )
      {
        v68 = sub_CB6200(v68, (unsigned __int8 *)"{", 1u);
      }
      else
      {
        *v69 = 123;
        ++*(_QWORD *)(v68 + 32);
      }
      v70 = sub_CB6200(v68, v79, v80);
      v71 = *(_BYTE **)(v70 + 32);
      if ( *(_BYTE **)(v70 + 24) == v71 )
      {
        sub_CB6200(v70, (unsigned __int8 *)"}", 1u);
      }
      else
      {
        *v71 = 125;
        ++*(_QWORD *)(v70 + 32);
      }
    }
  }
  v32 = *(_QWORD *)a1;
  v33 = *(_WORD **)(*(_QWORD *)a1 + 32LL);
  v34 = *(_QWORD *)(*(_QWORD *)a1 + 24LL) - (_QWORD)v33;
  if ( a1[16] )
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
  v37 = *(unsigned __int64 **)(a2 + 24);
  v38 = &v37[*(unsigned int *)(a2 + 32)];
  if ( v38 != v37 )
  {
    v76 = a2;
    v39 = 0;
    while ( 1 )
    {
      v45 = *v37;
      if ( *v37 )
      {
        v46 = *(_QWORD *)a1;
        LOBYTE(v84[0]) = 0;
        v82 = (unsigned __int64)v84;
        v83 = 0;
        v47 = *(_QWORD *)(v46 + 32);
        if ( (unsigned __int64)(*(_QWORD *)(v46 + 24) - v47) > 4 )
        {
          *(_DWORD *)v47 = 1685016073;
          *(_BYTE *)(v47 + 4) = 101;
          *(_QWORD *)(v46 + 32) += 5LL;
        }
        else
        {
          v78 = v45;
          v48 = sub_CB6200(v46, "\tNode", 5u);
          v45 = v78;
          v46 = v48;
        }
        v77 = v45;
        sub_CB5A80(v46, v76);
        v40 = *(_QWORD *)a1;
        v41 = v77;
        v42 = *(_QWORD **)(*(_QWORD *)a1 + 32LL);
        if ( *(_QWORD *)(*(_QWORD *)a1 + 24LL) - (_QWORD)v42 <= 7u )
        {
          v49 = sub_CB6200(v40, " -> Node", 8u);
          v41 = v77;
          v40 = v49;
        }
        else
        {
          *v42 = 0x65646F4E203E2D20LL;
          *(_QWORD *)(v40 + 32) += 8LL;
        }
        sub_CB5A80(v40, v41);
        if ( v83 )
        {
          v50 = *(_QWORD *)a1;
          v51 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
          if ( *(_BYTE **)(*(_QWORD *)a1 + 24LL) == v51 )
          {
            v50 = sub_CB6200(v50, (unsigned __int8 *)"[", 1u);
          }
          else
          {
            *v51 = 91;
            ++*(_QWORD *)(v50 + 32);
          }
          v52 = sub_CB6200(v50, (unsigned __int8 *)v82, (size_t)v83);
          v53 = *(_BYTE **)(v52 + 32);
          if ( *(_BYTE **)(v52 + 24) == v53 )
          {
            sub_CB6200(v52, (unsigned __int8 *)"]", 1u);
          }
          else
          {
            *v53 = 93;
            ++*(_QWORD *)(v52 + 32);
          }
        }
        v43 = *(_QWORD *)a1;
        v44 = *(_WORD **)(*(_QWORD *)a1 + 32LL);
        if ( *(_QWORD *)(*(_QWORD *)a1 + 24LL) - (_QWORD)v44 <= 1u )
        {
          sub_CB6200(v43, (unsigned __int8 *)";\n", 2u);
        }
        else
        {
          *v44 = 2619;
          *(_QWORD *)(v43 + 32) += 2LL;
        }
        if ( (_QWORD *)v82 != v84 )
          j_j___libc_free_0(v82);
      }
      ++v37;
      ++v39;
      if ( v37 == v38 )
        break;
      if ( v39 == 64 )
      {
        for ( ; v37 != v38; ++v37 )
        {
          v58 = *v37;
          if ( *v37 )
          {
            v59 = *(_QWORD *)a1;
            LOBYTE(v84[0]) = 0;
            v82 = (unsigned __int64)v84;
            v83 = 0;
            v60 = *(_QWORD *)(v59 + 32);
            if ( (unsigned __int64)(*(_QWORD *)(v59 + 24) - v60) > 4 )
            {
              *(_DWORD *)v60 = 1685016073;
              *(_BYTE *)(v60 + 4) = 101;
              *(_QWORD *)(v59 + 32) += 5LL;
            }
            else
            {
              v59 = sub_CB6200(v59, "\tNode", 5u);
            }
            sub_CB5A80(v59, v76);
            v54 = *(_QWORD *)a1;
            v55 = *(_QWORD **)(*(_QWORD *)a1 + 32LL);
            if ( *(_QWORD *)(*(_QWORD *)a1 + 24LL) - (_QWORD)v55 <= 7u )
            {
              v54 = sub_CB6200(v54, " -> Node", 8u);
            }
            else
            {
              *v55 = 0x65646F4E203E2D20LL;
              *(_QWORD *)(v54 + 32) += 8LL;
            }
            sub_CB5A80(v54, v58);
            if ( v83 )
            {
              v61 = *(_QWORD *)a1;
              v62 = *(_BYTE **)(*(_QWORD *)a1 + 32LL);
              if ( *(_BYTE **)(*(_QWORD *)a1 + 24LL) == v62 )
              {
                v61 = sub_CB6200(v61, (unsigned __int8 *)"[", 1u);
              }
              else
              {
                *v62 = 91;
                ++*(_QWORD *)(v61 + 32);
              }
              v63 = sub_CB6200(v61, (unsigned __int8 *)v82, (size_t)v83);
              v64 = *(_BYTE **)(v63 + 32);
              if ( *(_BYTE **)(v63 + 24) == v64 )
              {
                sub_CB6200(v63, (unsigned __int8 *)"]", 1u);
              }
              else
              {
                *v64 = 93;
                ++*(_QWORD *)(v63 + 32);
              }
            }
            v56 = *(_QWORD *)a1;
            v57 = *(_WORD **)(*(_QWORD *)a1 + 32LL);
            if ( *(_QWORD *)(*(_QWORD *)a1 + 24LL) - (_QWORD)v57 <= 1u )
            {
              sub_CB6200(v56, (unsigned __int8 *)";\n", 2u);
            }
            else
            {
              *v57 = 2619;
              *(_QWORD *)(v56 + 32) += 2LL;
            }
            if ( (_QWORD *)v82 != v84 )
              j_j___libc_free_0(v82);
          }
        }
        break;
      }
    }
  }
  v85 = (unsigned __int64)&unk_49DD210;
  sub_CB5840((__int64)&v85);
  if ( v79 != v81 )
    j_j___libc_free_0((unsigned __int64)v79);
}
