// Function: sub_E93650
// Address: 0xe93650
//
_BYTE *__fastcall sub_E93650(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  void *v9; // rdx
  int v10; // eax
  _WORD *v11; // rdx
  _BYTE *v12; // rax
  int v13; // edx
  int v14; // edx
  unsigned __int64 v15; // rcx
  _BYTE *v16; // rax
  _BYTE *v17; // rax
  unsigned __int64 v18; // rdx
  __int64 v19; // r8
  int v20; // eax
  unsigned __int64 v21; // rdx
  __int64 v22; // rsi
  __int64 v23; // rax
  _BYTE *result; // rax
  int v25; // edx
  __int64 v26; // rdi
  void *v27; // rdx
  __int64 v28; // rdi
  _BYTE *v29; // rax
  __int64 v30; // r13
  void *v31; // rdi
  unsigned __int64 v32; // r15
  unsigned __int8 *v33; // rsi
  int v34; // edx
  unsigned __int64 v35; // rdx
  _BYTE *v36; // rax
  __int64 v37; // rax
  unsigned __int64 v38; // rax
  size_t *v39; // rsi
  size_t v40; // rdx
  void *v41; // rsi
  __int64 v42; // rax
  size_t *v43; // rsi
  size_t v44; // rdx
  void *v45; // rsi
  __int64 v46; // rdi
  __int64 v47; // rdx
  _BYTE *v48; // rax
  __m128i v49; // xmm0
  __m128i si128; // xmm0
  __m128i v51; // xmm0
  __int64 v52; // [rsp+8h] [rbp-68h] BYREF
  _QWORD v53[4]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v54; // [rsp+30h] [rbp-40h]

  if ( (unsigned __int8)sub_E93620(a1, *(_QWORD *)(a1 + 128), *(_QWORD *)(a1 + 136), a2) )
  {
    v29 = *(_BYTE **)(a4 + 32);
    if ( (unsigned __int64)v29 >= *(_QWORD *)(a4 + 24) )
    {
      v30 = sub_CB5D20(a4, 9);
    }
    else
    {
      v30 = a4;
      *(_QWORD *)(a4 + 32) = v29 + 1;
      *v29 = 9;
    }
    v31 = *(void **)(v30 + 32);
    v32 = *(_QWORD *)(a1 + 136);
    v33 = *(unsigned __int8 **)(a1 + 128);
    if ( v32 > *(_QWORD *)(v30 + 24) - (_QWORD)v31 )
    {
      sub_CB6200(v30, v33, *(_QWORD *)(a1 + 136));
    }
    else if ( v32 )
    {
      memcpy(v31, v33, *(_QWORD *)(a1 + 136));
      *(_QWORD *)(v30 + 32) += v32;
    }
    if ( !a5 )
    {
LABEL_71:
      result = *(_BYTE **)(a4 + 32);
      if ( (unsigned __int64)result < *(_QWORD *)(a4 + 24) )
      {
        *(_QWORD *)(a4 + 32) = result + 1;
        *result = 10;
        return result;
      }
      return (_BYTE *)sub_CB5D20(a4, 10);
    }
    v36 = *(_BYTE **)(a4 + 32);
    if ( (unsigned __int64)v36 >= *(_QWORD *)(a4 + 24) )
    {
      v28 = sub_CB5D20(a4, 9);
    }
    else
    {
      v28 = a4;
      *(_QWORD *)(a4 + 32) = v36 + 1;
      *v36 = 9;
    }
LABEL_131:
    sub_CB59D0(v28, a5);
    goto LABEL_71;
  }
  v9 = *(void **)(a4 + 32);
  if ( *(_QWORD *)(a4 + 24) - (_QWORD)v9 <= 9u )
  {
    sub_CB6200(a4, "\t.section\t", 0xAu);
  }
  else
  {
    qmemcpy(v9, "\t.section\t", 10);
    *(_QWORD *)(a4 + 32) += 10LL;
  }
  sub_E933A0(a4, *(void **)(a1 + 128), *(_QWORD *)(a1 + 136));
  if ( *(_BYTE *)(a2 + 257) )
  {
    v10 = *(_DWORD *)(a1 + 152);
    if ( (v10 & 0x10) == 0 )
    {
      v35 = *(_QWORD *)(a4 + 32);
      if ( (v10 & 2) != 0 )
      {
        if ( *(_QWORD *)(a4 + 24) - v35 <= 6 )
        {
          sub_CB6200(a4, ",#alloc", 7u);
          v35 = *(_QWORD *)(a4 + 32);
        }
        else
        {
          *(_DWORD *)v35 = 1818305324;
          *(_WORD *)(v35 + 4) = 28524;
          *(_BYTE *)(v35 + 6) = 99;
          v35 = *(_QWORD *)(a4 + 32) + 7LL;
          *(_QWORD *)(a4 + 32) = v35;
        }
        v10 = *(_DWORD *)(a1 + 152);
      }
      if ( (v10 & 4) != 0 )
      {
        if ( *(_QWORD *)(a4 + 24) - v35 <= 0xA )
        {
          sub_CB6200(a4, ",#execinstr", 0xBu);
          v35 = *(_QWORD *)(a4 + 32);
        }
        else
        {
          qmemcpy((void *)v35, ",#execinstr", 11);
          v35 = *(_QWORD *)(a4 + 32) + 11LL;
          *(_QWORD *)(a4 + 32) = v35;
        }
        v10 = *(_DWORD *)(a1 + 152);
      }
      if ( (v10 & 1) != 0 )
      {
        if ( *(_QWORD *)(a4 + 24) - v35 <= 6 )
        {
          sub_CB6200(a4, ",#write", 7u);
          v35 = *(_QWORD *)(a4 + 32);
        }
        else
        {
          *(_DWORD *)v35 = 1920410412;
          *(_WORD *)(v35 + 4) = 29801;
          *(_BYTE *)(v35 + 6) = 101;
          v35 = *(_QWORD *)(a4 + 32) + 7LL;
          *(_QWORD *)(a4 + 32) = v35;
        }
        v10 = *(_DWORD *)(a1 + 152);
      }
      if ( v10 < 0 )
      {
        if ( *(_QWORD *)(a4 + 24) - v35 <= 8 )
        {
          sub_CB6200(a4, ",#exclude", 9u);
          v35 = *(_QWORD *)(a4 + 32);
        }
        else
        {
          *(_BYTE *)(v35 + 8) = 101;
          *(_QWORD *)v35 = 0x64756C637865232CLL;
          v35 = *(_QWORD *)(a4 + 32) + 9LL;
          *(_QWORD *)(a4 + 32) = v35;
        }
        v10 = *(_DWORD *)(a1 + 152);
      }
      if ( (v10 & 0x400) != 0 )
      {
        if ( *(_QWORD *)(a4 + 24) - v35 <= 4 )
        {
          sub_CB6200(a4, ",#tls", 5u);
          v35 = *(_QWORD *)(a4 + 32);
        }
        else
        {
          *(_DWORD *)v35 = 1819550508;
          *(_BYTE *)(v35 + 4) = 115;
          v35 = *(_QWORD *)(a4 + 32) + 5LL;
          *(_QWORD *)(a4 + 32) = v35;
        }
      }
      if ( *(_QWORD *)(a4 + 24) > v35 )
      {
        *(_QWORD *)(a4 + 32) = v35 + 1;
        *(_BYTE *)v35 = 10;
        return (_BYTE *)(v35 + 1);
      }
      return (_BYTE *)sub_CB5D20(a4, 10);
    }
  }
  v11 = *(_WORD **)(a4 + 32);
  if ( *(_QWORD *)(a4 + 24) - (_QWORD)v11 <= 1u )
  {
    sub_CB6200(a4, (unsigned __int8 *)",\"", 2u);
    v12 = *(_BYTE **)(a4 + 32);
  }
  else
  {
    *v11 = 8748;
    v12 = (_BYTE *)(*(_QWORD *)(a4 + 32) + 2LL);
    *(_QWORD *)(a4 + 32) = v12;
  }
  v13 = *(_DWORD *)(a1 + 152);
  if ( (v13 & 2) != 0 )
  {
    if ( *(_QWORD *)(a4 + 24) > (unsigned __int64)v12 )
    {
      *(_QWORD *)(a4 + 32) = v12 + 1;
      *v12 = 97;
      v13 = *(_DWORD *)(a1 + 152);
      v12 = *(_BYTE **)(a4 + 32);
      if ( v13 >= 0 )
        goto LABEL_11;
      goto LABEL_110;
    }
    sub_CB5D20(a4, 97);
    v13 = *(_DWORD *)(a1 + 152);
    v12 = *(_BYTE **)(a4 + 32);
  }
  if ( v13 >= 0 )
    goto LABEL_11;
LABEL_110:
  if ( *(_QWORD *)(a4 + 24) <= (unsigned __int64)v12 )
  {
    sub_CB5D20(a4, 101);
  }
  else
  {
    *(_QWORD *)(a4 + 32) = v12 + 1;
    *v12 = 101;
  }
  v13 = *(_DWORD *)(a1 + 152);
  v12 = *(_BYTE **)(a4 + 32);
LABEL_11:
  if ( (v13 & 4) != 0 )
  {
    if ( *(_QWORD *)(a4 + 24) <= (unsigned __int64)v12 )
    {
      sub_CB5D20(a4, 120);
    }
    else
    {
      *(_QWORD *)(a4 + 32) = v12 + 1;
      *v12 = 120;
    }
    v13 = *(_DWORD *)(a1 + 152);
    v12 = *(_BYTE **)(a4 + 32);
  }
  if ( (v13 & 1) != 0 )
  {
    if ( *(_QWORD *)(a4 + 24) > (unsigned __int64)v12 )
    {
      *(_QWORD *)(a4 + 32) = v12 + 1;
      *v12 = 119;
      v13 = *(_DWORD *)(a1 + 152);
      v12 = *(_BYTE **)(a4 + 32);
      if ( (v13 & 0x10) == 0 )
        goto LABEL_14;
      goto LABEL_115;
    }
    sub_CB5D20(a4, 119);
    v13 = *(_DWORD *)(a1 + 152);
    v12 = *(_BYTE **)(a4 + 32);
  }
  if ( (v13 & 0x10) == 0 )
    goto LABEL_14;
LABEL_115:
  if ( *(_QWORD *)(a4 + 24) <= (unsigned __int64)v12 )
  {
    sub_CB5D20(a4, 77);
  }
  else
  {
    *(_QWORD *)(a4 + 32) = v12 + 1;
    *v12 = 77;
  }
  v13 = *(_DWORD *)(a1 + 152);
  v12 = *(_BYTE **)(a4 + 32);
LABEL_14:
  if ( (v13 & 0x20) != 0 )
  {
    if ( *(_QWORD *)(a4 + 24) > (unsigned __int64)v12 )
    {
      *(_QWORD *)(a4 + 32) = v12 + 1;
      *v12 = 83;
      v13 = *(_DWORD *)(a1 + 152);
      v12 = *(_BYTE **)(a4 + 32);
      if ( (v13 & 0x400) == 0 )
        goto LABEL_16;
      goto LABEL_103;
    }
    sub_CB5D20(a4, 83);
    v13 = *(_DWORD *)(a1 + 152);
    v12 = *(_BYTE **)(a4 + 32);
  }
  if ( (v13 & 0x400) == 0 )
    goto LABEL_16;
LABEL_103:
  if ( *(_QWORD *)(a4 + 24) > (unsigned __int64)v12 )
  {
    *(_QWORD *)(a4 + 32) = v12 + 1;
    *v12 = 84;
    v13 = *(_DWORD *)(a1 + 152);
    v12 = *(_BYTE **)(a4 + 32);
    if ( (v13 & 0x80u) == 0 )
      goto LABEL_17;
    goto LABEL_105;
  }
  sub_CB5D20(a4, 84);
  v13 = *(_DWORD *)(a1 + 152);
  v12 = *(_BYTE **)(a4 + 32);
LABEL_16:
  if ( (v13 & 0x80u) == 0 )
    goto LABEL_17;
LABEL_105:
  if ( *(_QWORD *)(a4 + 24) <= (unsigned __int64)v12 )
  {
    sub_CB5D20(a4, 111);
  }
  else
  {
    *(_QWORD *)(a4 + 32) = v12 + 1;
    *v12 = 111;
  }
  v13 = *(_DWORD *)(a1 + 152);
  v12 = *(_BYTE **)(a4 + 32);
LABEL_17:
  if ( (v13 & 0x200) != 0 )
  {
    if ( *(_QWORD *)(a4 + 24) <= (unsigned __int64)v12 )
    {
      sub_CB5D20(a4, 71);
    }
    else
    {
      *(_QWORD *)(a4 + 32) = v12 + 1;
      *v12 = 71;
    }
    v13 = *(_DWORD *)(a1 + 152);
    v12 = *(_BYTE **)(a4 + 32);
  }
  if ( (v13 & 0x200000) != 0 )
  {
    if ( (unsigned __int64)v12 >= *(_QWORD *)(a4 + 24) )
    {
      sub_CB5D20(a4, 82);
    }
    else
    {
      *(_QWORD *)(a4 + 32) = v12 + 1;
      *v12 = 82;
    }
    v12 = *(_BYTE **)(a4 + 32);
  }
  if ( *(_DWORD *)(a3 + 44) == 12 && (*(_BYTE *)(a1 + 154) & 0x10) != 0 )
  {
    if ( *(_QWORD *)(a4 + 24) <= (unsigned __int64)v12 )
    {
      sub_CB5D20(a4, 82);
    }
    else
    {
      *(_QWORD *)(a4 + 32) = v12 + 1;
      *v12 = 82;
    }
    v12 = *(_BYTE **)(a4 + 32);
  }
  v14 = *(_DWORD *)(a3 + 32);
  v15 = *(_QWORD *)(a4 + 24);
  if ( v14 == 40 )
  {
    v34 = *(_DWORD *)(a1 + 152);
    if ( (v34 & 0x20000000) != 0 )
    {
      if ( (unsigned __int64)v12 >= v15 )
      {
        sub_CB5D20(a4, 99);
      }
      else
      {
        *(_QWORD *)(a4 + 32) = v12 + 1;
        *v12 = 99;
      }
      v34 = *(_DWORD *)(a1 + 152);
      v12 = *(_BYTE **)(a4 + 32);
    }
    if ( (v34 & 0x10000000) != 0 )
    {
      if ( *(_QWORD *)(a4 + 24) <= (unsigned __int64)v12 )
      {
        sub_CB5D20(a4, 100);
      }
      else
      {
        *(_QWORD *)(a4 + 32) = v12 + 1;
        *v12 = 100;
      }
      v12 = *(_BYTE **)(a4 + 32);
    }
  }
  else if ( (unsigned int)(v14 - 1) <= 1 || (unsigned int)(v14 - 36) <= 1 || (unsigned int)(v14 - 3) <= 2 )
  {
    if ( (*(_BYTE *)(a1 + 155) & 0x20) != 0 )
    {
      if ( (unsigned __int64)v12 >= v15 )
      {
        sub_CB5D20(a4, 121);
      }
      else
      {
        *(_QWORD *)(a4 + 32) = v12 + 1;
        *v12 = 121;
      }
      v12 = *(_BYTE **)(a4 + 32);
    }
  }
  else if ( v14 == 12 )
  {
    if ( (*(_BYTE *)(a1 + 155) & 0x10) != 0 )
    {
      if ( (unsigned __int64)v12 >= v15 )
      {
        sub_CB5D20(a4, 115);
      }
      else
      {
        *(_QWORD *)(a4 + 32) = v12 + 1;
        *v12 = 115;
      }
      v12 = *(_BYTE **)(a4 + 32);
    }
  }
  else if ( v14 == 39 && (*(_BYTE *)(a1 + 155) & 0x10) != 0 )
  {
    if ( (unsigned __int64)v12 >= v15 )
    {
      sub_CB5D20(a4, 108);
    }
    else
    {
      *(_QWORD *)(a4 + 32) = v12 + 1;
      *v12 = 108;
    }
    v12 = *(_BYTE **)(a4 + 32);
  }
  if ( *(_QWORD *)(a4 + 24) <= (unsigned __int64)v12 )
  {
    sub_CB5D20(a4, 34);
  }
  else
  {
    *(_QWORD *)(a4 + 32) = v12 + 1;
    *v12 = 34;
  }
  v16 = *(_BYTE **)(a4 + 32);
  if ( (unsigned __int64)v16 >= *(_QWORD *)(a4 + 24) )
  {
    sub_CB5D20(a4, 44);
  }
  else
  {
    *(_QWORD *)(a4 + 32) = v16 + 1;
    *v16 = 44;
  }
  v17 = *(_BYTE **)(a4 + 32);
  v18 = *(_QWORD *)(a4 + 24);
  if ( **(_BYTE **)(a2 + 48) == 64 )
  {
    if ( v18 <= (unsigned __int64)v17 )
    {
      sub_CB5D20(a4, 37);
    }
    else
    {
      *(_QWORD *)(a4 + 32) = v17 + 1;
      *v17 = 37;
    }
  }
  else if ( v18 <= (unsigned __int64)v17 )
  {
    sub_CB5D20(a4, 64);
  }
  else
  {
    *(_QWORD *)(a4 + 32) = v17 + 1;
    *v17 = 64;
  }
  v19 = *(_QWORD *)(a4 + 32);
  v20 = *(_DWORD *)(a1 + 148);
  v21 = *(_QWORD *)(a4 + 24) - v19;
  switch ( v20 )
  {
    case 14:
      v37 = 0x7272615F74696E69LL;
      if ( v21 <= 9 )
      {
        sub_CB6200(a4, (unsigned __int8 *)"init_array", 0xAu);
        result = *(_BYTE **)(a4 + 32);
        break;
      }
LABEL_139:
      *(_QWORD *)v19 = v37;
      *(_WORD *)(v19 + 8) = 31073;
      result = (_BYTE *)(*(_QWORD *)(a4 + 32) + 10LL);
      *(_QWORD *)(a4 + 32) = result;
      break;
    case 15:
      v37 = 0x7272615F696E6966LL;
      if ( v21 <= 9 )
      {
        sub_CB6200(a4, (unsigned __int8 *)"fini_array", 0xAu);
        result = *(_BYTE **)(a4 + 32);
        break;
      }
      goto LABEL_139;
    case 16:
      if ( v21 <= 0xC )
      {
        sub_CB6200(a4, (unsigned __int8 *)"preinit_array", 0xDu);
        result = *(_BYTE **)(a4 + 32);
      }
      else
      {
        qmemcpy((void *)v19, "preinit_array", 13);
        result = (_BYTE *)(*(_QWORD *)(a4 + 32) + 13LL);
        *(_QWORD *)(a4 + 32) = result;
      }
      break;
    case 8:
      if ( v21 <= 5 )
      {
        sub_CB6200(a4, "nobits", 6u);
        result = *(_BYTE **)(a4 + 32);
      }
      else
      {
        *(_DWORD *)v19 = 1768058734;
        *(_WORD *)(v19 + 4) = 29556;
        result = (_BYTE *)(*(_QWORD *)(a4 + 32) + 6LL);
        *(_QWORD *)(a4 + 32) = result;
      }
      break;
    case 7:
      if ( v21 <= 3 )
      {
        sub_CB6200(a4, (unsigned __int8 *)"note", 4u);
        result = *(_BYTE **)(a4 + 32);
      }
      else
      {
        *(_DWORD *)v19 = 1702129518;
        result = (_BYTE *)(*(_QWORD *)(a4 + 32) + 4LL);
        *(_QWORD *)(a4 + 32) = result;
      }
      break;
    case 1:
      if ( v21 <= 7 )
      {
        sub_CB6200(a4, "progbits", 8u);
        result = *(_BYTE **)(a4 + 32);
      }
      else
      {
        *(_QWORD *)v19 = 0x73746962676F7270LL;
        result = (_BYTE *)(*(_QWORD *)(a4 + 32) + 8LL);
        *(_QWORD *)(a4 + 32) = result;
      }
      break;
    case 1879048193:
      if ( v21 <= 5 )
      {
        sub_CB6200(a4, (unsigned __int8 *)"unwind", 6u);
        result = *(_BYTE **)(a4 + 32);
      }
      else
      {
        *(_DWORD *)v19 = 1769434741;
        *(_WORD *)(v19 + 4) = 25710;
        result = (_BYTE *)(*(_QWORD *)(a4 + 32) + 6LL);
        *(_QWORD *)(a4 + 32) = result;
      }
      break;
    case 1879048222:
      if ( v21 <= 9 )
      {
        sub_CB6200(a4, "0x7000001e", 0xAu);
        result = *(_BYTE **)(a4 + 32);
      }
      else
      {
        qmemcpy((void *)v19, "0x7000001e", 10);
        result = (_BYTE *)(*(_QWORD *)(a4 + 32) + 10LL);
        *(_QWORD *)(a4 + 32) = result;
      }
      break;
    case 1879002112:
      if ( v21 <= 0xA )
      {
        sub_CB6200(a4, "llvm_odrtab", 0xBu);
        result = *(_BYTE **)(a4 + 32);
      }
      else
      {
        qmemcpy((void *)v19, "llvm_odrtab", 11);
        result = (_BYTE *)(*(_QWORD *)(a4 + 32) + 11LL);
        *(_QWORD *)(a4 + 32) = result;
      }
      break;
    case 1879002113:
      if ( v21 <= 0x12 )
      {
        sub_CB6200(a4, "llvm_linker_options", 0x13u);
        result = *(_BYTE **)(a4 + 32);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_3F816F0);
        *(_BYTE *)(v19 + 18) = 115;
        *(_WORD *)(v19 + 16) = 28271;
        *(__m128i *)v19 = si128;
        result = (_BYTE *)(*(_QWORD *)(a4 + 32) + 19LL);
        *(_QWORD *)(a4 + 32) = result;
      }
      break;
    case 1879002121:
      if ( v21 <= 0x16 )
      {
        sub_CB6200(a4, "llvm_call_graph_profile", 0x17u);
        result = *(_BYTE **)(a4 + 32);
      }
      else
      {
        v49 = _mm_load_si128((const __m128i *)&xmmword_3F81700);
        *(_BYTE *)(v19 + 22) = 101;
        *(_DWORD *)(v19 + 16) = 1718579824;
        *(_WORD *)(v19 + 20) = 27753;
        *(__m128i *)v19 = v49;
        result = (_BYTE *)(*(_QWORD *)(a4 + 32) + 23LL);
        *(_QWORD *)(a4 + 32) = result;
      }
      break;
    case 1879002116:
      if ( v21 <= 0x17 )
      {
        sub_CB6200(a4, "llvm_dependent_libraries", 0x18u);
        result = *(_BYTE **)(a4 + 32);
      }
      else
      {
        v51 = _mm_load_si128((const __m128i *)&xmmword_3F81710);
        *(_QWORD *)(v19 + 16) = 0x7365697261726269LL;
        *(__m128i *)v19 = v51;
        result = (_BYTE *)(*(_QWORD *)(a4 + 32) + 24LL);
        *(_QWORD *)(a4 + 32) = result;
      }
      break;
    case 1879002117:
      if ( v21 <= 0xB )
      {
        sub_CB6200(a4, (unsigned __int8 *)"llvm_sympart", 0xCu);
        result = *(_BYTE **)(a4 + 32);
      }
      else
      {
        qmemcpy((void *)v19, "llvm_sympart", 12);
        result = (_BYTE *)(*(_QWORD *)(a4 + 32) + 12LL);
        *(_QWORD *)(a4 + 32) = result;
      }
      break;
    case 1879002122:
      if ( v21 <= 0xF )
      {
        sub_CB6200(a4, (unsigned __int8 *)"llvm_bb_addr_map", 0x10u);
        result = *(_BYTE **)(a4 + 32);
      }
      else
      {
        *(__m128i *)v19 = _mm_load_si128((const __m128i *)&xmmword_3F81720);
        result = (_BYTE *)(*(_QWORD *)(a4 + 32) + 16LL);
        *(_QWORD *)(a4 + 32) = result;
      }
      break;
    default:
      switch ( v20 )
      {
        case 1879002120:
          if ( v21 <= 0x12 )
          {
            sub_CB6200(a4, "llvm_bb_addr_map_v0", 0x13u);
          }
          else
          {
            qmemcpy((void *)v19, "llvm_bb_addr_map_v0", 0x13u);
            *(_QWORD *)(a4 + 32) += 19LL;
          }
          break;
        case 1879002123:
          if ( v21 <= 0xE )
          {
            sub_CB6200(a4, "llvm_offloading", 0xFu);
          }
          else
          {
            qmemcpy((void *)v19, "llvm_offloading", 0xFu);
            *(_QWORD *)(a4 + 32) += 15LL;
          }
          break;
        case 1879002124:
          if ( v21 <= 7 )
          {
            sub_CB6200(a4, "llvm_lto", 8u);
          }
          else
          {
            *(_QWORD *)v19 = 0x6F746C5F6D766C6CLL;
            *(_QWORD *)(a4 + 32) += 8LL;
          }
          break;
        case 1879002125:
          if ( v21 <= 0xC )
          {
            sub_CB6200(a4, "llvm_jt_sizes", 0xDu);
          }
          else
          {
            qmemcpy((void *)v19, "llvm_jt_sizes", 0xDu);
            *(_QWORD *)(a4 + 32) += 13LL;
          }
          break;
        default:
          if ( v21 <= 1 )
          {
            v22 = sub_CB6200(a4, (unsigned __int8 *)"0x", 2u);
          }
          else
          {
            *(_WORD *)v19 = 30768;
            v22 = a4;
            *(_QWORD *)(a4 + 32) += 2LL;
          }
          v23 = *(unsigned int *)(a1 + 148);
          v53[2] = 0;
          v54 = 271;
          v52 = v23;
          v53[0] = &v52;
          sub_CA0E80((__int64)v53, v22);
          break;
      }
      result = *(_BYTE **)(a4 + 32);
      break;
  }
  if ( *(_DWORD *)(a1 + 160) )
  {
    if ( *(_BYTE **)(a4 + 24) == result )
    {
      v46 = sub_CB6200(a4, (unsigned __int8 *)",", 1u);
    }
    else
    {
      *result = 44;
      v46 = a4;
      ++*(_QWORD *)(a4 + 32);
    }
    sub_CB59D0(v46, *(unsigned int *)(a1 + 160));
    result = *(_BYTE **)(a4 + 32);
  }
  v25 = *(_DWORD *)(a1 + 152);
  if ( (v25 & 0x80u) != 0 )
  {
    if ( *(_BYTE **)(a4 + 24) == result )
    {
      sub_CB6200(a4, (unsigned __int8 *)",", 1u);
    }
    else
    {
      *result = 44;
      ++*(_QWORD *)(a4 + 32);
    }
    v42 = *(_QWORD *)(a1 + 176);
    if ( v42 )
    {
      if ( (*(_BYTE *)(v42 + 8) & 1) != 0 )
      {
        v43 = *(size_t **)(v42 - 8);
        v44 = *v43;
        v45 = v43 + 3;
      }
      else
      {
        v44 = 0;
        v45 = 0;
      }
      sub_E933A0(a4, v45, v44);
      v25 = *(_DWORD *)(a1 + 152);
      result = *(_BYTE **)(a4 + 32);
    }
    else
    {
      v48 = *(_BYTE **)(a4 + 32);
      if ( (unsigned __int64)v48 >= *(_QWORD *)(a4 + 24) )
      {
        sub_CB5D20(a4, 48);
      }
      else
      {
        *(_QWORD *)(a4 + 32) = v48 + 1;
        *v48 = 48;
      }
      v25 = *(_DWORD *)(a1 + 152);
      result = *(_BYTE **)(a4 + 32);
    }
  }
  if ( (v25 & 0x200) != 0 )
  {
    if ( result == *(_BYTE **)(a4 + 24) )
    {
      sub_CB6200(a4, (unsigned __int8 *)",", 1u);
    }
    else
    {
      *result = 44;
      ++*(_QWORD *)(a4 + 32);
    }
    v38 = *(_QWORD *)(a1 + 168) & 0xFFFFFFFFFFFFFFF8LL;
    if ( (*(_BYTE *)(v38 + 8) & 1) != 0 )
    {
      v39 = *(size_t **)(v38 - 8);
      v40 = *v39;
      v41 = v39 + 3;
    }
    else
    {
      v40 = 0;
      v41 = 0;
    }
    sub_E933A0(a4, v41, v40);
    if ( (*(_BYTE *)(a1 + 168) & 4) != 0 )
    {
      v47 = *(_QWORD *)(a4 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(a4 + 24) - v47) > 6 )
      {
        *(_DWORD *)v47 = 1836016428;
        *(_WORD *)(v47 + 4) = 24932;
        *(_BYTE *)(v47 + 6) = 116;
        result = (_BYTE *)(*(_QWORD *)(a4 + 32) + 7LL);
        *(_QWORD *)(a4 + 32) = result;
        goto LABEL_56;
      }
      sub_CB6200(a4, ",comdat", 7u);
    }
    result = *(_BYTE **)(a4 + 32);
  }
LABEL_56:
  if ( *(_DWORD *)(a1 + 156) != -1 )
  {
    if ( *(_QWORD *)(a4 + 24) - (_QWORD)result <= 7u )
    {
      v26 = sub_CB6200(a4, ",unique,", 8u);
    }
    else
    {
      v26 = a4;
      *(_QWORD *)result = 0x2C657571696E752CLL;
      *(_QWORD *)(a4 + 32) += 8LL;
    }
    sub_CB59D0(v26, *(unsigned int *)(a1 + 156));
    result = *(_BYTE **)(a4 + 32);
  }
  if ( (unsigned __int64)result >= *(_QWORD *)(a4 + 24) )
  {
    result = (_BYTE *)sub_CB5D20(a4, 10);
  }
  else
  {
    *(_QWORD *)(a4 + 32) = result + 1;
    *result = 10;
  }
  if ( a5 )
  {
    v27 = *(void **)(a4 + 32);
    if ( *(_QWORD *)(a4 + 24) - (_QWORD)v27 <= 0xCu )
    {
      v28 = sub_CB6200(a4, "\t.subsection\t", 0xDu);
    }
    else
    {
      v28 = a4;
      qmemcpy(v27, "\t.subsection\t", 13);
      *(_QWORD *)(a4 + 32) += 13LL;
    }
    goto LABEL_131;
  }
  return result;
}
