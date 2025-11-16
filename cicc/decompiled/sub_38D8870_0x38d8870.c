// Function: sub_38D8870
// Address: 0x38d8870
//
_BYTE *__fastcall sub_38D8870(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  void *v10; // rdx
  int v11; // eax
  _WORD *v12; // rdx
  _BYTE *v13; // rax
  int v14; // edx
  int v15; // edx
  unsigned __int64 v16; // rsi
  _BYTE *v17; // rax
  _BYTE *v18; // rax
  unsigned __int64 v19; // rdx
  __int64 v20; // rax
  __m128i *v21; // rdx
  __m128i v22; // xmm0
  _BYTE *result; // rax
  _BYTE *v24; // rax
  __int64 v25; // r15
  void *v26; // rdi
  char *v27; // rsi
  size_t v28; // rbx
  _BYTE *v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rax
  int v32; // edx
  unsigned __int64 v33; // rdx
  void *v34; // rdx
  int v35; // edx
  __int64 v36; // rdi
  void *v37; // rdx
  __int64 v38; // rdx
  _DWORD *v39; // rdx
  __int64 v40; // rdi
  _BYTE *v41; // rax
  size_t *v42; // rsi
  size_t v43; // rdx
  void *v44; // rsi
  __int64 v45; // rdx
  _BYTE *v46; // rax
  size_t *v47; // rsi
  size_t v48; // rdx
  void *v49; // rsi
  _QWORD *v50; // rdx
  __int64 v51; // rdx
  void *v52; // rdx
  void *v53; // rdx
  __m128i *v54; // rdx
  __m128i si128; // xmm0
  __int64 v56; // rdx
  __int64 v57; // rcx
  __int64 v58; // [rsp+8h] [rbp-128h] BYREF
  _QWORD v59[2]; // [rsp+10h] [rbp-120h] BYREF
  __m128i v60; // [rsp+20h] [rbp-110h] BYREF
  __int16 v61; // [rsp+30h] [rbp-100h]
  __m128i v62; // [rsp+40h] [rbp-F0h] BYREF
  char v63; // [rsp+50h] [rbp-E0h]
  char v64; // [rsp+51h] [rbp-DFh]
  __m128i v65[2]; // [rsp+60h] [rbp-D0h] BYREF
  __m128i v66; // [rsp+80h] [rbp-B0h] BYREF
  char v67; // [rsp+90h] [rbp-A0h]
  char v68; // [rsp+91h] [rbp-9Fh]
  __m128i v69[2]; // [rsp+A0h] [rbp-90h] BYREF
  __m128i v70; // [rsp+C0h] [rbp-70h] BYREF
  __int16 v71; // [rsp+D0h] [rbp-60h]
  __m128i v72[5]; // [rsp+E0h] [rbp-50h] BYREF

  if ( (unsigned __int8)sub_38D8840(a1, *(_QWORD *)(a1 + 152), *(_QWORD *)(a1 + 160), a2) )
  {
    v24 = *(_BYTE **)(a4 + 24);
    if ( (unsigned __int64)v24 >= *(_QWORD *)(a4 + 16) )
    {
      v25 = sub_16E7DE0(a4, 9);
    }
    else
    {
      v25 = a4;
      *(_QWORD *)(a4 + 24) = v24 + 1;
      *v24 = 9;
    }
    v26 = *(void **)(v25 + 24);
    v27 = *(char **)(a1 + 152);
    v28 = *(_QWORD *)(a1 + 160);
    if ( v28 > *(_QWORD *)(v25 + 16) - (_QWORD)v26 )
    {
      sub_16E7EE0(v25, v27, v28);
      if ( a5 )
      {
LABEL_48:
        v29 = *(_BYTE **)(a4 + 24);
        if ( *(_QWORD *)(a4 + 16) <= (unsigned __int64)v29 )
        {
          sub_16E7DE0(a4, 9);
        }
        else
        {
          *(_QWORD *)(a4 + 24) = v29 + 1;
          *v29 = 9;
        }
        goto LABEL_50;
      }
    }
    else
    {
      if ( v28 )
      {
        memcpy(v26, v27, v28);
        *(_QWORD *)(v25 + 24) += v28;
      }
      if ( a5 )
        goto LABEL_48;
    }
LABEL_51:
    result = *(_BYTE **)(a4 + 24);
    if ( (unsigned __int64)result < *(_QWORD *)(a4 + 16) )
    {
      *(_QWORD *)(a4 + 24) = result + 1;
      *result = 10;
      return result;
    }
    return (_BYTE *)sub_16E7DE0(a4, 10);
  }
  v10 = *(void **)(a4 + 24);
  if ( *(_QWORD *)(a4 + 16) - (_QWORD)v10 <= 9u )
  {
    sub_16E7EE0(a4, "\t.section\t", 0xAu);
  }
  else
  {
    qmemcpy(v10, "\t.section\t", 10);
    *(_QWORD *)(a4 + 24) += 10LL;
  }
  sub_38D85A0(a4, *(void **)(a1 + 152), *(_QWORD *)(a1 + 160));
  if ( *(_BYTE *)(a2 + 280) )
  {
    v11 = *(_DWORD *)(a1 + 172);
    if ( (v11 & 0x10) == 0 )
    {
      v33 = *(_QWORD *)(a4 + 24);
      if ( (v11 & 2) != 0 )
      {
        if ( *(_QWORD *)(a4 + 16) - v33 <= 6 )
        {
          sub_16E7EE0(a4, ",#alloc", 7u);
          v33 = *(_QWORD *)(a4 + 24);
        }
        else
        {
          *(_DWORD *)v33 = 1818305324;
          *(_WORD *)(v33 + 4) = 28524;
          *(_BYTE *)(v33 + 6) = 99;
          v33 = *(_QWORD *)(a4 + 24) + 7LL;
          *(_QWORD *)(a4 + 24) = v33;
        }
        v11 = *(_DWORD *)(a1 + 172);
      }
      if ( (v11 & 4) != 0 )
      {
        if ( *(_QWORD *)(a4 + 16) - v33 <= 0xA )
        {
          sub_16E7EE0(a4, ",#execinstr", 0xBu);
          v33 = *(_QWORD *)(a4 + 24);
        }
        else
        {
          qmemcpy((void *)v33, ",#execinstr", 11);
          v33 = *(_QWORD *)(a4 + 24) + 11LL;
          *(_QWORD *)(a4 + 24) = v33;
        }
        v11 = *(_DWORD *)(a1 + 172);
      }
      if ( (v11 & 1) != 0 )
      {
        if ( *(_QWORD *)(a4 + 16) - v33 <= 6 )
        {
          sub_16E7EE0(a4, ",#write", 7u);
          v33 = *(_QWORD *)(a4 + 24);
        }
        else
        {
          *(_DWORD *)v33 = 1920410412;
          *(_WORD *)(v33 + 4) = 29801;
          *(_BYTE *)(v33 + 6) = 101;
          v33 = *(_QWORD *)(a4 + 24) + 7LL;
          *(_QWORD *)(a4 + 24) = v33;
        }
        v11 = *(_DWORD *)(a1 + 172);
      }
      if ( v11 < 0 )
      {
        if ( *(_QWORD *)(a4 + 16) - v33 <= 8 )
        {
          sub_16E7EE0(a4, ",#exclude", 9u);
          v33 = *(_QWORD *)(a4 + 24);
        }
        else
        {
          *(_BYTE *)(v33 + 8) = 101;
          *(_QWORD *)v33 = 0x64756C637865232CLL;
          v33 = *(_QWORD *)(a4 + 24) + 9LL;
          *(_QWORD *)(a4 + 24) = v33;
        }
        v11 = *(_DWORD *)(a1 + 172);
      }
      if ( (v11 & 0x400) != 0 )
      {
        if ( *(_QWORD *)(a4 + 16) - v33 <= 4 )
        {
          sub_16E7EE0(a4, ",#tls", 5u);
          v33 = *(_QWORD *)(a4 + 24);
        }
        else
        {
          *(_DWORD *)v33 = 1819550508;
          *(_BYTE *)(v33 + 4) = 115;
          v33 = *(_QWORD *)(a4 + 24) + 5LL;
          *(_QWORD *)(a4 + 24) = v33;
        }
      }
      if ( *(_QWORD *)(a4 + 16) > v33 )
      {
        *(_QWORD *)(a4 + 24) = v33 + 1;
        *(_BYTE *)v33 = 10;
        return (_BYTE *)(v33 + 1);
      }
      return (_BYTE *)sub_16E7DE0(a4, 10);
    }
  }
  v12 = *(_WORD **)(a4 + 24);
  if ( *(_QWORD *)(a4 + 16) - (_QWORD)v12 <= 1u )
  {
    sub_16E7EE0(a4, ",\"", 2u);
    v13 = *(_BYTE **)(a4 + 24);
  }
  else
  {
    *v12 = 8748;
    v13 = (_BYTE *)(*(_QWORD *)(a4 + 24) + 2LL);
    *(_QWORD *)(a4 + 24) = v13;
  }
  v14 = *(_DWORD *)(a1 + 172);
  if ( (v14 & 2) != 0 )
  {
    if ( *(_QWORD *)(a4 + 16) > (unsigned __int64)v13 )
    {
      *(_QWORD *)(a4 + 24) = v13 + 1;
      *v13 = 97;
      v14 = *(_DWORD *)(a1 + 172);
      v13 = *(_BYTE **)(a4 + 24);
      if ( v14 >= 0 )
        goto LABEL_11;
      goto LABEL_80;
    }
    sub_16E7DE0(a4, 97);
    v14 = *(_DWORD *)(a1 + 172);
    v13 = *(_BYTE **)(a4 + 24);
  }
  if ( v14 >= 0 )
    goto LABEL_11;
LABEL_80:
  if ( *(_QWORD *)(a4 + 16) <= (unsigned __int64)v13 )
  {
    sub_16E7DE0(a4, 101);
  }
  else
  {
    *(_QWORD *)(a4 + 24) = v13 + 1;
    *v13 = 101;
  }
  v14 = *(_DWORD *)(a1 + 172);
  v13 = *(_BYTE **)(a4 + 24);
LABEL_11:
  if ( (v14 & 4) != 0 )
  {
    if ( (unsigned __int64)v13 >= *(_QWORD *)(a4 + 16) )
    {
      sub_16E7DE0(a4, 120);
    }
    else
    {
      *(_QWORD *)(a4 + 24) = v13 + 1;
      *v13 = 120;
    }
    v14 = *(_DWORD *)(a1 + 172);
    v13 = *(_BYTE **)(a4 + 24);
  }
  if ( (v14 & 0x200) != 0 )
  {
    if ( *(_QWORD *)(a4 + 16) > (unsigned __int64)v13 )
    {
      *(_QWORD *)(a4 + 24) = v13 + 1;
      *v13 = 71;
      v14 = *(_DWORD *)(a1 + 172);
      v13 = *(_BYTE **)(a4 + 24);
      if ( (v14 & 1) == 0 )
        goto LABEL_14;
      goto LABEL_86;
    }
    sub_16E7DE0(a4, 71);
    v14 = *(_DWORD *)(a1 + 172);
    v13 = *(_BYTE **)(a4 + 24);
  }
  if ( (v14 & 1) == 0 )
    goto LABEL_14;
LABEL_86:
  if ( *(_QWORD *)(a4 + 16) <= (unsigned __int64)v13 )
  {
    sub_16E7DE0(a4, 119);
  }
  else
  {
    *(_QWORD *)(a4 + 24) = v13 + 1;
    *v13 = 119;
  }
  v14 = *(_DWORD *)(a1 + 172);
  v13 = *(_BYTE **)(a4 + 24);
LABEL_14:
  if ( (v14 & 0x10) != 0 )
  {
    if ( (unsigned __int64)v13 < *(_QWORD *)(a4 + 16) )
    {
      *(_QWORD *)(a4 + 24) = v13 + 1;
      *v13 = 77;
      v14 = *(_DWORD *)(a1 + 172);
      v13 = *(_BYTE **)(a4 + 24);
      if ( (v14 & 0x20) == 0 )
        goto LABEL_16;
      goto LABEL_70;
    }
    sub_16E7DE0(a4, 77);
    v14 = *(_DWORD *)(a1 + 172);
    v13 = *(_BYTE **)(a4 + 24);
  }
  if ( (v14 & 0x20) == 0 )
    goto LABEL_16;
LABEL_70:
  if ( *(_QWORD *)(a4 + 16) > (unsigned __int64)v13 )
  {
    *(_QWORD *)(a4 + 24) = v13 + 1;
    *v13 = 83;
    v14 = *(_DWORD *)(a1 + 172);
    v13 = *(_BYTE **)(a4 + 24);
    if ( (v14 & 0x400) == 0 )
      goto LABEL_17;
    goto LABEL_72;
  }
  sub_16E7DE0(a4, 83);
  v14 = *(_DWORD *)(a1 + 172);
  v13 = *(_BYTE **)(a4 + 24);
LABEL_16:
  if ( (v14 & 0x400) == 0 )
    goto LABEL_17;
LABEL_72:
  if ( (unsigned __int64)v13 < *(_QWORD *)(a4 + 16) )
  {
    *(_QWORD *)(a4 + 24) = v13 + 1;
    *v13 = 84;
    v13 = *(_BYTE **)(a4 + 24);
    if ( (*(_DWORD *)(a1 + 172) & 0x80) == 0 )
      goto LABEL_18;
    goto LABEL_74;
  }
  sub_16E7DE0(a4, 84);
  v14 = *(_DWORD *)(a1 + 172);
  v13 = *(_BYTE **)(a4 + 24);
LABEL_17:
  if ( (v14 & 0x80) == 0 )
    goto LABEL_18;
LABEL_74:
  if ( *(_QWORD *)(a4 + 16) <= (unsigned __int64)v13 )
  {
    sub_16E7DE0(a4, 111);
  }
  else
  {
    *(_QWORD *)(a4 + 24) = v13 + 1;
    *v13 = 111;
  }
  v13 = *(_BYTE **)(a4 + 24);
LABEL_18:
  v15 = *(_DWORD *)(a3 + 32);
  v16 = *(_QWORD *)(a4 + 16);
  if ( v15 == 33 )
  {
    v32 = *(_DWORD *)(a1 + 172);
    if ( (v32 & 0x20000000) != 0 )
    {
      if ( v16 <= (unsigned __int64)v13 )
      {
        sub_16E7DE0(a4, 99);
      }
      else
      {
        *(_QWORD *)(a4 + 24) = v13 + 1;
        *v13 = 99;
      }
      v32 = *(_DWORD *)(a1 + 172);
      v13 = *(_BYTE **)(a4 + 24);
    }
    if ( (v32 & 0x10000000) != 0 )
    {
      if ( *(_QWORD *)(a4 + 16) <= (unsigned __int64)v13 )
      {
        sub_16E7DE0(a4, 100);
      }
      else
      {
        *(_QWORD *)(a4 + 24) = v13 + 1;
        *v13 = 100;
      }
      v13 = *(_BYTE **)(a4 + 24);
    }
  }
  else if ( ((unsigned int)(v15 - 29) <= 1 || (unsigned int)(v15 - 1) <= 1) && (*(_BYTE *)(a1 + 175) & 0x20) != 0 )
  {
    if ( v16 <= (unsigned __int64)v13 )
    {
      sub_16E7DE0(a4, 121);
    }
    else
    {
      *(_QWORD *)(a4 + 24) = v13 + 1;
      *v13 = 121;
    }
    v13 = *(_BYTE **)(a4 + 24);
  }
  if ( *(_QWORD *)(a4 + 16) <= (unsigned __int64)v13 )
  {
    sub_16E7DE0(a4, 34);
  }
  else
  {
    *(_QWORD *)(a4 + 24) = v13 + 1;
    *v13 = 34;
  }
  v17 = *(_BYTE **)(a4 + 24);
  if ( (unsigned __int64)v17 >= *(_QWORD *)(a4 + 16) )
  {
    sub_16E7DE0(a4, 44);
  }
  else
  {
    *(_QWORD *)(a4 + 24) = v17 + 1;
    *v17 = 44;
  }
  v18 = *(_BYTE **)(a4 + 24);
  v19 = *(_QWORD *)(a4 + 16);
  if ( **(_BYTE **)(a2 + 48) != 64 )
  {
    if ( (unsigned __int64)v18 >= v19 )
    {
      sub_16E7DE0(a4, 64);
    }
    else
    {
      *(_QWORD *)(a4 + 24) = v18 + 1;
      *v18 = 64;
    }
LABEL_29:
    v20 = *(unsigned int *)(a1 + 168);
    if ( (_DWORD)v20 != 14 )
      goto LABEL_30;
LABEL_60:
    v30 = *(_QWORD *)(a4 + 24);
    if ( (unsigned __int64)(*(_QWORD *)(a4 + 16) - v30) <= 9 )
    {
      sub_16E7EE0(a4, "init_array", 0xAu);
      result = *(_BYTE **)(a4 + 24);
      goto LABEL_108;
    }
    v31 = 0x7272615F74696E69LL;
    goto LABEL_62;
  }
  if ( (unsigned __int64)v18 >= v19 )
  {
    sub_16E7DE0(a4, 37);
    goto LABEL_29;
  }
  *(_QWORD *)(a4 + 24) = v18 + 1;
  *v18 = 37;
  v20 = *(unsigned int *)(a1 + 168);
  if ( (_DWORD)v20 == 14 )
    goto LABEL_60;
LABEL_30:
  switch ( (_DWORD)v20 )
  {
    case 0xF:
      v30 = *(_QWORD *)(a4 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(a4 + 16) - v30) <= 9 )
      {
        sub_16E7EE0(a4, "fini_array", 0xAu);
        result = *(_BYTE **)(a4 + 24);
        break;
      }
      v31 = 0x7272615F696E6966LL;
LABEL_62:
      *(_QWORD *)v30 = v31;
      *(_WORD *)(v30 + 8) = 31073;
      result = (_BYTE *)(*(_QWORD *)(a4 + 24) + 10LL);
      *(_QWORD *)(a4 + 24) = result;
      break;
    case 0x10:
      v34 = *(void **)(a4 + 24);
      if ( *(_QWORD *)(a4 + 16) - (_QWORD)v34 <= 0xCu )
      {
        sub_16E7EE0(a4, "preinit_array", 0xDu);
        result = *(_BYTE **)(a4 + 24);
      }
      else
      {
        qmemcpy(v34, "preinit_array", 13);
        result = (_BYTE *)(*(_QWORD *)(a4 + 24) + 13LL);
        *(_QWORD *)(a4 + 24) = result;
      }
      break;
    case 8:
      v38 = *(_QWORD *)(a4 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(a4 + 16) - v38) <= 5 )
      {
        sub_16E7EE0(a4, "nobits", 6u);
        result = *(_BYTE **)(a4 + 24);
      }
      else
      {
        *(_DWORD *)v38 = 1768058734;
        *(_WORD *)(v38 + 4) = 29556;
        result = (_BYTE *)(*(_QWORD *)(a4 + 24) + 6LL);
        *(_QWORD *)(a4 + 24) = result;
      }
      break;
    case 7:
      v39 = *(_DWORD **)(a4 + 24);
      if ( *(_QWORD *)(a4 + 16) - (_QWORD)v39 <= 3u )
      {
        sub_16E7EE0(a4, "note", 4u);
        result = *(_BYTE **)(a4 + 24);
      }
      else
      {
        *v39 = 1702129518;
        result = (_BYTE *)(*(_QWORD *)(a4 + 24) + 4LL);
        *(_QWORD *)(a4 + 24) = result;
      }
      break;
    case 1:
      v50 = *(_QWORD **)(a4 + 24);
      if ( *(_QWORD *)(a4 + 16) - (_QWORD)v50 <= 7u )
      {
        sub_16E7EE0(a4, "progbits", 8u);
        result = *(_BYTE **)(a4 + 24);
      }
      else
      {
        *v50 = 0x73746962676F7270LL;
        result = (_BYTE *)(*(_QWORD *)(a4 + 24) + 8LL);
        *(_QWORD *)(a4 + 24) = result;
      }
      break;
    case 0x70000001:
      v51 = *(_QWORD *)(a4 + 24);
      if ( (unsigned __int64)(*(_QWORD *)(a4 + 16) - v51) <= 5 )
      {
        sub_16E7EE0(a4, "unwind", 6u);
        result = *(_BYTE **)(a4 + 24);
      }
      else
      {
        *(_DWORD *)v51 = 1769434741;
        *(_WORD *)(v51 + 4) = 25710;
        result = (_BYTE *)(*(_QWORD *)(a4 + 24) + 6LL);
        *(_QWORD *)(a4 + 24) = result;
      }
      break;
    case 0x7000001E:
      v52 = *(void **)(a4 + 24);
      if ( *(_QWORD *)(a4 + 16) - (_QWORD)v52 <= 9u )
      {
        sub_16E7EE0(a4, "0x7000001e", 0xAu);
        result = *(_BYTE **)(a4 + 24);
      }
      else
      {
        qmemcpy(v52, "0x7000001e", 10);
        result = (_BYTE *)(*(_QWORD *)(a4 + 24) + 10LL);
        *(_QWORD *)(a4 + 24) = result;
      }
      break;
    case 0x6FFF4C00:
      v53 = *(void **)(a4 + 24);
      if ( *(_QWORD *)(a4 + 16) - (_QWORD)v53 <= 0xAu )
      {
        sub_16E7EE0(a4, "llvm_odrtab", 0xBu);
        result = *(_BYTE **)(a4 + 24);
      }
      else
      {
        qmemcpy(v53, "llvm_odrtab", 11);
        result = (_BYTE *)(*(_QWORD *)(a4 + 24) + 11LL);
        *(_QWORD *)(a4 + 24) = result;
      }
      break;
    case 0x6FFF4C01:
      v54 = *(__m128i **)(a4 + 24);
      if ( *(_QWORD *)(a4 + 16) - (_QWORD)v54 <= 0x12u )
      {
        sub_16E7EE0(a4, "llvm_linker_options", 0x13u);
        result = *(_BYTE **)(a4 + 24);
      }
      else
      {
        si128 = _mm_load_si128((const __m128i *)&xmmword_3F816F0);
        v54[1].m128i_i8[2] = 115;
        v54[1].m128i_i16[0] = 28271;
        *v54 = si128;
        result = (_BYTE *)(*(_QWORD *)(a4 + 24) + 19LL);
        *(_QWORD *)(a4 + 24) = result;
      }
      break;
    case 0x6FFF4C02:
      v21 = *(__m128i **)(a4 + 24);
      if ( *(_QWORD *)(a4 + 16) - (_QWORD)v21 <= 0x16u )
      {
        sub_16E7EE0(a4, "llvm_call_graph_profile", 0x17u);
        result = *(_BYTE **)(a4 + 24);
      }
      else
      {
        v22 = _mm_load_si128((const __m128i *)&xmmword_3F81700);
        v21[1].m128i_i32[0] = 1718579824;
        v21[1].m128i_i16[2] = 27753;
        v21[1].m128i_i8[6] = 101;
        *v21 = v22;
        result = (_BYTE *)(*(_QWORD *)(a4 + 24) + 23LL);
        *(_QWORD *)(a4 + 24) = result;
      }
      break;
    default:
      v56 = *(_QWORD *)(a1 + 160);
      v57 = *(_QWORD *)(a1 + 152);
      v58 = v20;
      v71 = 261;
      v59[1] = v56;
      v59[0] = v57;
      v70.m128i_i64[0] = (__int64)v59;
      v60.m128i_i64[0] = (__int64)&v58;
      v66.m128i_i64[0] = (__int64)" for section ";
      v62.m128i_i64[0] = (__int64)"unsupported type 0x";
      v68 = 1;
      v67 = 3;
      v60.m128i_i64[1] = 0;
      v61 = 271;
      v64 = 1;
      v63 = 3;
      sub_14EC200(v65, &v62, &v60);
      sub_14EC200(v69, v65, &v66);
      sub_14EC200(v72, v69, &v70);
      sub_16BCFB0((__int64)v72, 1u);
  }
LABEL_108:
  if ( *(_DWORD *)(a1 + 180) )
  {
    if ( *(_BYTE **)(a4 + 16) == result )
    {
      v40 = sub_16E7EE0(a4, ",", 1u);
    }
    else
    {
      *result = 44;
      v40 = a4;
      ++*(_QWORD *)(a4 + 24);
    }
    sub_16E7A90(v40, *(unsigned int *)(a1 + 180));
    v35 = *(_DWORD *)(a1 + 172);
    result = *(_BYTE **)(a4 + 24);
    if ( (v35 & 0x200) == 0 )
    {
LABEL_110:
      if ( (v35 & 0x80) == 0 )
        goto LABEL_111;
      goto LABEL_152;
    }
  }
  else
  {
    v35 = *(_DWORD *)(a1 + 172);
    if ( (v35 & 0x200) == 0 )
      goto LABEL_110;
  }
  if ( *(_BYTE **)(a4 + 16) == result )
  {
    sub_16E7EE0(a4, ",", 1u);
  }
  else
  {
    *result = 44;
    ++*(_QWORD *)(a4 + 24);
  }
  v41 = *(_BYTE **)(a1 + 184);
  if ( (*v41 & 4) != 0 )
  {
    v42 = (size_t *)*((_QWORD *)v41 - 1);
    v43 = *v42;
    v44 = v42 + 2;
  }
  else
  {
    v43 = 0;
    v44 = 0;
  }
  sub_38D85A0(a4, v44, v43);
  v45 = *(_QWORD *)(a4 + 24);
  if ( (unsigned __int64)(*(_QWORD *)(a4 + 16) - v45) <= 6 )
  {
    sub_16E7EE0(a4, ",comdat", 7u);
    result = *(_BYTE **)(a4 + 24);
  }
  else
  {
    *(_DWORD *)v45 = 1836016428;
    *(_WORD *)(v45 + 4) = 24932;
    *(_BYTE *)(v45 + 6) = 116;
    result = (_BYTE *)(*(_QWORD *)(a4 + 24) + 7LL);
    *(_QWORD *)(a4 + 24) = result;
  }
  if ( (*(_DWORD *)(a1 + 172) & 0x80) != 0 )
  {
LABEL_152:
    if ( *(_BYTE **)(a4 + 16) == result )
    {
      sub_16E7EE0(a4, ",", 1u);
    }
    else
    {
      *result = 44;
      ++*(_QWORD *)(a4 + 24);
    }
    v46 = *(_BYTE **)(a1 + 192);
    if ( (*v46 & 4) != 0 )
    {
      v47 = (size_t *)*((_QWORD *)v46 - 1);
      v48 = *v47;
      v49 = v47 + 2;
    }
    else
    {
      v48 = 0;
      v49 = 0;
    }
    sub_38D85A0(a4, v49, v48);
    result = *(_BYTE **)(a4 + 24);
  }
LABEL_111:
  if ( *(_DWORD *)(a1 + 176) != -1 )
  {
    if ( *(_QWORD *)(a4 + 16) - (_QWORD)result <= 7u )
    {
      v36 = sub_16E7EE0(a4, ",unique,", 8u);
    }
    else
    {
      v36 = a4;
      *(_QWORD *)result = 0x2C657571696E752CLL;
      *(_QWORD *)(a4 + 24) += 8LL;
    }
    sub_16E7A90(v36, *(unsigned int *)(a1 + 176));
    result = *(_BYTE **)(a4 + 24);
  }
  if ( (unsigned __int64)result >= *(_QWORD *)(a4 + 16) )
  {
    result = (_BYTE *)sub_16E7DE0(a4, 10);
  }
  else
  {
    *(_QWORD *)(a4 + 24) = result + 1;
    *result = 10;
  }
  if ( a5 )
  {
    v37 = *(void **)(a4 + 24);
    if ( *(_QWORD *)(a4 + 16) - (_QWORD)v37 <= 0xCu )
    {
      sub_16E7EE0(a4, "\t.subsection\t", 0xDu);
    }
    else
    {
      qmemcpy(v37, "\t.subsection\t", 13);
      *(_QWORD *)(a4 + 24) += 13LL;
    }
LABEL_50:
    sub_38CDBE0(a5, a4, a2);
    goto LABEL_51;
  }
  return result;
}
