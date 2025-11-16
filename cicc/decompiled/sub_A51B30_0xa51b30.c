// Function: sub_A51B30
// Address: 0xa51b30
//
_BYTE *__fastcall sub_A51B30(__int64 *a1, __int64 a2)
{
  __int64 v4; // rdi
  void *v5; // rdx
  __int64 v6; // r13
  __m128i *v7; // rdx
  __m128i si128; // xmm0
  const char *v9; // r14
  size_t v10; // r8
  _QWORD *v11; // rax
  __m128i *v12; // rax
  __m128i v13; // xmm0
  __int64 v14; // rdi
  _BYTE *v15; // rax
  __int64 v16; // rdi
  _BYTE *result; // rax
  __int64 v18; // r15
  _WORD *v19; // rdx
  __int64 v20; // rdi
  __int64 v21; // r8
  __int64 v22; // rdx
  __int64 v23; // rax
  _WORD *v24; // rdx
  int v25; // eax
  __int64 v26; // rdi
  const char *v27; // rsi
  __int64 v28; // rdi
  _BYTE *v29; // rax
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // r13
  __int64 v33; // rdi
  __m128i *v34; // rdx
  unsigned int v35; // eax
  __int64 v36; // r14
  bool v37; // cc
  const char *v38; // r12
  size_t v39; // rdx
  _QWORD *v40; // rax
  __int64 v41; // rdi
  _BYTE *v42; // rax
  __int64 v43; // rdi
  _WORD *v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // rax
  __int64 v48; // rax
  unsigned __int64 v49; // r8
  char *v50; // rax
  const char *v51; // rsi
  unsigned int v52; // eax
  unsigned int v53; // eax
  unsigned int v54; // ecx
  __int64 v55; // rdi
  __int64 v56; // rax
  __int64 v57; // rax
  __int64 v58; // rax
  __int64 v59; // rax
  unsigned __int64 v60; // rsi
  char *v61; // rax
  const char *v62; // r14
  unsigned int v63; // ecx
  unsigned int v64; // eax
  __int64 v65; // rdx
  __int64 v66; // [rsp+8h] [rbp-48h]
  char v67; // [rsp+17h] [rbp-39h]

  v4 = *a1;
  v5 = *(void **)(v4 + 32);
  if ( *(_QWORD *)(v4 + 24) - (_QWORD)v5 <= 0xBu )
  {
    sub_CB6200(v4, ", summary: (", 12);
  }
  else
  {
    qmemcpy(v5, ", summary: (", 12);
    *(_QWORD *)(v4 + 32) += 12LL;
  }
  v6 = *a1;
  v7 = *(__m128i **)(*a1 + 32);
  if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v7 <= 0x13u )
  {
    v6 = sub_CB6200(*a1, "typeTestRes: (kind: ", 20);
  }
  else
  {
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F24AA0);
    v7[1].m128i_i32[0] = 540697710;
    *v7 = si128;
    *(_QWORD *)(v6 + 32) += 20LL;
  }
  switch ( *(_DWORD *)a2 )
  {
    case 0:
      v9 = "unsat";
      goto LABEL_9;
    case 1:
      v9 = "byteArray";
      goto LABEL_9;
    case 2:
      v9 = "inline";
      goto LABEL_9;
    case 3:
      v9 = "single";
      goto LABEL_9;
    case 4:
      v9 = "allOnes";
      goto LABEL_9;
    case 5:
      v9 = "unknown";
LABEL_9:
      v10 = strlen(v9);
      v11 = *(_QWORD **)(v6 + 32);
      if ( v10 > *(_QWORD *)(v6 + 24) - (_QWORD)v11 )
      {
        v6 = sub_CB6200(v6, v9, v10);
        v12 = *(__m128i **)(v6 + 32);
        goto LABEL_11;
      }
      if ( (unsigned int)v10 >= 8 )
      {
        v60 = (unsigned __int64)(v11 + 1) & 0xFFFFFFFFFFFFFFF8LL;
        *v11 = *(_QWORD *)v9;
        *(_QWORD *)((char *)v11 + (unsigned int)v10 - 8) = *(_QWORD *)&v9[(unsigned int)v10 - 8];
        v61 = (char *)v11 - v60;
        v62 = (const char *)(v9 - v61);
        if ( (((_DWORD)v10 + (_DWORD)v61) & 0xFFFFFFF8) >= 8 )
        {
          v63 = (v10 + (_DWORD)v61) & 0xFFFFFFF8;
          v64 = 0;
          do
          {
            v65 = v64;
            v64 += 8;
            *(_QWORD *)(v60 + v65) = *(_QWORD *)&v62[v65];
          }
          while ( v64 < v63 );
        }
      }
      else
      {
        if ( (v10 & 4) != 0 )
        {
          *(_DWORD *)v11 = *(_DWORD *)v9;
          *(_DWORD *)((char *)v11 + (unsigned int)v10 - 4) = *(_DWORD *)&v9[(unsigned int)v10 - 4];
          v11 = *(_QWORD **)(v6 + 32);
          goto LABEL_99;
        }
        if ( !(_DWORD)v10 )
        {
LABEL_99:
          v12 = (__m128i *)((char *)v11 + v10);
          *(_QWORD *)(v6 + 32) = v12;
          goto LABEL_11;
        }
        *(_BYTE *)v11 = *v9;
        if ( (v10 & 2) != 0 )
        {
          *(_WORD *)((char *)v11 + (unsigned int)v10 - 2) = *(_WORD *)&v9[(unsigned int)v10 - 2];
          v11 = *(_QWORD **)(v6 + 32);
          goto LABEL_99;
        }
      }
      v12 = (__m128i *)(v10 + *(_QWORD *)(v6 + 32));
      *(_QWORD *)(v6 + 32) = v12;
LABEL_11:
      if ( *(_QWORD *)(v6 + 24) - (_QWORD)v12 <= 0x11u )
      {
        v6 = sub_CB6200(v6, ", sizeM1BitWidth: ", 18);
      }
      else
      {
        v13 = _mm_load_si128((const __m128i *)&xmmword_3F24AB0);
        v12[1].m128i_i16[0] = 8250;
        *v12 = v13;
        *(_QWORD *)(v6 + 32) += 18LL;
      }
      sub_CB59D0(v6, *(unsigned int *)(a2 + 4));
      if ( *(_QWORD *)(a2 + 8) )
      {
        v59 = sub_904010(*a1, ", alignLog2: ");
        sub_CB59D0(v59, *(_QWORD *)(a2 + 8));
      }
      if ( *(_QWORD *)(a2 + 16) )
      {
        v58 = sub_904010(*a1, ", sizeM1: ");
        sub_CB59D0(v58, *(_QWORD *)(a2 + 16));
      }
      if ( *(_BYTE *)(a2 + 24) )
      {
        v57 = sub_904010(*a1, ", bitMask: ");
        sub_CB59D0(v57, *(unsigned __int8 *)(a2 + 24));
      }
      if ( *(_QWORD *)(a2 + 32) )
      {
        v56 = sub_904010(*a1, ", inlineBits: ");
        sub_CB59D0(v56, *(_QWORD *)(a2 + 32));
      }
      v14 = *a1;
      v15 = *(_BYTE **)(*a1 + 32);
      if ( *(_BYTE **)(*a1 + 24) == v15 )
      {
        sub_CB6200(v14, ")", 1);
      }
      else
      {
        *v15 = 41;
        ++*(_QWORD *)(v14 + 32);
      }
      if ( !*(_QWORD *)(a2 + 80) )
        goto LABEL_24;
      sub_904010(*a1, ", wpdResolutions: (");
      v18 = *(_QWORD *)(a2 + 64);
      v67 = 1;
      v66 = a2 + 48;
      if ( a2 + 48 == v18 )
        goto LABEL_91;
      break;
    default:
      goto LABEL_107;
  }
  do
  {
    v20 = *a1;
    v21 = *a1;
    if ( v67 )
    {
      v67 = 0;
    }
    else
    {
      v19 = *(_WORD **)(v20 + 32);
      if ( *(_QWORD *)(v20 + 24) - (_QWORD)v19 <= 1u )
      {
        sub_CB6200(v20, ", ", 2);
      }
      else
      {
        *v19 = 8236;
        *(_QWORD *)(v20 + 32) += 2LL;
      }
      v20 = *a1;
      v21 = *a1;
    }
    v22 = *(_QWORD *)(v20 + 32);
    if ( (unsigned __int64)(*(_QWORD *)(v20 + 24) - v22) <= 8 )
    {
      v21 = sub_CB6200(v20, "(offset: ", 9);
    }
    else
    {
      *(_BYTE *)(v22 + 8) = 32;
      *(_QWORD *)v22 = 0x3A74657366666F28LL;
      *(_QWORD *)(v20 + 32) += 9LL;
    }
    v23 = sub_CB59D0(v21, *(_QWORD *)(v18 + 32));
    v24 = *(_WORD **)(v23 + 32);
    if ( *(_QWORD *)(v23 + 24) - (_QWORD)v24 <= 1u )
    {
      sub_CB6200(v23, ", ", 2);
    }
    else
    {
      *v24 = 8236;
      *(_QWORD *)(v23 + 32) += 2LL;
    }
    sub_904010(*a1, "wpdRes: (kind: ");
    v25 = *(_DWORD *)(v18 + 40);
    v26 = *a1;
    if ( v25 == 1 )
    {
      sub_904010(v26, "singleImpl");
      if ( *(_DWORD *)(v18 + 40) == 1 )
        goto LABEL_52;
    }
    else
    {
      v27 = "branchFunnel";
      if ( v25 != 2 )
      {
        if ( v25 )
LABEL_107:
          BUG();
        v27 = "indir";
      }
      sub_904010(v26, v27);
      if ( *(_DWORD *)(v18 + 40) == 1 )
      {
LABEL_52:
        v30 = sub_904010(*a1, ", singleImplName: \"");
        v31 = sub_CB6200(v30, *(_QWORD *)(v18 + 48), *(_QWORD *)(v18 + 56));
        sub_904010(v31, "\"");
        if ( !*(_QWORD *)(v18 + 120) )
          goto LABEL_46;
        goto LABEL_53;
      }
    }
    if ( !*(_QWORD *)(v18 + 120) )
      goto LABEL_46;
LABEL_53:
    sub_904010(*a1, ", resByArg: (");
    v32 = *(_QWORD *)(v18 + 104);
    if ( v32 != v18 + 88 )
    {
      while ( 1 )
      {
        sub_A50F50(a1, (__int64 **)(v32 + 32));
        v33 = *a1;
        v34 = *(__m128i **)(*a1 + 32);
        if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v34 <= 0xFu )
        {
          sub_CB6200(v33, ", byArg: (kind: ", 16);
          v35 = *(_DWORD *)(v32 + 56);
          v36 = *a1;
          v37 = v35 <= 2;
          if ( v35 == 2 )
          {
LABEL_76:
            v38 = "uniqueRetVal";
            goto LABEL_59;
          }
        }
        else
        {
          *v34 = _mm_load_si128((const __m128i *)&xmmword_3F24AC0);
          *(_QWORD *)(v33 + 32) += 16LL;
          v35 = *(_DWORD *)(v32 + 56);
          v36 = *a1;
          v37 = v35 <= 2;
          if ( v35 == 2 )
            goto LABEL_76;
        }
        if ( v37 )
        {
          v38 = "uniformRetVal";
          if ( !v35 )
            v38 = "indir";
        }
        else
        {
          if ( v35 != 3 )
            goto LABEL_107;
          v38 = "virtualConstProp";
        }
LABEL_59:
        v39 = strlen(v38);
        v40 = *(_QWORD **)(v36 + 32);
        if ( *(_QWORD *)(v36 + 24) - (_QWORD)v40 >= v39 )
        {
          if ( (unsigned int)v39 >= 8 )
          {
            v49 = (unsigned __int64)(v40 + 1) & 0xFFFFFFFFFFFFFFF8LL;
            *v40 = *(_QWORD *)v38;
            *(_QWORD *)((char *)v40 + (unsigned int)v39 - 8) = *(_QWORD *)&v38[(unsigned int)v39 - 8];
            v50 = (char *)v40 - v49;
            v51 = (const char *)(v38 - v50);
            v52 = (v39 + (_DWORD)v50) & 0xFFFFFFF8;
            if ( v52 >= 8 )
            {
              v53 = v52 & 0xFFFFFFF8;
              v54 = 0;
              do
              {
                v55 = v54;
                v54 += 8;
                *(_QWORD *)(v49 + v55) = *(_QWORD *)&v51[v55];
              }
              while ( v54 < v53 );
            }
          }
          else if ( (v39 & 4) != 0 )
          {
            *(_DWORD *)v40 = *(_DWORD *)v38;
            *(_DWORD *)((char *)v40 + (unsigned int)v39 - 4) = *(_DWORD *)&v38[(unsigned int)v39 - 4];
          }
          else if ( (_DWORD)v39 )
          {
            *(_BYTE *)v40 = *v38;
            if ( (v39 & 2) != 0 )
              *(_WORD *)((char *)v40 + (unsigned int)v39 - 2) = *(_WORD *)&v38[(unsigned int)v39 - 2];
          }
          *(_QWORD *)(v36 + 32) += v39;
        }
        else
        {
          sub_CB6200(v36, v38, v39);
        }
        if ( (unsigned int)(*(_DWORD *)(v32 + 56) - 1) <= 1 )
        {
          v48 = sub_904010(*a1, ", info: ");
          sub_CB59D0(v48, *(_QWORD *)(v32 + 64));
        }
        if ( *(_QWORD *)(v32 + 72) )
        {
          v45 = sub_904010(*a1, ", byte: ");
          v46 = sub_CB59D0(v45, *(unsigned int *)(v32 + 72));
          v47 = sub_904010(v46, ", bit: ");
          sub_CB59D0(v47, *(unsigned int *)(v32 + 76));
        }
        v41 = *a1;
        v42 = *(_BYTE **)(*a1 + 32);
        if ( *(_BYTE **)(*a1 + 24) == v42 )
        {
          sub_CB6200(v41, ")", 1);
        }
        else
        {
          *v42 = 41;
          ++*(_QWORD *)(v41 + 32);
        }
        v32 = sub_220EF30(v32);
        if ( v18 + 88 == v32 )
          break;
        v43 = *a1;
        v44 = *(_WORD **)(*a1 + 32);
        if ( *(_QWORD *)(*a1 + 24) - (_QWORD)v44 <= 1u )
        {
          sub_CB6200(v43, ", ", 2);
        }
        else
        {
          *v44 = 8236;
          *(_QWORD *)(v43 + 32) += 2LL;
        }
      }
    }
    sub_904010(*a1, ")");
LABEL_46:
    sub_904010(*a1, ")");
    v28 = *a1;
    v29 = *(_BYTE **)(*a1 + 32);
    if ( *(_BYTE **)(*a1 + 24) == v29 )
    {
      sub_CB6200(v28, ")", 1);
    }
    else
    {
      *v29 = 41;
      ++*(_QWORD *)(v28 + 32);
    }
    v18 = sub_220EF30(v18);
  }
  while ( v66 != v18 );
LABEL_91:
  sub_904010(*a1, ")");
LABEL_24:
  v16 = *a1;
  result = *(_BYTE **)(*a1 + 32);
  if ( *(_BYTE **)(*a1 + 24) == result )
    return (_BYTE *)sub_CB6200(v16, ")", 1);
  *result = 41;
  ++*(_QWORD *)(v16 + 32);
  return result;
}
