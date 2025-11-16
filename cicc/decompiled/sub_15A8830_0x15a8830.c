// Function: sub_15A8830
// Address: 0x15a8830
//
__int64 __fastcall sub_15A8830(_QWORD *a1, __int8 *a2, size_t a3)
{
  __m128i *p_src; // rdx
  size_t v6; // rax
  _BYTE *v7; // rdi
  __int64 result; // rax
  __int64 v9; // rsi
  size_t v10; // rdi
  __int64 v11; // rcx
  size_t v12; // rdx
  size_t *v13; // r12
  __int64 v14; // r15
  __m128i v15; // xmm2
  __int128 v16; // rdi
  __m128i v17; // kr10_16
  char v18; // dl
  unsigned int v19; // ebx
  unsigned int v20; // r8d
  __m128i v21; // xmm6
  void *v22; // r14
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  unsigned int v27; // r13d
  unsigned int v28; // r13d
  unsigned int v29; // r8d
  unsigned int v30; // eax
  unsigned int v31; // ecx
  unsigned int v32; // eax
  __m128i v33; // xmm4
  void *v34; // rbx
  void *v35; // r13
  unsigned int v36; // eax
  int v37; // r14d
  __m128i v38; // xmm5
  void *v39; // r13
  void *v40; // rbx
  unsigned int v41; // r11d
  unsigned int v42; // r11d
  __m128i v43; // xmm7
  void *v44; // rbx
  unsigned int v45; // r13d
  unsigned int v46; // r13d
  int v47; // r9d
  unsigned int v48; // eax
  size_t *v49; // rax
  void *v50; // r12
  void *v51; // r14
  __int64 v52; // r13
  __m128i v53; // xmm3
  int v54; // ebx
  unsigned int v55; // eax
  __m128i *v56; // rdi
  __m128i i; // kr30_16
  int v58; // eax
  char v59; // r13
  void *v60; // [rsp+8h] [rbp-A8h]
  unsigned int v61; // [rsp+14h] [rbp-9Ch]
  unsigned int v62; // [rsp+14h] [rbp-9Ch]
  void *v63; // [rsp+18h] [rbp-98h]
  unsigned int v64; // [rsp+18h] [rbp-98h]
  _QWORD *v65; // [rsp+20h] [rbp-90h]
  __m128i v66; // [rsp+30h] [rbp-80h]
  __m128i v67; // [rsp+40h] [rbp-70h] BYREF
  __m128i v68; // [rsp+50h] [rbp-60h]
  size_t n[2]; // [rsp+60h] [rbp-50h] BYREF
  __m128i src; // [rsp+70h] [rbp-40h] BYREF

  v66.m128i_i64[0] = (__int64)a2;
  v66.m128i_i64[1] = a3;
  if ( !a2 )
  {
    src.m128i_i8[0] = 0;
    v7 = (_BYTE *)a1[24];
    v12 = 0;
    n[0] = (size_t)&src;
LABEL_10:
    a1[25] = v12;
    v7[v12] = 0;
    result = n[0];
    goto LABEL_11;
  }
  v67.m128i_i64[0] = a3;
  n[0] = (size_t)&src;
  if ( a3 > 0xF )
  {
    n[0] = sub_22409D0(n, &v67, 0);
    v56 = (__m128i *)n[0];
    src.m128i_i64[0] = v67.m128i_i64[0];
  }
  else
  {
    if ( a3 == 1 )
    {
      p_src = &src;
      src.m128i_i8[0] = *a2;
      v6 = 1;
      goto LABEL_5;
    }
    if ( !a3 )
    {
      v6 = 0;
      p_src = &src;
      goto LABEL_5;
    }
    v56 = &src;
  }
  memcpy(v56, a2, a3);
  v6 = v67.m128i_i64[0];
  p_src = (__m128i *)n[0];
LABEL_5:
  n[1] = v6;
  p_src->m128i_i8[v6] = 0;
  v7 = (_BYTE *)a1[24];
  result = (__int64)v7;
  if ( (__m128i *)n[0] == &src )
  {
    v12 = n[1];
    if ( n[1] )
    {
      if ( n[1] == 1 )
        *v7 = src.m128i_i8[0];
      else
        memcpy(v7, &src, n[1]);
      v12 = n[1];
      v7 = (_BYTE *)a1[24];
    }
    goto LABEL_10;
  }
  v9 = src.m128i_i64[0];
  v10 = n[1];
  if ( (_QWORD *)result == a1 + 26 )
  {
    a1[24] = n[0];
    a1[25] = v10;
    a1[26] = v9;
  }
  else
  {
    v11 = a1[26];
    a1[24] = n[0];
    a1[25] = v10;
    a1[26] = v9;
    if ( result )
    {
      n[0] = result;
      src.m128i_i64[0] = v11;
      goto LABEL_11;
    }
  }
  n[0] = (size_t)&src;
  result = (__int64)&src;
LABEL_11:
  n[1] = 0;
  *(_BYTE *)result = 0;
  if ( (__m128i *)n[0] != &src )
    result = j_j___libc_free_0(n[0], src.m128i_i64[0] + 1);
  if ( a3 )
  {
    v13 = n;
    v14 = (__int64)a1;
    v65 = a1 + 5;
    do
    {
      sub_15A7EB0((__int64)&v67, (void *)v66.m128i_i64[0], (void *)v66.m128i_i64[1], 45);
      v66 = v68;
      sub_15A7EB0((__int64)v13, (void *)v67.m128i_i64[0], (void *)v67.m128i_i64[1], 58);
      v15 = _mm_loadu_si128(&src);
      v16 = *(_OWORD *)n;
      v17 = src;
      v67 = _mm_loadu_si128((const __m128i *)n);
      v68 = v15;
      if ( n[1] == 2 )
      {
        if ( *(_WORD *)n[0] == 26990 )
        {
          v49 = v13;
          v51 = (void *)src.m128i_i64[1];
          v50 = (void *)src.m128i_i64[0];
          v52 = (__int64)v49;
          do
          {
            v68.m128i_i64[0] = (__int64)v50;
            v68.m128i_i64[1] = (__int64)v51;
            sub_15A7EB0(v52, v50, v51, 58);
            v53 = _mm_loadu_si128(&src);
            v50 = (void *)src.m128i_i64[0];
            v67 = *(__m128i *)n;
            v51 = (void *)src.m128i_i64[1];
            v68 = v53;
            v54 = sub_15A7FD0(n[0], n[1]);
            if ( !v54 )
              sub_16BD130("Address space 0 can never be non-integral", 1);
            result = *(unsigned int *)(v14 + 416);
            if ( (unsigned int)result >= *(_DWORD *)(v14 + 420) )
            {
              sub_16CD150(v14 + 408, v14 + 424, 0, 4);
              result = *(unsigned int *)(v14 + 416);
            }
            *(_DWORD *)(*(_QWORD *)(v14 + 408) + 4 * result) = v54;
            ++*(_DWORD *)(v14 + 416);
          }
          while ( v51 );
          v13 = (size_t *)v52;
          continue;
        }
        v18 = *(_BYTE *)n[0];
      }
      else
      {
        v18 = *(_BYTE *)n[0];
        if ( !n[1] )
          goto LABEL_18;
      }
      *((_QWORD *)&v16 + 1) = n[1] - 1;
      *(_QWORD *)&v16 = n[0] + 1;
LABEL_18:
      result = (unsigned __int8)(v18 - 65);
      switch ( v18 )
      {
        case 'A':
          result = sub_15A7FD0(v16, *((__int64 *)&v16 + 1));
          if ( (unsigned int)result > 0xFFFFFF )
            goto LABEL_112;
          *(_DWORD *)(v14 + 4) = result;
          break;
        case 'E':
          *(_BYTE *)v14 = 1;
          break;
        case 'P':
          result = sub_15A7FD0(v16, *((__int64 *)&v16 + 1));
          if ( (unsigned int)result > 0xFFFFFF )
LABEL_112:
            sub_16BD130("Invalid address space, must be a 24-bit integer", 1);
          *(_DWORD *)(v14 + 12) = result;
          break;
        case 'S':
          v67 = (__m128i)v16;
          v32 = sub_15A7FD0(v16, *((__int64 *)&v16 + 1));
          if ( (v32 & 7) != 0 )
            goto LABEL_99;
          result = v32 >> 3;
          *(_DWORD *)(v14 + 8) = result;
          break;
        case 'a':
        case 'f':
        case 'i':
        case 'v':
          v19 = 105;
          if ( v18 != 105 )
          {
            v19 = 118;
            if ( v18 <= 105 )
              v19 = 5 * (v18 != 97) + 97;
          }
          v20 = 0;
          if ( *((_QWORD *)&v16 + 1) )
          {
            v67 = (__m128i)v16;
            v55 = sub_15A7FD0(v16, *((__int64 *)&v16 + 1));
            v20 = v55;
            if ( v19 == 97 )
            {
              if ( v55 )
                sub_16BD130("Sized aggregate specification in datalayout string", 1);
            }
          }
          if ( !v17.m128i_i64[1] )
            sub_16BD130("Missing alignment specification in datalayout string", 1);
          v61 = v20;
          v68 = v17;
          sub_15A7EB0((__int64)v13, (void *)v17.m128i_i64[0], (void *)v17.m128i_i64[1], 58);
          v16 = *(_OWORD *)n;
          v21 = _mm_loadu_si128(&src);
          v67 = *(__m128i *)n;
          v22 = (void *)src.m128i_i64[1];
          v63 = (void *)src.m128i_i64[0];
          v68 = v21;
          v27 = sub_15A7FD0(n[0], n[1]);
          if ( (v27 & 7) != 0 )
            goto LABEL_99;
          v28 = v27 >> 3;
          v29 = v61;
          if ( v19 != 97 && !v28 )
            sub_16BD130("ABI alignment specification must be >0 for non-aggregate types", 1);
          if ( v22 )
          {
            v68.m128i_i64[0] = (__int64)v63;
            v68.m128i_i64[1] = (__int64)v22;
            sub_15A7EB0((__int64)v13, v63, v22, 58);
            v16 = *(_OWORD *)n;
            v67 = *(__m128i *)n;
            v30 = sub_15A7FD0(n[0], n[1]);
            v24 = v30;
            if ( (v30 & 7) != 0 )
              goto LABEL_99;
            v29 = v61;
            v31 = v30 >> 3;
          }
          else
          {
            v31 = v28;
          }
          result = sub_15A82D0(v14, v19, v28, v31, v29);
          break;
        case 'e':
          *(_BYTE *)v14 = 0;
          break;
        case 'm':
          if ( *((_QWORD *)&v16 + 1) )
            sub_16BD130("Unexpected trailing characters after mangling specifier in datalayout string", 1);
          if ( !src.m128i_i64[1] )
            sub_16BD130("Expected mangling specifier in datalayout string", 1);
          if ( src.m128i_i64[1] != 1 )
            sub_16BD130("Unknown mangling specifier in datalayout string", 1);
          result = (unsigned __int8)(*(_BYTE *)src.m128i_i64[0] - 101);
          switch ( *(_BYTE *)src.m128i_i64[0] )
          {
            case 'e':
              *(_DWORD *)(v14 + 16) = 1;
              break;
            case 'm':
              *(_DWORD *)(v14 + 16) = 5;
              break;
            case 'o':
              *(_DWORD *)(v14 + 16) = 2;
              break;
            case 'w':
              *(_DWORD *)(v14 + 16) = 3;
              break;
            case 'x':
              *(_DWORD *)(v14 + 16) = 4;
              break;
            default:
              sub_16BD130("Unknown mangling in datalayout string", 1);
          }
          break;
        case 'n':
          for ( i = src; ; i = src )
          {
            v67 = (__m128i)v16;
            v58 = sub_15A7FD0(v16, *((__int64 *)&v16 + 1));
            v59 = v58;
            if ( !v58 )
              sub_16BD130("Zero width native integer type in datalayout string", 1);
            result = *(unsigned int *)(v14 + 32);
            if ( (unsigned int)result >= *(_DWORD *)(v14 + 36) )
            {
              sub_16CD150(v14 + 24, v65, 0, 1);
              result = *(unsigned int *)(v14 + 32);
            }
            *(_BYTE *)(*(_QWORD *)(v14 + 24) + result) = v59;
            ++*(_DWORD *)(v14 + 32);
            if ( !i.m128i_i64[1] )
              break;
            v68 = i;
            sub_15A7EB0((__int64)v13, (void *)i.m128i_i64[0], (void *)i.m128i_i64[1], 58);
            v16 = *(_OWORD *)n;
            v68 = _mm_loadu_si128(&src);
          }
          break;
        case 'p':
          v64 = 0;
          if ( *((_QWORD *)&v16 + 1) )
          {
            v67 = (__m128i)v16;
            v64 = sub_15A7FD0(v16, *((__int64 *)&v16 + 1));
            if ( v64 > 0xFFFFFF )
              sub_16BD130("Invalid address space, must be a 24bit integer", 1);
          }
          if ( !v17.m128i_i64[1] )
            sub_16BD130("Missing size specification for pointer in datalayout string", 1);
          v68 = v17;
          sub_15A7EB0((__int64)v13, (void *)v17.m128i_i64[0], (void *)v17.m128i_i64[1], 58);
          v16 = *(_OWORD *)n;
          v33 = _mm_loadu_si128(&src);
          v34 = (void *)src.m128i_i64[0];
          v67 = *(__m128i *)n;
          v35 = (void *)src.m128i_i64[1];
          v68 = v33;
          v36 = sub_15A7FD0(n[0], n[1]);
          if ( (v36 & 7) != 0 )
            goto LABEL_99;
          v37 = v36 >> 3;
          if ( !(v36 >> 3) )
            sub_16BD130("Invalid pointer size of 0 bytes", 1);
          if ( !v35 )
            sub_16BD130("Missing alignment specification for pointer in datalayout string", 1);
          v68.m128i_i64[0] = (__int64)v34;
          v68.m128i_i64[1] = (__int64)v35;
          sub_15A7EB0((__int64)v13, v34, v35, 58);
          v16 = *(_OWORD *)n;
          v38 = _mm_loadu_si128(&src);
          v39 = (void *)src.m128i_i64[0];
          v67 = *(__m128i *)n;
          v40 = (void *)src.m128i_i64[1];
          v68 = v38;
          v41 = sub_15A7FD0(n[0], n[1]);
          if ( (v41 & 7) != 0 )
            goto LABEL_99;
          v42 = v41 >> 3;
          if ( !v42 || (v42 & (v42 - 1)) != 0 )
            sub_16BD130("Pointer ABI alignment must be a power of 2", 1);
          if ( v40 )
          {
            v62 = v42;
            v68.m128i_i64[0] = (__int64)v39;
            v68.m128i_i64[1] = (__int64)v40;
            sub_15A7EB0((__int64)v13, v39, v40, 58);
            v16 = *(_OWORD *)n;
            v43 = _mm_loadu_si128(&src);
            v67 = *(__m128i *)n;
            v44 = (void *)src.m128i_i64[1];
            v60 = (void *)src.m128i_i64[0];
            v68 = v43;
            v45 = sub_15A7FD0(n[0], n[1]);
            if ( (v45 & 7) != 0 )
              goto LABEL_99;
            v46 = v45 >> 3;
            if ( !v46 || (v42 = v62, (v46 & (v46 - 1)) != 0) )
              sub_16BD130("Pointer preferred alignment must be a power of 2", 1);
            v47 = v37;
            if ( v44 )
            {
              v68.m128i_i64[1] = (__int64)v44;
              v68.m128i_i64[0] = (__int64)v60;
              sub_15A7EB0((__int64)v13, v60, v44, 58);
              v16 = *(_OWORD *)n;
              v67 = *(__m128i *)n;
              v48 = sub_15A7FD0(n[0], n[1]);
              v26 = v48;
              if ( (v48 & 7) != 0 )
LABEL_99:
                sub_15A7E90(v16, *((_QWORD *)&v16 + 1), v23, v24, v25, v26);
              v47 = v48 >> 3;
              v42 = v62;
              if ( !(v48 >> 3) )
                sub_16BD130("Invalid index size of 0 bytes", 1);
            }
          }
          else
          {
            v46 = v42;
            v47 = v37;
          }
          result = sub_15A85E0(v14, v64, v42, v46, v37, v47);
          break;
        case 's':
          break;
        default:
          sub_16BD130("Unknown specifier in datalayout string", 1);
      }
    }
    while ( v66.m128i_i64[1] );
  }
  return result;
}
