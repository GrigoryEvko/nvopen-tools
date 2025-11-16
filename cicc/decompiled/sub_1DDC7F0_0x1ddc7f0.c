// Function: sub_1DDC7F0
// Address: 0x1ddc7f0
//
__int64 __fastcall sub_1DDC7F0(__int64 a1, __int64 a2)
{
  __m128i *v3; // rdx
  __m128i si128; // xmm0
  __int64 v6; // r14
  __int64 v8; // rax
  size_t v9; // rdx
  _BYTE *v10; // rdi
  char *v11; // rsi
  _BYTE *v12; // rax
  size_t v13; // r15
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rbx
  __int64 v17; // r15
  __int64 v18; // rax
  void *v19; // rdx
  int v20; // edx
  int v21; // eax
  int v22; // ecx
  __int64 v23; // rsi
  unsigned int v24; // edx
  __int64 *v25; // rax
  __int64 v26; // rdi
  unsigned __int64 v27; // rax
  __int16 v28; // dx
  __int64 v29; // rax
  _QWORD *v30; // rdx
  __int64 v31; // r15
  int v32; // edx
  int v33; // eax
  int v34; // edx
  __int64 v35; // rcx
  unsigned int v36; // esi
  __int64 *v37; // rax
  __int64 v38; // rdi
  __int64 v39; // rax
  int v40; // edx
  unsigned int v41; // eax
  int v42; // ecx
  __int64 v43; // rsi
  unsigned int v44; // edx
  __int64 *v45; // rax
  __int64 v46; // rdi
  __m128i *v47; // rdx
  __int64 v48; // r8
  __m128i v49; // xmm0
  __int64 v50; // rdi
  _BYTE *v51; // rax
  void *v52; // rdx
  __int64 v53; // rdi
  __int64 v54; // rax
  int v55; // eax
  int v56; // eax
  int v57; // eax
  int v58; // r8d
  int v59; // r8d
  int v60; // r8d
  __int64 v61; // [rsp+0h] [rbp-80h]
  __int64 i; // [rsp+18h] [rbp-68h]
  unsigned int v63; // [rsp+2Ch] [rbp-54h] BYREF
  char *v64; // [rsp+30h] [rbp-50h] BYREF
  size_t v65; // [rsp+38h] [rbp-48h]
  __int64 v66; // [rsp+40h] [rbp-40h] BYREF

  if ( *(_QWORD *)(a1 + 128) )
  {
    v3 = *(__m128i **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v3 <= 0x15u )
    {
      v6 = sub_16E7EE0(a2, "block-frequency-info: ", 0x16u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CAE0);
      v3[1].m128i_i32[0] = 1868983913;
      v6 = a2;
      v3[1].m128i_i16[2] = 8250;
      *v3 = si128;
      *(_QWORD *)(a2 + 24) += 22LL;
    }
    v8 = sub_1E0A440(*(_QWORD *)(a1 + 128));
    v10 = *(_BYTE **)(v6 + 24);
    v11 = (char *)v8;
    v12 = *(_BYTE **)(v6 + 16);
    v13 = v9;
    if ( v12 - v10 < v9 )
    {
      v6 = sub_16E7EE0(v6, v11, v9);
      v12 = *(_BYTE **)(v6 + 16);
      v10 = *(_BYTE **)(v6 + 24);
    }
    else if ( v9 )
    {
      memcpy(v10, v11, v9);
      v12 = *(_BYTE **)(v6 + 16);
      v10 = (_BYTE *)(v13 + *(_QWORD *)(v6 + 24));
      *(_QWORD *)(v6 + 24) = v10;
    }
    if ( v10 == v12 )
    {
      sub_16E7EE0(v6, "\n", 1u);
    }
    else
    {
      *v10 = 10;
      ++*(_QWORD *)(v6 + 24);
    }
    v14 = *(_QWORD *)(a1 + 128);
    v15 = *(_QWORD *)(a2 + 24);
    v16 = *(_QWORD *)(v14 + 328);
    for ( i = v14 + 320; i != v16; v16 = *(_QWORD *)(v16 + 8) )
    {
      while ( 1 )
      {
        if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v15) > 2 )
        {
          *(_BYTE *)(v15 + 2) = 32;
          v17 = a2;
          *(_WORD *)v15 = 11552;
          *(_QWORD *)(a2 + 24) += 3LL;
        }
        else
        {
          v17 = sub_16E7EE0(a2, " - ", 3u);
        }
        sub_1DDC610((__int64 *)&v64, v16);
        v18 = sub_16E7EE0(v17, v64, v65);
        v19 = *(void **)(v18 + 24);
        if ( *(_QWORD *)(v18 + 16) - (_QWORD)v19 <= 9u )
        {
          sub_16E7EE0(v18, ": float = ", 0xAu);
        }
        else
        {
          qmemcpy(v19, ": float = ", 10);
          *(_QWORD *)(v18 + 24) += 10LL;
        }
        if ( v64 != (char *)&v66 )
          j_j___libc_free_0(v64, v66 + 1);
        v20 = *(_DWORD *)(a1 + 184);
        v21 = -1;
        if ( v20 )
        {
          v22 = v20 - 1;
          v23 = *(_QWORD *)(a1 + 168);
          v24 = (v20 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          v25 = (__int64 *)(v23 + 16LL * v24);
          v26 = *v25;
          if ( *v25 == v16 )
          {
LABEL_21:
            v21 = *((_DWORD *)v25 + 2);
          }
          else
          {
            v57 = 1;
            while ( v26 != -8 )
            {
              v60 = v57 + 1;
              v24 = v22 & (v57 + v24);
              v25 = (__int64 *)(v23 + 16LL * v24);
              v26 = *v25;
              if ( *v25 == v16 )
                goto LABEL_21;
              v57 = v60;
            }
            v21 = -1;
          }
        }
        LODWORD(v64) = v21;
        v27 = sub_1370F90(a1, (unsigned int *)&v64);
        v29 = sub_16CC0B0(a2, v27, v28, 64, 5u);
        v30 = *(_QWORD **)(v29 + 24);
        v31 = v29;
        if ( *(_QWORD *)(v29 + 16) - (_QWORD)v30 <= 7u )
        {
          v31 = sub_16E7EE0(v29, ", int = ", 8u);
        }
        else
        {
          *v30 = 0x203D20746E69202CLL;
          *(_QWORD *)(v29 + 24) += 8LL;
        }
        v32 = *(_DWORD *)(a1 + 184);
        v33 = -1;
        if ( v32 )
        {
          v34 = v32 - 1;
          v35 = *(_QWORD *)(a1 + 168);
          v36 = v34 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          v37 = (__int64 *)(v35 + 16LL * v36);
          v38 = *v37;
          if ( *v37 == v16 )
          {
LABEL_26:
            v33 = *((_DWORD *)v37 + 2);
          }
          else
          {
            v56 = 1;
            while ( v38 != -8 )
            {
              v59 = v56 + 1;
              v36 = v34 & (v56 + v36);
              v37 = (__int64 *)(v35 + 16LL * v36);
              v38 = *v37;
              if ( *v37 == v16 )
                goto LABEL_26;
              v56 = v59;
            }
            v33 = -1;
          }
        }
        LODWORD(v64) = v33;
        v39 = sub_1370CD0(a1, (unsigned int *)&v64);
        sub_16E7A90(v31, v39);
        v40 = *(_DWORD *)(a1 + 184);
        v41 = -1;
        if ( v40 )
        {
          v42 = v40 - 1;
          v43 = *(_QWORD *)(a1 + 168);
          v44 = (v40 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          v45 = (__int64 *)(v43 + 16LL * v44);
          v46 = *v45;
          if ( *v45 == v16 )
          {
LABEL_29:
            v41 = *((_DWORD *)v45 + 2);
          }
          else
          {
            v55 = 1;
            while ( v46 != -8 )
            {
              v58 = v55 + 1;
              v44 = v42 & (v55 + v44);
              v45 = (__int64 *)(v43 + 16LL * v44);
              v46 = *v45;
              if ( *v45 == v16 )
                goto LABEL_29;
              v55 = v58;
            }
            v41 = -1;
          }
        }
        v63 = v41;
        sub_1370E50((__int64)&v64, a1, **(_QWORD **)(a1 + 128), &v63);
        if ( (_BYTE)v65 )
        {
          v52 = *(void **)(a2 + 24);
          if ( *(_QWORD *)(a2 + 16) - (_QWORD)v52 <= 9u )
          {
            v53 = sub_16E7EE0(a2, ", count = ", 0xAu);
          }
          else
          {
            v53 = a2;
            qmemcpy(v52, ", count = ", 10);
            *(_QWORD *)(a2 + 24) += 10LL;
          }
          sub_16E7A90(v53, (__int64)v64);
        }
        if ( *(_BYTE *)(v16 + 144) )
        {
          v47 = *(__m128i **)(a2 + 24);
          v48 = *(_QWORD *)(v16 + 136);
          if ( *(_QWORD *)(a2 + 16) - (_QWORD)v47 <= 0x1Au )
          {
            v61 = *(_QWORD *)(v16 + 136);
            v54 = sub_16E7EE0(a2, ", irr_loop_header_weight = ", 0x1Bu);
            v48 = v61;
            v50 = v54;
          }
          else
          {
            v49 = _mm_load_si128((const __m128i *)&xmmword_3F8CAF0);
            v50 = a2;
            qmemcpy(&v47[1], "r_weight = ", 11);
            *v47 = v49;
            *(_QWORD *)(a2 + 24) += 27LL;
          }
          sub_16E7A90(v50, v48);
        }
        v51 = *(_BYTE **)(a2 + 24);
        if ( *(_BYTE **)(a2 + 16) == v51 )
          break;
        *v51 = 10;
        v15 = *(_QWORD *)(a2 + 24) + 1LL;
        *(_QWORD *)(a2 + 24) = v15;
        v16 = *(_QWORD *)(v16 + 8);
        if ( i == v16 )
          goto LABEL_40;
      }
      sub_16E7EE0(a2, "\n", 1u);
      v15 = *(_QWORD *)(a2 + 24);
    }
LABEL_40:
    if ( *(_QWORD *)(a2 + 16) == v15 )
    {
      sub_16E7EE0(a2, "\n", 1u);
    }
    else
    {
      *(_BYTE *)v15 = 10;
      ++*(_QWORD *)(a2 + 24);
    }
  }
  return a2;
}
