// Function: sub_13683A0
// Address: 0x13683a0
//
__int64 __fastcall sub_13683A0(__int64 a1, __int64 a2)
{
  __m128i *v3; // rdx
  __m128i si128; // xmm0
  __int64 v6; // r12
  __int64 v8; // rax
  size_t v9; // rdx
  _BYTE *v10; // rdi
  const char *v11; // rsi
  _BYTE *v12; // rax
  size_t v13; // r14
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rax
  void *v17; // rdx
  int v18; // edx
  int v19; // eax
  int v20; // ecx
  __int64 v21; // rsi
  unsigned int v22; // edx
  __int64 *v23; // rax
  __int64 v24; // rdi
  __int64 v25; // rax
  __int16 v26; // dx
  __int64 v27; // rax
  _QWORD *v28; // rdx
  __int64 v29; // r15
  int v30; // edx
  int v31; // eax
  int v32; // edx
  __int64 v33; // rcx
  unsigned int v34; // esi
  __int64 *v35; // rax
  __int64 v36; // rdi
  __int64 v37; // rax
  int v38; // edx
  int v39; // eax
  int v40; // ecx
  __int64 v41; // rsi
  unsigned int v42; // edx
  __int64 *v43; // rax
  __int64 v44; // rdi
  __int64 v45; // rdx
  _BYTE *v46; // rax
  __int64 v47; // r12
  __int64 v48; // r15
  __int64 v49; // rdx
  _BYTE *v50; // rsi
  __m128i *v51; // rdx
  __m128i v52; // xmm0
  __int64 v53; // rdi
  void *v54; // rdx
  __int64 v55; // rdi
  int v56; // eax
  int v57; // eax
  int v58; // eax
  int v59; // r8d
  int v60; // r8d
  int v61; // r8d
  __int64 v62; // [rsp+18h] [rbp-78h]
  __int64 v63; // [rsp+28h] [rbp-68h]
  int v64; // [rsp+3Ch] [rbp-54h] BYREF
  const char *v65; // [rsp+40h] [rbp-50h] BYREF
  __int64 v66; // [rsp+48h] [rbp-48h]
  char v67[64]; // [rsp+50h] [rbp-40h] BYREF

  if ( *(_QWORD *)(a1 + 128) )
  {
    v3 = *(__m128i **)(a2 + 24);
    if ( *(_QWORD *)(a2 + 16) - (_QWORD)v3 <= 0x15u )
    {
      v6 = sub_16E7EE0(a2, "block-frequency-info: ", 22);
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
    v8 = sub_1649960(*(_QWORD *)(a1 + 128));
    v10 = *(_BYTE **)(v6 + 24);
    v11 = (const char *)v8;
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
    if ( v12 == v10 )
    {
      sub_16E7EE0(v6, "\n", 1);
    }
    else
    {
      *v10 = 10;
      ++*(_QWORD *)(v6 + 24);
    }
    v14 = *(_QWORD *)(a1 + 128);
    v15 = *(_QWORD *)(a2 + 24);
    v62 = v14 + 72;
    v63 = *(_QWORD *)(v14 + 80);
    if ( v63 != v14 + 72 )
    {
      do
      {
        v47 = v63 - 24;
        if ( !v63 )
          v47 = 0;
        if ( (unsigned __int64)(*(_QWORD *)(a2 + 16) - v15) <= 2 )
        {
          v48 = sub_16E7EE0(a2, " - ", 3);
        }
        else
        {
          *(_BYTE *)(v15 + 2) = 32;
          v48 = a2;
          *(_WORD *)v15 = 11552;
          *(_QWORD *)(a2 + 24) += 3LL;
        }
        v50 = (_BYTE *)sub_1649960(v47);
        v65 = v67;
        if ( v50 )
        {
          sub_1367D20((__int64 *)&v65, v50, (__int64)&v50[v49]);
          v16 = sub_16E7EE0(v48, v65, v66);
        }
        else
        {
          v67[0] = 0;
          v66 = 0;
          v16 = sub_16E7EE0(v48, v67, 0);
        }
        v17 = *(void **)(v16 + 24);
        if ( *(_QWORD *)(v16 + 16) - (_QWORD)v17 <= 9u )
        {
          sub_16E7EE0(v16, ": float = ", 10);
        }
        else
        {
          qmemcpy(v17, ": float = ", 10);
          *(_QWORD *)(v16 + 24) += 10LL;
        }
        if ( v65 != v67 )
          j_j___libc_free_0(v65, *(_QWORD *)v67 + 1LL);
        v18 = *(_DWORD *)(a1 + 184);
        v19 = -1;
        if ( v18 )
        {
          v20 = v18 - 1;
          v21 = *(_QWORD *)(a1 + 168);
          v22 = (v18 - 1) & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
          v23 = (__int64 *)(v21 + 16LL * v22);
          v24 = *v23;
          if ( v47 == *v23 )
          {
LABEL_21:
            v19 = *((_DWORD *)v23 + 2);
          }
          else
          {
            v56 = 1;
            while ( v24 != -8 )
            {
              v59 = v56 + 1;
              v22 = v20 & (v56 + v22);
              v23 = (__int64 *)(v21 + 16LL * v22);
              v24 = *v23;
              if ( v47 == *v23 )
                goto LABEL_21;
              v56 = v59;
            }
            v19 = -1;
          }
        }
        LODWORD(v65) = v19;
        v25 = sub_1370F90(a1, &v65);
        v27 = sub_16CC0B0(a2, v25, (unsigned int)v26, 64, 5);
        v28 = *(_QWORD **)(v27 + 24);
        v29 = v27;
        if ( *(_QWORD *)(v27 + 16) - (_QWORD)v28 <= 7u )
        {
          v29 = sub_16E7EE0(v27, ", int = ", 8);
        }
        else
        {
          *v28 = 0x203D20746E69202CLL;
          *(_QWORD *)(v27 + 24) += 8LL;
        }
        v30 = *(_DWORD *)(a1 + 184);
        v31 = -1;
        if ( v30 )
        {
          v32 = v30 - 1;
          v33 = *(_QWORD *)(a1 + 168);
          v34 = v32 & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
          v35 = (__int64 *)(v33 + 16LL * v34);
          v36 = *v35;
          if ( v47 == *v35 )
          {
LABEL_26:
            v31 = *((_DWORD *)v35 + 2);
          }
          else
          {
            v57 = 1;
            while ( v36 != -8 )
            {
              v60 = v57 + 1;
              v34 = v32 & (v57 + v34);
              v35 = (__int64 *)(v33 + 16LL * v34);
              v36 = *v35;
              if ( v47 == *v35 )
                goto LABEL_26;
              v57 = v60;
            }
            v31 = -1;
          }
        }
        LODWORD(v65) = v31;
        v37 = sub_1370CD0(a1, &v65);
        sub_16E7A90(v29, v37);
        v38 = *(_DWORD *)(a1 + 184);
        v39 = -1;
        if ( v38 )
        {
          v40 = v38 - 1;
          v41 = *(_QWORD *)(a1 + 168);
          v42 = (v38 - 1) & (((unsigned int)v47 >> 9) ^ ((unsigned int)v47 >> 4));
          v43 = (__int64 *)(v41 + 16LL * v42);
          v44 = *v43;
          if ( v47 == *v43 )
          {
LABEL_29:
            v39 = *((_DWORD *)v43 + 2);
          }
          else
          {
            v58 = 1;
            while ( v44 != -8 )
            {
              v61 = v58 + 1;
              v42 = v40 & (v58 + v42);
              v43 = (__int64 *)(v41 + 16LL * v42);
              v44 = *v43;
              if ( v47 == *v43 )
                goto LABEL_29;
              v58 = v61;
            }
            v39 = -1;
          }
        }
        v45 = *(_QWORD *)(a1 + 128);
        v64 = v39;
        sub_1370E50(&v65, a1, v45, &v64);
        if ( (_BYTE)v66 )
        {
          v54 = *(void **)(a2 + 24);
          if ( *(_QWORD *)(a2 + 16) - (_QWORD)v54 <= 9u )
          {
            v55 = sub_16E7EE0(a2, ", count = ", 10);
          }
          else
          {
            v55 = a2;
            qmemcpy(v54, ", count = ", 10);
            *(_QWORD *)(a2 + 24) += 10LL;
          }
          sub_16E7A90(v55, v65);
        }
        sub_157F7D0(&v65, v47);
        if ( (_BYTE)v66 )
        {
          v51 = *(__m128i **)(a2 + 24);
          if ( *(_QWORD *)(a2 + 16) - (_QWORD)v51 <= 0x1Au )
          {
            v53 = sub_16E7EE0(a2, ", irr_loop_header_weight = ", 27);
          }
          else
          {
            v52 = _mm_load_si128((const __m128i *)&xmmword_3F8CAF0);
            v53 = a2;
            qmemcpy(&v51[1], "r_weight = ", 11);
            *v51 = v52;
            *(_QWORD *)(a2 + 24) += 27LL;
          }
          sub_16E7A90(v53, v65);
        }
        v46 = *(_BYTE **)(a2 + 24);
        if ( *(_BYTE **)(a2 + 16) == v46 )
        {
          sub_16E7EE0(a2, "\n", 1);
          v15 = *(_QWORD *)(a2 + 24);
        }
        else
        {
          *v46 = 10;
          v15 = *(_QWORD *)(a2 + 24) + 1LL;
          *(_QWORD *)(a2 + 24) = v15;
        }
        v63 = *(_QWORD *)(v63 + 8);
      }
      while ( v62 != v63 );
    }
    if ( *(_QWORD *)(a2 + 16) == v15 )
    {
      sub_16E7EE0(a2, "\n", 1);
    }
    else
    {
      *(_BYTE *)v15 = 10;
      ++*(_QWORD *)(a2 + 24);
    }
  }
  return a2;
}
