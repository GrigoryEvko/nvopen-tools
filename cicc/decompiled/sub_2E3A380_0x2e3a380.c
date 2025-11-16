// Function: sub_2E3A380
// Address: 0x2e3a380
//
__int64 __fastcall sub_2E3A380(__int64 a1, __int64 a2)
{
  __m128i *v3; // rdx
  __m128i si128; // xmm0
  __int64 v6; // r14
  __int64 v8; // rax
  size_t v9; // rdx
  _BYTE *v10; // rdi
  unsigned __int8 *v11; // rsi
  _BYTE *v12; // rax
  size_t v13; // r15
  __int64 v14; // rax
  __int64 v15; // rdx
  __int64 v16; // rbx
  __int64 v17; // r15
  __int64 v18; // rax
  void *v19; // rdx
  int v20; // ecx
  __int64 v21; // rsi
  int v22; // ecx
  unsigned int v23; // edx
  __int64 *v24; // rax
  __int64 v25; // rdi
  int v26; // eax
  unsigned __int64 v27; // rax
  __int16 v28; // dx
  __int64 v29; // rax
  _QWORD *v30; // rdx
  __int64 v31; // r15
  int v32; // ecx
  __int64 v33; // rsi
  int v34; // ecx
  unsigned int v35; // edx
  __int64 *v36; // rax
  __int64 v37; // rdi
  int v38; // eax
  unsigned __int64 v39; // rax
  int v40; // ecx
  __int64 v41; // rsi
  int v42; // ecx
  unsigned int v43; // edx
  __int64 *v44; // rax
  __int64 v45; // rdi
  unsigned int v46; // eax
  __int64 *v47; // rax
  size_t v48; // rdx
  _BYTE *v49; // rax
  __m128i *v50; // rdx
  unsigned __int64 v51; // r15
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
  __int64 v62; // [rsp+18h] [rbp-68h]
  unsigned int v63; // [rsp+2Ch] [rbp-54h] BYREF
  unsigned __int8 *v64; // [rsp+30h] [rbp-50h] BYREF
  size_t v65; // [rsp+38h] [rbp-48h]
  __int64 v66; // [rsp+40h] [rbp-40h] BYREF

  if ( *(_QWORD *)(a1 + 128) )
  {
    v3 = *(__m128i **)(a2 + 32);
    if ( *(_QWORD *)(a2 + 24) - (_QWORD)v3 <= 0x15u )
    {
      v6 = sub_CB6200(a2, "block-frequency-info: ", 0x16u);
    }
    else
    {
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F8CAE0);
      v3[1].m128i_i32[0] = 1868983913;
      v6 = a2;
      v3[1].m128i_i16[2] = 8250;
      *v3 = si128;
      *(_QWORD *)(a2 + 32) += 22LL;
    }
    v8 = sub_2E791E0(*(_QWORD *)(a1 + 128));
    v10 = *(_BYTE **)(v6 + 32);
    v11 = (unsigned __int8 *)v8;
    v12 = *(_BYTE **)(v6 + 24);
    v13 = v9;
    if ( v9 > v12 - v10 )
    {
      v6 = sub_CB6200(v6, v11, v9);
      v12 = *(_BYTE **)(v6 + 24);
      v10 = *(_BYTE **)(v6 + 32);
    }
    else if ( v9 )
    {
      memcpy(v10, v11, v9);
      v12 = *(_BYTE **)(v6 + 24);
      v10 = (_BYTE *)(v13 + *(_QWORD *)(v6 + 32));
      *(_QWORD *)(v6 + 32) = v10;
    }
    if ( v12 == v10 )
    {
      sub_CB6200(v6, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *v10 = 10;
      ++*(_QWORD *)(v6 + 32);
    }
    v14 = *(_QWORD *)(a1 + 128);
    v15 = *(_QWORD *)(a2 + 32);
    v16 = *(_QWORD *)(v14 + 328);
    v62 = v14 + 320;
    if ( v16 != v14 + 320 )
    {
      while ( 1 )
      {
        if ( (unsigned __int64)(*(_QWORD *)(a2 + 24) - v15) > 2 )
        {
          *(_BYTE *)(v15 + 2) = 32;
          v17 = a2;
          *(_WORD *)v15 = 11552;
          *(_QWORD *)(a2 + 32) += 3LL;
        }
        else
        {
          v17 = sub_CB6200(a2, (unsigned __int8 *)" - ", 3u);
        }
        sub_2E3A130((__int64 *)&v64, v16);
        v18 = sub_CB6200(v17, v64, v65);
        v19 = *(void **)(v18 + 32);
        if ( *(_QWORD *)(v18 + 24) - (_QWORD)v19 <= 9u )
        {
          sub_CB6200(v18, ": float = ", 0xAu);
        }
        else
        {
          qmemcpy(v19, ": float = ", 10);
          *(_QWORD *)(v18 + 32) += 10LL;
        }
        if ( v64 != (unsigned __int8 *)&v66 )
          j_j___libc_free_0((unsigned __int64)v64);
        v20 = *(_DWORD *)(a1 + 184);
        v21 = *(_QWORD *)(a1 + 168);
        if ( v20 )
        {
          v22 = v20 - 1;
          v23 = v22 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          v24 = (__int64 *)(v21 + 16LL * v23);
          v25 = *v24;
          if ( v16 == *v24 )
          {
LABEL_21:
            v26 = *((_DWORD *)v24 + 2);
            goto LABEL_22;
          }
          v57 = 1;
          while ( v25 != -4096 )
          {
            v59 = v57 + 1;
            v23 = v22 & (v57 + v23);
            v24 = (__int64 *)(v21 + 16LL * v23);
            v25 = *v24;
            if ( *v24 == v16 )
              goto LABEL_21;
            v57 = v59;
          }
        }
        v26 = -1;
LABEL_22:
        LODWORD(v64) = v26;
        v27 = sub_FE8AC0(a1, (unsigned int *)&v64);
        v29 = sub_F04D90(a2, v27, v28, 64, 5u);
        v30 = *(_QWORD **)(v29 + 32);
        v31 = v29;
        if ( *(_QWORD *)(v29 + 24) - (_QWORD)v30 <= 7u )
        {
          v31 = sub_CB6200(v29, ", int = ", 8u);
        }
        else
        {
          *v30 = 0x203D20746E69202CLL;
          *(_QWORD *)(v29 + 32) += 8LL;
        }
        v32 = *(_DWORD *)(a1 + 184);
        v33 = *(_QWORD *)(a1 + 168);
        if ( v32 )
        {
          v34 = v32 - 1;
          v35 = v34 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
          v36 = (__int64 *)(v33 + 16LL * v35);
          v37 = *v36;
          if ( v16 == *v36 )
          {
LABEL_26:
            v38 = *((_DWORD *)v36 + 2);
            goto LABEL_27;
          }
          v58 = 1;
          while ( v37 != -4096 )
          {
            v61 = v58 + 1;
            v35 = v34 & (v58 + v35);
            v36 = (__int64 *)(v33 + 16LL * v35);
            v37 = *v36;
            if ( *v36 == v16 )
              goto LABEL_26;
            v58 = v61;
          }
        }
        v38 = -1;
LABEL_27:
        LODWORD(v64) = v38;
        v39 = sub_FE8720(a1, (unsigned int *)&v64);
        sub_CB59D0(v31, v39);
        v40 = *(_DWORD *)(a1 + 184);
        v41 = *(_QWORD *)(a1 + 168);
        if ( !v40 )
          goto LABEL_49;
        v42 = v40 - 1;
        v43 = v42 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
        v44 = (__int64 *)(v41 + 16LL * v43);
        v45 = *v44;
        if ( v16 != *v44 )
        {
          v56 = 1;
          while ( v45 != -4096 )
          {
            v60 = v56 + 1;
            v43 = v42 & (v56 + v43);
            v44 = (__int64 *)(v41 + 16LL * v43);
            v45 = *v44;
            if ( *v44 == v16 )
              goto LABEL_29;
            v56 = v60;
          }
LABEL_49:
          v46 = -1;
          goto LABEL_30;
        }
LABEL_29:
        v46 = *((_DWORD *)v44 + 2);
LABEL_30:
        v63 = v46;
        v47 = sub_FE8990(a1, **(_QWORD **)(a1 + 128), &v63, 0);
        v65 = v48;
        v64 = (unsigned __int8 *)v47;
        if ( (_BYTE)v48 )
        {
          v54 = *(void **)(a2 + 32);
          if ( *(_QWORD *)(a2 + 24) - (_QWORD)v54 <= 9u )
          {
            v55 = sub_CB6200(a2, ", count = ", 0xAu);
          }
          else
          {
            v55 = a2;
            qmemcpy(v54, ", count = ", 10);
            *(_QWORD *)(a2 + 32) += 10LL;
          }
          sub_CB59D0(v55, (unsigned __int64)v64);
        }
        if ( *(_BYTE *)(v16 + 176) )
        {
          v50 = *(__m128i **)(a2 + 32);
          v51 = *(_QWORD *)(v16 + 168);
          if ( *(_QWORD *)(a2 + 24) - (_QWORD)v50 <= 0x1Au )
          {
            v53 = sub_CB6200(a2, ", irr_loop_header_weight = ", 0x1Bu);
          }
          else
          {
            v52 = _mm_load_si128((const __m128i *)&xmmword_3F8CAF0);
            v53 = a2;
            qmemcpy(&v50[1], "r_weight = ", 11);
            *v50 = v52;
            *(_QWORD *)(a2 + 32) += 27LL;
          }
          sub_CB59D0(v53, v51);
        }
        v49 = *(_BYTE **)(a2 + 32);
        if ( *(_BYTE **)(a2 + 24) == v49 )
        {
          sub_CB6200(a2, (unsigned __int8 *)"\n", 1u);
          v15 = *(_QWORD *)(a2 + 32);
          v16 = *(_QWORD *)(v16 + 8);
          if ( v62 == v16 )
            break;
        }
        else
        {
          *v49 = 10;
          v15 = *(_QWORD *)(a2 + 32) + 1LL;
          *(_QWORD *)(a2 + 32) = v15;
          v16 = *(_QWORD *)(v16 + 8);
          if ( v62 == v16 )
            break;
        }
      }
    }
    if ( v15 == *(_QWORD *)(a2 + 24) )
    {
      sub_CB6200(a2, (unsigned __int8 *)"\n", 1u);
    }
    else
    {
      *(_BYTE *)v15 = 10;
      ++*(_QWORD *)(a2 + 32);
    }
  }
  return a2;
}
