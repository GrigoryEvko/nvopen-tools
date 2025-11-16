// Function: sub_258BE70
// Address: 0x258be70
//
__int64 __fastcall sub_258BE70(__int64 a1, __int64 a2, const __m128i *a3, unsigned __int8 a4)
{
  __int64 *v4; // rax
  int v5; // r13d
  unsigned __int8 *v6; // r12
  unsigned int v7; // eax
  const __m128i *v8; // rdi
  __int64 v9; // rax
  const __m128i *v10; // r15
  const __m128i *v11; // rbx
  __m128i v12; // xmm0
  __int64 *v13; // r8
  int v14; // esi
  int v15; // r14d
  __int64 *v16; // r11
  unsigned int i; // eax
  __int64 *v18; // r9
  __int64 v19; // r10
  __m128i *v20; // rdi
  __m128i *v21; // rbx
  __int64 v22; // r14
  __int64 v23; // rdi
  char v24; // r9
  unsigned __int8 *v25; // r8
  unsigned __int64 v26; // rcx
  unsigned __int8 v27; // al
  unsigned int v28; // r14d
  unsigned int v30; // esi
  __int64 v31; // rax
  unsigned int v32; // eax
  unsigned int v33; // ecx
  unsigned int v34; // edi
  const __m128i *v35; // r14
  __int64 v36; // r8
  __int64 v37; // rax
  __m128i v38; // xmm1
  unsigned __int64 v39; // rdx
  __int64 v40; // rdx
  __m128i *v41; // rax
  unsigned int v42; // eax
  __int64 v43; // [rsp+8h] [rbp-2B8h]
  __int64 v44; // [rsp+10h] [rbp-2B0h]
  __int64 v45; // [rsp+10h] [rbp-2B0h]
  char v50; // [rsp+6Fh] [rbp-251h] BYREF
  __m128i v51; // [rsp+70h] [rbp-250h] BYREF
  int v52; // [rsp+80h] [rbp-240h]
  __m128i v53; // [rsp+90h] [rbp-230h] BYREF
  int v54; // [rsp+A0h] [rbp-220h]
  const __m128i *v55; // [rsp+B0h] [rbp-210h] BYREF
  __int64 v56; // [rsp+B8h] [rbp-208h]
  _BYTE v57[48]; // [rsp+C0h] [rbp-200h] BYREF
  __int64 v58; // [rsp+F0h] [rbp-1D0h] BYREF
  __int64 v59; // [rsp+F8h] [rbp-1C8h]
  __int64 *v60; // [rsp+100h] [rbp-1C0h] BYREF
  unsigned int v61; // [rsp+108h] [rbp-1B8h]
  __m128i *v62; // [rsp+1C0h] [rbp-100h] BYREF
  __int64 v63; // [rsp+1C8h] [rbp-F8h]
  _BYTE v64[240]; // [rsp+1D0h] [rbp-F0h] BYREF

  v4 = (__int64 *)&v60;
  v58 = 0;
  v59 = 1;
  do
  {
    *v4 = -4096;
    v4 += 3;
    *(v4 - 2) = -4096;
  }
  while ( v4 != (__int64 *)&v62 );
  v5 = 1;
  v62 = (__m128i *)v64;
  v63 = 0x800000000LL;
  v6 = (unsigned __int8 *)&unk_438A62F;
  if ( (a4 & 1) != 0 )
    goto LABEL_6;
  while ( &unk_438A631 != (_UNKNOWN *)++v6 )
  {
    v5 = *v6;
    if ( ((unsigned __int8)v5 & a4) != 0 )
    {
LABEL_6:
      v55 = (const __m128i *)v57;
      v50 = 0;
      v56 = 0x300000000LL;
      v7 = sub_2526B50(a2, a3, a1, (__int64)&v55, v5, &v50, 1u);
      if ( !(_BYTE)v7 )
      {
        v28 = v7;
        if ( v55 != (const __m128i *)v57 )
          _libc_free((unsigned __int64)v55);
        v20 = v62;
        goto LABEL_31;
      }
      v8 = v55;
      v9 = (unsigned int)v56;
      v10 = &v55[v9];
      if ( &v55[v9] != v55 )
      {
        v11 = v55;
        while ( 1 )
        {
          v12 = _mm_loadu_si128(v11);
          v52 = 0;
          v53 = v12;
          v51 = v12;
          if ( (v59 & 1) != 0 )
          {
            v13 = (__int64 *)&v60;
            v14 = 7;
          }
          else
          {
            v30 = v61;
            v13 = v60;
            if ( !v61 )
            {
              v32 = v59;
              ++v58;
              v53.m128i_i64[0] = 0;
              v33 = ((unsigned int)v59 >> 1) + 1;
LABEL_44:
              v34 = 3 * v30;
              goto LABEL_45;
            }
            v14 = v61 - 1;
          }
          v15 = 1;
          v16 = 0;
          for ( i = v14
                  & (((0xBF58476D1CE4E5B9LL
                     * (((unsigned __int32)v51.m128i_i32[2] >> 9) ^ ((unsigned __int32)v51.m128i_i32[2] >> 4)
                      | ((unsigned __int64)(((unsigned __int32)v51.m128i_i32[0] >> 9)
                                          ^ ((unsigned __int32)v51.m128i_i32[0] >> 4)) << 32))) >> 31)
                   ^ (484763065 * (((unsigned __int32)v51.m128i_i32[2] >> 9) ^ ((unsigned __int32)v51.m128i_i32[2] >> 4))));
                ;
                i = v14 & v42 )
          {
            v18 = &v13[3 * i];
            v19 = *v18;
            if ( *(_OWORD *)v18 == *(_OWORD *)&v51 )
            {
              v31 = *((unsigned int *)v18 + 4);
              goto LABEL_39;
            }
            if ( v19 == -4096 )
              break;
            if ( v19 == -8192 && v18[1] == -8192 && !v16 )
              v16 = &v13[3 * i];
LABEL_63:
            v42 = v15 + i;
            ++v15;
          }
          if ( v18[1] != -4096 )
            goto LABEL_63;
          v32 = v59;
          if ( !v16 )
            v16 = v18;
          ++v58;
          v53.m128i_i64[0] = (__int64)v16;
          v33 = ((unsigned int)v59 >> 1) + 1;
          if ( (v59 & 1) == 0 )
          {
            v30 = v61;
            goto LABEL_44;
          }
          v34 = 24;
          v30 = 8;
LABEL_45:
          if ( v34 <= 4 * v33 )
          {
            v35 = &v53;
            sub_2577460((__int64)&v58, 2 * v30);
          }
          else
          {
            v35 = &v53;
            if ( v30 - HIDWORD(v59) - v33 > v30 >> 3 )
              goto LABEL_47;
            sub_2577460((__int64)&v58, v30);
          }
          sub_2568570((__int64)&v58, v51.m128i_i64, (__int64 **)&v53);
          v32 = v59;
LABEL_47:
          v36 = v53.m128i_i64[0];
          LODWORD(v59) = (2 * (v32 >> 1) + 2) | v32 & 1;
          if ( *(_QWORD *)v53.m128i_i64[0] != -4096 || *(_QWORD *)(v53.m128i_i64[0] + 8) != -4096 )
            --HIDWORD(v59);
          *(__m128i *)v53.m128i_i64[0] = v51;
          *(_DWORD *)(v36 + 16) = v52;
          v37 = (unsigned int)v63;
          v38 = _mm_loadu_si128(v11);
          v54 = 0;
          v39 = (unsigned int)v63 + 1LL;
          v53 = v38;
          if ( v39 > HIDWORD(v63) )
          {
            if ( v62 > &v53 || (v43 = (__int64)v62, &v53 >= (__m128i *)((char *)v62 + 24 * (unsigned int)v63)) )
            {
              v45 = v36;
              sub_C8D5F0((__int64)&v62, v64, v39, 0x18u, v36, (__int64)v62);
              v40 = (__int64)v62;
              v37 = (unsigned int)v63;
              v36 = v45;
            }
            else
            {
              v44 = v36;
              sub_C8D5F0((__int64)&v62, v64, v39, 0x18u, v36, (__int64)v62);
              v40 = (__int64)v62;
              v37 = (unsigned int)v63;
              v36 = v44;
              v35 = (__m128i *)((char *)&v53 + (_QWORD)v62 - v43);
            }
          }
          else
          {
            v40 = (__int64)v62;
          }
          v41 = (__m128i *)(v40 + 24 * v37);
          *v41 = _mm_loadu_si128(v35);
          v41[1].m128i_i64[0] = v35[1].m128i_i64[0];
          v31 = (unsigned int)v63;
          LODWORD(v63) = v63 + 1;
          *(_DWORD *)(v36 + 16) = v31;
LABEL_39:
          ++v11;
          v62[1].m128i_i32[6 * v31] += v5;
          if ( v10 == v11 )
          {
            v8 = v55;
            break;
          }
        }
      }
      if ( v8 != (const __m128i *)v57 )
        _libc_free((unsigned __int64)v8);
    }
  }
  v20 = v62;
  v21 = (__m128i *)((char *)v62 + 24 * (unsigned int)v63);
  if ( v21 == v62 )
  {
    v28 = 1;
  }
  else
  {
    v22 = (__int64)v62;
    do
    {
      v23 = *(_QWORD *)(a1 + 72) & 0xFFFFFFFFFFFFFFFCLL;
      if ( (*(_QWORD *)(a1 + 72) & 3LL) == 3 )
        v23 = *(_QWORD *)(v23 + 24);
      v27 = *(_BYTE *)v23;
      if ( *(_BYTE *)v23 )
      {
        if ( v27 == 22 )
        {
          v23 = *(_QWORD *)(v23 + 24);
        }
        else if ( v27 <= 0x1Cu )
        {
          v23 = 0;
        }
        else
        {
          v23 = sub_B43CB0(v23);
        }
      }
      v24 = *(_BYTE *)(v22 + 16);
      v25 = *(unsigned __int8 **)(v22 + 8);
      v26 = *(_QWORD *)v22;
      v22 += 24;
      sub_258BA20(a1, a2, (_BYTE *)(a1 + 88), v26, v25, v24, v23);
    }
    while ( v21 != (__m128i *)v22 );
    v20 = v62;
    v28 = 1;
  }
LABEL_31:
  if ( v20 != (__m128i *)v64 )
    _libc_free((unsigned __int64)v20);
  if ( (v59 & 1) == 0 )
    sub_C7D6A0((__int64)v60, 24LL * v61, 8);
  return v28;
}
