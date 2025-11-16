// Function: sub_349FA50
// Address: 0x349fa50
//
__int64 __fastcall sub_349FA50(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rax
  __int64 v5; // r9
  __int64 v6; // r12
  __int64 v7; // r14
  __int64 v8; // r13
  __int64 v9; // rsi
  __int64 v10; // rax
  int v11; // edx
  __m128i v12; // xmm0
  __int64 v13; // rax
  bool v14; // zf
  __int64 v15; // rcx
  const __m128i *v16; // rbx
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  unsigned __int64 v19; // r8
  __m128i *v20; // rax
  const __m128i *v21; // r8
  __int64 v22; // r9
  __int64 v23; // rdx
  __int64 v24; // r12
  __int64 v26; // rax
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rax
  int v30; // r11d
  unsigned __int64 v31; // rdx
  __int64 v32; // rdx
  unsigned __int64 v33; // rcx
  unsigned __int64 v34; // r8
  const __m128i *v35; // rax
  __m128i *v36; // rdx
  __int32 v37; // eax
  __int64 v38; // rax
  _BYTE *v39; // rax
  __int8 *v40; // rbx
  char *v41; // rbx
  _WORD *v42; // [rsp+8h] [rbp-218h]
  __int64 v43; // [rsp+18h] [rbp-208h]
  char v44; // [rsp+34h] [rbp-1ECh]
  _QWORD *v46; // [rsp+40h] [rbp-1E0h]
  __m128i v47; // [rsp+50h] [rbp-1D0h] BYREF
  __int64 v48; // [rsp+60h] [rbp-1C0h]
  __int64 *v49; // [rsp+70h] [rbp-1B0h] BYREF
  __int64 v50; // [rsp+78h] [rbp-1A8h]
  __int64 v51; // [rsp+80h] [rbp-1A0h] BYREF
  __int64 v52; // [rsp+88h] [rbp-198h]
  __int64 v53; // [rsp+90h] [rbp-190h]
  unsigned __int64 v54; // [rsp+A0h] [rbp-180h] BYREF
  __int64 v55; // [rsp+A8h] [rbp-178h]
  _BYTE v56[368]; // [rsp+B0h] [rbp-170h] BYREF

  v3 = *(_QWORD *)(a1 + 48);
  v44 = 0;
  if ( *(_WORD *)(v3 + 68) == 14 )
  {
    v39 = *(_BYTE **)(v3 + 32);
    if ( v39[40] == 1 )
      v44 = *v39 == 0;
  }
  v42 = *(_WORD **)(v3 + 16);
  v4 = sub_2E89170(v3);
  v6 = *(unsigned int *)(a1 + 72);
  v43 = v4;
  v46 = *(_QWORD **)(a1 + 40);
  v54 = (unsigned __int64)v56;
  v55 = 0x800000000LL;
  if ( (_DWORD)v6 )
  {
    v7 = a1;
    v8 = 0;
    while ( 2 )
    {
      v9 = *(_QWORD *)(v7 + 48);
      v10 = *(_QWORD *)(v7 + 64) + 32 * v8;
      v11 = *(_DWORD *)v10;
      v12 = _mm_loadu_si128((const __m128i *)(v10 + 8));
      v13 = *(_QWORD *)(v10 + 24);
      v14 = *(_WORD *)(v9 + 68) == 14;
      v47 = v12;
      v48 = v13;
      v15 = *(_QWORD *)(v9 + 32);
      if ( !v14 )
        v15 += 80;
      v16 = (const __m128i *)(v15 + 40LL * *(unsigned int *)(*(_QWORD *)(v7 + 336) + 4 * v8));
      switch ( v11 )
      {
        case 0:
          BUG();
        case 1:
          v37 = v47.m128i_i32[0];
          if ( *(_DWORD *)(v7 + 56) == 1 )
            v37 = v16->m128i_i32[2];
          v32 = (unsigned int)v55;
          v49 = 0;
          LODWORD(v50) = v37;
          v33 = v54;
          v34 = (unsigned int)v55 + 1LL;
          v51 = 0;
          v35 = (const __m128i *)&v49;
          v52 = 0;
          v53 = 0;
          if ( v34 <= HIDWORD(v55) )
            goto LABEL_20;
          if ( v54 > (unsigned __int64)&v49 || (unsigned __int64)&v49 >= v54 + 40LL * (unsigned int)v55 )
            goto LABEL_43;
          goto LABEL_36;
        case 2:
          v26 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a2 + 16) + 200LL))(*(_QWORD *)(a2 + 16));
          if ( *(_WORD *)(*(_QWORD *)(v7 + 48) + 68LL) == 14 )
          {
            v38 = sub_2FF7570(v26, v46, 2 * (v44 & 1u), &v47.m128i_i64[1]);
            v44 = 1;
            v46 = (_QWORD *)v38;
          }
          else
          {
            v49 = &v51;
            v50 = 0x400000000LL;
            (*(void (__fastcall **)(__int64, __int8 *, __int64 **))(*(_QWORD *)v26 + 592LL))(
              v26,
              &v47.m128i_i8[8],
              &v49);
            v29 = (unsigned int)v50;
            v30 = v8;
            v31 = (unsigned int)v50 + 1LL;
            if ( v31 > HIDWORD(v50) )
            {
              sub_C8D5F0((__int64)&v49, &v51, v31, 8u, v27, v28);
              v29 = (unsigned int)v50;
              v30 = v8;
            }
            v49[v29] = 6;
            LODWORD(v50) = v50 + 1;
            v46 = (_QWORD *)sub_B0DBA0(v46, v49, (unsigned int)v50, v30, 0);
            if ( v49 != &v51 )
              _libc_free((unsigned __int64)v49);
          }
          v32 = (unsigned int)v55;
          v49 = 0;
          LODWORD(v50) = v47.m128i_i32[0];
          v33 = v54;
          v34 = (unsigned int)v55 + 1LL;
          v51 = 0;
          v35 = (const __m128i *)&v49;
          v52 = 0;
          v53 = 0;
          if ( v34 <= HIDWORD(v55) )
            goto LABEL_20;
          if ( v54 > (unsigned __int64)&v49 || (unsigned __int64)&v49 >= v54 + 40LL * (unsigned int)v55 )
          {
LABEL_43:
            sub_C8D5F0((__int64)&v54, v56, (unsigned int)v55 + 1LL, 0x28u, v34, v5);
            v33 = v54;
            v32 = (unsigned int)v55;
            v35 = (const __m128i *)&v49;
          }
          else
          {
LABEL_36:
            v41 = (char *)&v49 - v54;
            sub_C8D5F0((__int64)&v54, v56, (unsigned int)v55 + 1LL, 0x28u, v34, v5);
            v33 = v54;
            v32 = (unsigned int)v55;
            v35 = (const __m128i *)&v41[v54];
          }
LABEL_20:
          v36 = (__m128i *)(v33 + 40 * v32);
          *v36 = _mm_loadu_si128(v35);
          v36[1] = _mm_loadu_si128(v35 + 1);
          v36[2].m128i_i64[0] = v35[2].m128i_i64[0];
          LODWORD(v55) = v55 + 1;
          goto LABEL_9;
        case 3:
          v17 = (unsigned int)v55;
          v18 = v54;
          v19 = (unsigned int)v55 + 1LL;
          if ( v19 <= HIDWORD(v55) )
            goto LABEL_8;
          if ( v54 <= (unsigned __int64)v16 && (unsigned __int64)v16 < v54 + 40LL * (unsigned int)v55 )
            goto LABEL_32;
          goto LABEL_40;
        case 4:
          v17 = (unsigned int)v55;
          v18 = v54;
          v19 = (unsigned int)v55 + 1LL;
          if ( v19 <= HIDWORD(v55) )
            goto LABEL_8;
          if ( v54 <= (unsigned __int64)v16 && (unsigned __int64)v16 < v54 + 40LL * (unsigned int)v55 )
          {
LABEL_32:
            v40 = &v16->m128i_i8[-v54];
            sub_C8D5F0((__int64)&v54, v56, (unsigned int)v55 + 1LL, 0x28u, v19, v5);
            v18 = v54;
            v17 = (unsigned int)v55;
            v16 = (const __m128i *)&v40[v54];
          }
          else
          {
LABEL_40:
            sub_C8D5F0((__int64)&v54, v56, (unsigned int)v55 + 1LL, 0x28u, v19, v5);
            v18 = v54;
            v17 = (unsigned int)v55;
          }
LABEL_8:
          v20 = (__m128i *)(v18 + 40 * v17);
          *v20 = _mm_loadu_si128(v16);
          v20[1] = _mm_loadu_si128(v16 + 1);
          v20[2].m128i_i64[0] = v16[2].m128i_i64[0];
          LODWORD(v55) = v55 + 1;
LABEL_9:
          if ( v6 != ++v8 )
            continue;
          v21 = (const __m128i *)v54;
          v22 = (unsigned int)v55;
          break;
        default:
          goto LABEL_9;
      }
      break;
    }
  }
  else
  {
    v21 = (const __m128i *)v56;
    v22 = 0;
  }
  sub_2E908B0((_QWORD *)a2, (unsigned __int8 **)(v3 + 56), v42, v44, v21, v22, v43, (__int64)v46);
  v24 = v23;
  if ( (_BYTE *)v54 != v56 )
    _libc_free(v54);
  return v24;
}
