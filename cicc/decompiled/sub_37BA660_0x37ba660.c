// Function: sub_37BA660
// Address: 0x37ba660
//
_QWORD *__fastcall sub_37BA660(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rsi
  __int64 v8; // r8
  unsigned __int64 v9; // r9
  __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rcx
  _QWORD *v13; // r14
  unsigned __int64 v14; // rbx
  __int64 v15; // rdi
  unsigned int v16; // eax
  unsigned int v17; // eax
  int v18; // r10d
  __int64 v19; // r11
  int *v20; // r13
  unsigned int v21; // eax
  char v22; // cl
  _DWORD *v23; // rax
  __int64 *v24; // rdi
  _DWORD *v25; // rsi
  bool v26; // cf
  __int64 v27; // rax
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // r8
  __int64 v31; // r9
  __int64 v32; // rdx
  char v33; // r8
  __int64 v34; // rax
  __int64 v35; // rdx
  const __m128i *v36; // rax
  __m128i *v37; // rdx
  __m128i v38; // xmm3
  const __m128i *v39; // r13
  unsigned int v40; // eax
  unsigned int v41; // edx
  __int64 v42; // rax
  unsigned __int64 v43; // r8
  const __m128i *v44; // rdx
  __m128i *v45; // rax
  _QWORD *v46; // r12
  __int64 v47; // rax
  unsigned __int64 v48; // rdx
  unsigned __int64 v49; // r8
  _BYTE *v50; // rax
  __int64 v51; // rdx
  __int64 v53; // rax
  unsigned __int64 v54; // rdx
  __int64 v55; // rax
  __int64 v56; // r8
  signed __int64 v57; // r13
  int v58; // r13d
  int v59; // r11d
  signed __int64 v60; // r13
  int v61; // esi
  int v62; // r13d
  __int64 v63; // r11
  int v64; // eax
  _WORD *v65; // [rsp+8h] [rbp-178h]
  char v66; // [rsp+23h] [rbp-15Dh]
  char v67; // [rsp+24h] [rbp-15Ch]
  unsigned __int16 v68; // [rsp+24h] [rbp-15Ch]
  int v69; // [rsp+28h] [rbp-158h]
  const __m128i *v70; // [rsp+28h] [rbp-158h]
  int v71; // [rsp+28h] [rbp-158h]
  unsigned int v72; // [rsp+30h] [rbp-150h]
  int v73; // [rsp+30h] [rbp-150h]
  char v76; // [rsp+57h] [rbp-129h] BYREF
  unsigned __int8 *v77; // [rsp+58h] [rbp-128h] BYREF
  __int64 v78; // [rsp+60h] [rbp-120h] BYREF
  int v79; // [rsp+68h] [rbp-118h]
  __int64 v80; // [rsp+70h] [rbp-110h]
  __int64 v81; // [rsp+78h] [rbp-108h]
  __int64 v82; // [rsp+80h] [rbp-100h]
  const __m128i *v83; // [rsp+90h] [rbp-F0h] BYREF
  __int64 v84; // [rsp+98h] [rbp-E8h]
  _BYTE v85[48]; // [rsp+A0h] [rbp-E0h] BYREF
  __int64 v86[8]; // [rsp+D0h] [rbp-B0h] BYREF
  unsigned __int64 v87; // [rsp+110h] [rbp-70h] BYREF
  __int64 v88; // [rsp+118h] [rbp-68h]
  _QWORD v89[12]; // [rsp+120h] [rbp-60h] BYREF

  v7 = a4;
  sub_B10CB0(&v77, a4);
  v10 = *(unsigned __int8 *)(a5 + 9);
  v11 = *(_QWORD *)(a1[1] + 8LL);
  if ( (_BYTE)v10 )
    v65 = (_WORD *)(v11 - 600);
  else
    v65 = (_WORD *)(v11 - 560);
  v86[1] = a5;
  v86[0] = (__int64)&v83;
  v86[2] = (__int64)&v76;
  v86[4] = (__int64)&v77;
  v83 = (const __m128i *)v85;
  v86[5] = (__int64)v65;
  v84 = 0x100000000LL;
  v86[6] = a3;
  v86[3] = (__int64)a1;
  v12 = *(unsigned int *)(a2 + 8);
  if ( (_DWORD)v12 )
  {
    v13 = *(_QWORD **)a5;
    v14 = 0;
    v66 = *(_BYTE *)(a5 + 8);
    while ( 1 )
    {
      if ( (_BYTE)v10 )
      {
        if ( v14 >= (unsigned int)sub_AF4EB0(*(_QWORD *)a5) )
        {
LABEL_32:
          v46 = sub_2E908B0((_QWORD *)*a1, &v77, v65, v66, v83, (unsigned int)v84, *(_QWORD *)a3, (__int64)v13);
          goto LABEL_45;
        }
      }
      else if ( v14 )
      {
        goto LABEL_32;
      }
      v39 = (const __m128i *)(*(_QWORD *)a2 + 48 * v14);
      if ( v39[2].m128i_i8[8] )
      {
        v42 = (unsigned int)v84;
        v12 = HIDWORD(v84);
        v44 = v83;
        v49 = (unsigned int)v84 + 1LL;
        if ( v49 > HIDWORD(v84) )
        {
          if ( v83 > v39 || v39 >= (const __m128i *)((char *)v83 + 40 * (unsigned int)v84) )
          {
            v7 = (__int64)v85;
            sub_C8D5F0((__int64)&v83, v85, (unsigned int)v84 + 1LL, 0x28u, v49, v9);
            v44 = v83;
            v42 = (unsigned int)v84;
          }
          else
          {
            v7 = (__int64)v85;
            v60 = (char *)v39 - (char *)v83;
            sub_C8D5F0((__int64)&v83, v85, (unsigned int)v84 + 1LL, 0x28u, v49, v9);
            v44 = v83;
            v42 = (unsigned int)v84;
            v39 = (const __m128i *)((char *)v83 + v60);
          }
        }
        goto LABEL_30;
      }
      v40 = *(_DWORD *)(a1[11] + 4LL * v39->m128i_u32[0]);
      v41 = *((_DWORD *)a1 + 71);
      if ( v41 > v40 )
      {
        LODWORD(v88) = *(_DWORD *)(a1[11] + 4LL * v39->m128i_u32[0]);
        v42 = (unsigned int)v84;
        v7 = 0x800000000LL;
        v12 = HIDWORD(v84);
        v43 = (unsigned int)v84 + 1LL;
        memset(v89, 0, 24);
        v44 = v83;
        v39 = (const __m128i *)&v87;
        v87 = 0x800000000LL;
        if ( v43 > HIDWORD(v84) )
        {
          if ( v83 > (const __m128i *)&v87 || &v87 >= (unsigned __int64 *)v83 + 5 * (unsigned int)v84 )
          {
            v7 = (__int64)v85;
            sub_C8D5F0((__int64)&v83, v85, (unsigned int)v84 + 1LL, 0x28u, v43, v9);
            v44 = v83;
            v42 = (unsigned int)v84;
            v39 = (const __m128i *)&v87;
          }
          else
          {
            v57 = (char *)&v87 - (char *)v83;
            v7 = (__int64)v85;
            sub_C8D5F0((__int64)&v83, v85, (unsigned int)v84 + 1LL, 0x28u, v43, v9);
            v44 = v83;
            v42 = (unsigned int)v84;
            v39 = (const __m128i *)((char *)v83 + v57);
          }
        }
LABEL_30:
        v45 = (__m128i *)((char *)v44 + 40 * v42);
        *v45 = _mm_loadu_si128(v39);
        v45[1] = _mm_loadu_si128(v39 + 1);
        v45[2].m128i_i64[0] = v39[2].m128i_i64[0];
        LODWORD(v84) = v84 + 1;
        goto LABEL_24;
      }
      v15 = *((unsigned int *)a1 + 220);
      v16 = v40 - v41;
      v8 = a1[108];
      v10 = v16 % *((_DWORD *)a1 + 72);
      v17 = v16 / *((_DWORD *)a1 + 72);
      if ( !(_DWORD)v15 )
        break;
      v18 = v15 - 1;
      v12 = ((_DWORD)v15 - 1) & (unsigned int)(37 * v10);
      v7 = v8 + 8 * v12;
      v9 = *(unsigned int *)v7;
      v19 = v7;
      if ( (_DWORD)v10 == (_DWORD)v9 )
      {
LABEL_7:
        if ( *(_WORD *)(v19 + 6) )
          goto LABEL_44;
        v20 = (int *)(a1[32] + 24LL * v17);
        v69 = *v20;
      }
      else
      {
        v73 = v12;
        v58 = *(_DWORD *)v7;
        v59 = 1;
        while ( v58 != -1 )
        {
          v62 = v59 + 1;
          v63 = v18 & (unsigned int)(v73 + v59);
          v71 = v62;
          v73 = v63;
          v19 = v8 + 8 * v63;
          v58 = *(_DWORD *)v19;
          if ( (_DWORD)v10 == *(_DWORD *)v19 )
            goto LABEL_7;
          v59 = v71;
        }
        v12 = (unsigned int)v15;
        if ( *(_WORD *)(v8 + 8LL * (unsigned int)v15 + 6) )
          goto LABEL_44;
        v20 = (int *)(a1[32] + 24LL * v17);
        v69 = *v20;
        LODWORD(v12) = v18 & (37 * v10);
        v7 = v8 + 8LL * (unsigned int)v12;
        LODWORD(v9) = *(_DWORD *)v7;
      }
      if ( (_DWORD)v10 != (_DWORD)v9 )
      {
        v61 = 1;
        while ( (_DWORD)v9 != -1 )
        {
          v64 = v61 + 1;
          LODWORD(v12) = v18 & (v61 + v12);
          v7 = v8 + 8LL * (unsigned int)v12;
          LODWORD(v9) = *(_DWORD *)v7;
          if ( (_DWORD)v10 == *(_DWORD *)v7 )
            goto LABEL_10;
          v61 = v64;
        }
        v8 += 8 * v15;
LABEL_43:
        v7 = v8;
      }
LABEL_10:
      v21 = *(unsigned __int16 *)(v7 + 4);
      v22 = *(_BYTE *)(a3 + 24);
      v72 = v21;
      if ( v22 )
      {
        if ( v21 == *(_DWORD *)(a3 + 8) )
          v22 = sub_AF4500((__int64)v13);
      }
      else
      {
        v68 = v21;
        v50 = (_BYTE *)sub_AF3FE0(*(_QWORD *)a3);
        v88 = v51;
        v22 = v51;
        v87 = (unsigned __int64)v50;
        if ( (_BYTE)v51 )
          v22 = v87 != v68;
      }
      v67 = v22;
      v23 = sub_AE2980(*(_QWORD *)(*a1 + 8LL) + 16LL, 0);
      v24 = (__int64 *)a1[2];
      v25 = v20 + 2;
      v26 = v23[1] < v72;
      v87 = (unsigned __int64)v89;
      v88 = 0x500000000LL;
      v27 = *v24;
      if ( v26 )
      {
        (*(void (__fastcall **)(__int64 *, _DWORD *, unsigned __int64 *))(v27 + 592))(v24, v25, &v87);
        if ( *(_BYTE *)(a5 + 8) )
          goto LABEL_34;
      }
      else
      {
        (*(void (__fastcall **)(__int64 *, _DWORD *, unsigned __int64 *))(v27 + 592))(v24, v25, &v87);
        if ( *(_BYTE *)(a5 + 8) )
          goto LABEL_34;
        if ( v67 && sub_AF4590((__int64)v13) )
        {
          v53 = (unsigned int)v88;
          v54 = (unsigned int)v88 + 1LL;
          if ( v54 > HIDWORD(v88) )
          {
            sub_C8D5F0((__int64)&v87, v89, v54, 8u, v30, v31);
            v53 = (unsigned int)v88;
          }
          *(_QWORD *)(v87 + 8 * v53) = 148;
          v56 = v72 >> 3;
          LODWORD(v88) = v88 + 1;
          v55 = (unsigned int)v88;
          if ( (unsigned __int64)(unsigned int)v88 + 1 > HIDWORD(v88) )
          {
            sub_C8D5F0((__int64)&v87, v89, (unsigned int)v88 + 1LL, 8u, v56, v31);
            v55 = (unsigned int)v88;
            v56 = v72 >> 3;
          }
          *(_QWORD *)(v87 + 8 * v55) = v56;
          v33 = 1;
          v32 = (unsigned int)(v88 + 1);
          LODWORD(v88) = v88 + 1;
          goto LABEL_20;
        }
      }
      if ( (unsigned __int8)sub_AF4500((__int64)v13) || *(_BYTE *)(a5 + 9) )
      {
LABEL_34:
        v47 = (unsigned int)v88;
        v48 = (unsigned int)v88 + 1LL;
        if ( v48 > HIDWORD(v88) )
        {
          sub_C8D5F0((__int64)&v87, v89, v48, 8u, v28, v29);
          v47 = (unsigned int)v88;
        }
        v33 = 0;
        *(_QWORD *)(v87 + 8 * v47) = 6;
        v32 = (unsigned int)(v88 + 1);
        LODWORD(v88) = v88 + 1;
        goto LABEL_20;
      }
      v66 = 1;
      v32 = (unsigned int)v88;
      v33 = 0;
LABEL_20:
      v7 = v87;
      v34 = sub_B0DBA0(v13, (_BYTE *)v87, v32, v14, v33);
      v35 = (unsigned int)v84;
      v80 = 0;
      v81 = 0;
      v13 = (_QWORD *)v34;
      v9 = (unsigned int)v84 + 1LL;
      v82 = 0;
      v79 = v69;
      v78 = 0x800000000LL;
      if ( v9 > HIDWORD(v84) )
      {
        if ( v83 > (const __m128i *)&v78 || (v70 = v83, &v78 >= &v83->m128i_i64[5 * (unsigned int)v84]) )
        {
          v7 = (__int64)v85;
          sub_C8D5F0((__int64)&v83, v85, (unsigned int)v84 + 1LL, 0x28u, (__int64)v83, v9);
          v12 = (__int64)v83;
          v35 = (unsigned int)v84;
          v36 = (const __m128i *)&v78;
        }
        else
        {
          v7 = (__int64)v85;
          sub_C8D5F0((__int64)&v83, v85, (unsigned int)v84 + 1LL, 0x28u, (__int64)v83, v9);
          v12 = (__int64)v83;
          v35 = (unsigned int)v84;
          v36 = (const __m128i *)((char *)v83 + (char *)&v78 - (char *)v70);
        }
      }
      else
      {
        v12 = (__int64)v83;
        v36 = (const __m128i *)&v78;
      }
      v37 = (__m128i *)(v12 + 40 * v35);
      *v37 = _mm_loadu_si128(v36);
      v38 = _mm_loadu_si128(v36 + 1);
      LODWORD(v84) = v84 + 1;
      v37[1] = v38;
      v37[2].m128i_i64[0] = v36[2].m128i_i64[0];
      if ( (_QWORD *)v87 != v89 )
        _libc_free(v87);
LABEL_24:
      LOBYTE(v10) = *(_BYTE *)(a5 + 9);
      ++v14;
    }
    if ( *(_WORD *)(v8 + 6) )
      goto LABEL_44;
    v20 = (int *)(a1[32] + 24LL * v17);
    v69 = *v20;
    goto LABEL_43;
  }
LABEL_44:
  v46 = sub_37B70B0(v86, v7, v10, v12, v8, v9);
LABEL_45:
  if ( v83 != (const __m128i *)v85 )
    _libc_free((unsigned __int64)v83);
  if ( v77 )
    sub_B91220((__int64)&v77, (__int64)v77);
  return v46;
}
