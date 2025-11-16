// Function: sub_1B78FF0
// Address: 0x1b78ff0
//
__int64 __fastcall sub_1B78FF0(
        __int64 a1,
        __m128 a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  unsigned int v10; // edx
  __int64 v11; // rax
  double v12; // xmm4_8
  double v13; // xmm5_8
  const __m128i *v14; // rax
  __int64 v15; // rbx
  char v16; // cl
  __int32 v17; // edi
  __int64 v18; // r12
  int v19; // esi
  __int64 result; // rax
  __int64 *v21; // rax
  __int64 v22; // r12
  __int64 v23; // rbx
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // r13
  __int64 v27; // rsi
  double v28; // xmm4_8
  double v29; // xmm5_8
  __int64 v30; // rax
  __int64 v31; // rcx
  int v32; // r13d
  unsigned int i; // r14d
  int v34; // r8d
  __int64 v35; // r9
  __int64 v36; // rax
  __int64 *v37; // rax
  __int64 *v38; // r13
  __int64 v39; // r14
  __int64 v40; // rsi
  __int64 v41; // r12
  __int64 v42; // rdx
  __int64 v43; // rcx
  __int64 v44; // rcx
  int v45; // r8d
  int v46; // r9d
  __int64 v47; // r12
  __int64 v48; // rax
  __int64 v49; // rdx
  __int64 v50; // r12
  __int128 v51; // rdi
  __int64 v52; // rax
  int v53; // r8d
  int v54; // r9d
  unsigned __int64 v55; // rax
  __int64 v56; // rdx
  _QWORD *v57; // rax
  _QWORD *j; // rdx
  _QWORD *v59; // rax
  __int64 *v60; // rax
  unsigned __int64 *v61; // rax
  __int64 v62; // rax
  _QWORD *v63; // rax
  __int64 v64; // [rsp+0h] [rbp-180h]
  __int64 v65; // [rsp+10h] [rbp-170h]
  __int64 v66; // [rsp+18h] [rbp-168h]
  __int64 v67; // [rsp+18h] [rbp-168h]
  unsigned __int64 v68; // [rsp+20h] [rbp-160h]
  __int64 **v69; // [rsp+30h] [rbp-150h]
  __int64 **v70; // [rsp+38h] [rbp-148h]
  __int64 v71; // [rsp+40h] [rbp-140h]
  __int64 *v72; // [rsp+40h] [rbp-140h]
  __int32 v73; // [rsp+48h] [rbp-138h]
  __int64 *v74; // [rsp+70h] [rbp-110h] BYREF
  __int64 v75; // [rsp+78h] [rbp-108h]
  __int64 v76[8]; // [rsp+80h] [rbp-100h] BYREF
  _BYTE *v77; // [rsp+C0h] [rbp-C0h] BYREF
  __int64 v78; // [rsp+C8h] [rbp-B8h]
  _BYTE v79[176]; // [rsp+D0h] [rbp-B0h] BYREF

  v10 = *(_DWORD *)(a1 + 80);
  if ( v10 )
  {
    while ( 1 )
    {
      while ( 1 )
      {
        v14 = (const __m128i *)(*(_QWORD *)(a1 + 72) + 24LL * v10 - 24);
        a2 = (__m128)_mm_loadu_si128(v14);
        v15 = v14[1].m128i_i64[0];
        v16 = v14->m128i_i8[0] & 3;
        v17 = v14->m128i_i32[1];
        v18 = v14->m128i_i64[1];
        v19 = ((unsigned __int32)v14->m128i_i32[0] >> 2) & 0x1FFFFFFF;
        *(_DWORD *)(a1 + 80) = v10 - 1;
        *(_DWORD *)(a1 + 16) = v19;
        if ( v16 == 2 )
        {
          v30 = sub_1B75C50(a1, v15, *(double *)a2.m128_u64, a3, a4);
          sub_15E5930(v18, v30);
          goto LABEL_5;
        }
        if ( v16 != 3 )
          break;
        sub_1B78E20(a1, v18, a2, a3, a4, a5, a6, a7, a8, a9);
        v10 = *(_DWORD *)(a1 + 80);
        if ( !v10 )
          goto LABEL_9;
      }
      if ( v16 != 1 )
      {
        v11 = sub_1B75C50(a1, v15, *(double *)a2.m128_u64, a3, a4);
        sub_15E5440(v18, v11);
        sub_1B78850(a1, v18, a2, a3, a4, a5, v12, v13, a8, a9);
        goto LABEL_5;
      }
      v31 = (unsigned int)(*(_DWORD *)(a1 + 224) - v17);
      v73 = *(_DWORD *)(a1 + 224) - v17;
      v66 = *(unsigned int *)(a1 + 224) - v31;
      v65 = 8 * v31;
      v71 = *(_QWORD *)(a1 + 216) + 8 * v31;
      v68 = v31;
      v77 = v79;
      v78 = 0x1000000000LL;
      if ( v15 )
      {
        v32 = *(_QWORD *)(*(_QWORD *)v15 + 32LL);
        if ( v32 )
        {
          for ( i = 0; i != v32; ++i )
          {
            v35 = sub_15A0A60(v15, i);
            v36 = (unsigned int)v78;
            if ( (unsigned int)v78 >= HIDWORD(v78) )
            {
              v64 = v35;
              sub_16CD150((__int64)&v77, v79, 0, 8, v34, v35);
              v36 = (unsigned int)v78;
              v35 = v64;
            }
            *(_QWORD *)&v77[8 * v36] = v35;
            LODWORD(v78) = v78 + 1;
          }
        }
      }
      if ( a2.m128_i8[3] < 0 )
      {
        v59 = (_QWORD *)sub_16498A0(v18);
        v60 = (__int64 *)sub_1643330(v59);
        v70 = (__int64 **)sub_1647190(v60, 0);
        v61 = *(unsigned __int64 **)(**(_QWORD **)v71 + 16LL);
        v74 = (__int64 *)*v61;
        v62 = v61[1];
        v76[0] = (__int64)v70;
        v75 = v62;
        v63 = (_QWORD *)sub_16498A0(v18);
        v69 = (__int64 **)sub_1645600(v63, &v74, 3, 0);
      }
      v37 = (__int64 *)v71;
      v72 = (__int64 *)(v71 + 8 * v66);
      if ( v37 != v72 )
        break;
      v49 = (unsigned int)v78;
LABEL_37:
      *((_QWORD *)&v51 + 1) = v77;
      *(_QWORD *)&v51 = *(_QWORD *)(*(_QWORD *)v18 + 24LL);
      v52 = sub_159DFD0(v51, v49, v31);
      sub_15E5440(v18, v52);
      if ( v77 != v79 )
        _libc_free((unsigned __int64)v77);
      v55 = *(unsigned int *)(a1 + 224);
      if ( v68 < v55 )
        goto LABEL_47;
      if ( v68 > v55 )
      {
        if ( v68 > *(unsigned int *)(a1 + 228) )
        {
          sub_16CD150(a1 + 216, (const void *)(a1 + 232), v68, 8, v53, v54);
          v55 = *(unsigned int *)(a1 + 224);
        }
        v56 = *(_QWORD *)(a1 + 216);
        v57 = (_QWORD *)(v56 + 8 * v55);
        for ( j = (_QWORD *)(v65 + v56); j != v57; ++v57 )
        {
          if ( v57 )
            *v57 = 0;
        }
LABEL_47:
        *(_DWORD *)(a1 + 224) = v73;
      }
LABEL_5:
      v10 = *(_DWORD *)(a1 + 80);
      if ( !v10 )
        goto LABEL_9;
    }
    v67 = v18;
    v38 = v37;
    while ( 1 )
    {
      v50 = *v38;
      if ( a2.m128_i8[3] < 0 )
        break;
      v47 = sub_1B75C50(a1, *v38, *(double *)a2.m128_u64, a3, a4);
      v48 = (unsigned int)v78;
      if ( (unsigned int)v78 >= HIDWORD(v78) )
        goto LABEL_35;
LABEL_32:
      ++v38;
      *(_QWORD *)&v77[8 * v48] = v47;
      v49 = (unsigned int)(v78 + 1);
      LODWORD(v78) = v78 + 1;
      if ( v72 == v38 )
      {
        v18 = v67;
        goto LABEL_37;
      }
    }
    v39 = sub_1B75C50(a1, *(_QWORD *)(v50 - 24LL * (*(_DWORD *)(v50 + 20) & 0xFFFFFFF)), *(double *)a2.m128_u64, a3, a4);
    v40 = *(_QWORD *)(v50 + 24 * (1LL - (*(_DWORD *)(v50 + 20) & 0xFFFFFFF)));
    v41 = sub_1B75C50(a1, v40, *(double *)a2.m128_u64, a3, a4);
    v76[2] = sub_15A06D0(v70, v40, v42, v43);
    v76[1] = v41;
    v74 = v76;
    v76[0] = v39;
    v75 = 0x800000003LL;
    v47 = sub_159F090(v69, v76, 3, v44);
    if ( v74 != v76 )
      _libc_free((unsigned __int64)v74);
    v48 = (unsigned int)v78;
    if ( (unsigned int)v78 < HIDWORD(v78) )
      goto LABEL_32;
LABEL_35:
    sub_16CD150((__int64)&v77, v79, 0, 8, v45, v46);
    v48 = (unsigned int)v78;
    goto LABEL_32;
  }
LABEL_9:
  result = *(unsigned int *)(a1 + 192);
  for ( *(_DWORD *)(a1 + 16) = 0; (_DWORD)result; result = *(unsigned int *)(a1 + 192) )
  {
    v21 = (__int64 *)(*(_QWORD *)(a1 + 184) + 16 * result - 16);
    v22 = v21[1];
    v23 = *v21;
    v21[1] = 0;
    v24 = (unsigned int)(*(_DWORD *)(a1 + 192) - 1);
    *(_DWORD *)(a1 + 192) = v24;
    v25 = *(_QWORD *)(a1 + 184) + 16 * v24;
    v26 = *(_QWORD *)(v25 + 8);
    if ( v26 )
    {
      sub_157EF40(*(_QWORD *)(v25 + 8));
      j_j___libc_free_0(v26, 64);
    }
    v27 = sub_1B75C50(a1, v23, *(double *)a2.m128_u64, a3, a4);
    if ( !v27 )
      v27 = v23;
    sub_164D160(v22, v27, a2, a3, a4, a5, v28, v29, a8, a9);
    if ( v22 )
    {
      sub_157EF40(v22);
      j_j___libc_free_0(v22, 64);
    }
  }
  return result;
}
