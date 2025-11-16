// Function: sub_CBFA60
// Address: 0xcbfa60
//
__int64 __fastcall sub_CBFA60(
        __int64 a1,
        unsigned __int64 a2,
        const __m128i *a3,
        __int64 a4,
        unsigned __int8 a5,
        __int64 a6)
{
  __int64 v10; // rcx
  __m128i *v11; // rdx
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rax
  __int64 result; // rax
  unsigned __int64 v15; // rcx
  __int64 v16; // rbx
  __int64 v17; // rax
  __int64 v18; // r11
  __int64 v19; // r12
  __int64 v20; // rax
  unsigned __int64 v21; // r12
  __m128i *v22; // rcx
  __m128i *v23; // rdx
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rax
  unsigned __int64 v26; // rbx
  __m128i *v27; // r9
  __m128i *v28; // r10
  __m128i v29; // xmm5
  __int64 v30; // rcx
  __int64 *v31; // r15
  char v32; // dl
  unsigned __int8 v33; // di
  __int64 *v34; // rbx
  unsigned __int8 v35; // al
  unsigned __int64 v36; // rdx
  __m128i *v37; // rcx
  __m128i v38; // xmm6
  __m128i v39; // xmm7
  __m128i v40; // xmm1
  __m128i v41; // xmm6
  __m128i v42; // xmm7
  __m128i v43; // xmm0
  __m128i v44; // xmm4
  __m128i v45; // xmm5
  __m128i v46; // xmm2
  __m128i v47; // xmm3
  __m128i v48; // xmm1
  __m128i *v49; // rbx
  unsigned __int64 v50; // r8
  __int8 *v51; // rcx
  char *v52; // r15
  unsigned int v53; // ecx
  unsigned int v54; // ecx
  unsigned int v55; // esi
  __int64 v56; // rdi
  __m128i v57; // xmm7
  __m128i v58; // xmm4
  __m128i v59; // xmm5
  __int64 v60; // [rsp+0h] [rbp-5C0h]
  __int64 v61; // [rsp+8h] [rbp-5B8h]
  unsigned __int64 v62; // [rsp+8h] [rbp-5B8h]
  unsigned __int64 v63; // [rsp+10h] [rbp-5B0h]
  __int64 v64; // [rsp+10h] [rbp-5B0h]
  unsigned __int64 v65; // [rsp+18h] [rbp-5A8h]
  unsigned __int64 v66; // [rsp+18h] [rbp-5A8h]
  __int64 v67; // [rsp+20h] [rbp-5A0h]
  unsigned __int64 v68; // [rsp+20h] [rbp-5A0h]
  __int64 v69; // [rsp+20h] [rbp-5A0h]
  __m128i v71; // [rsp+30h] [rbp-590h] BYREF
  __m128i v72; // [rsp+40h] [rbp-580h] BYREF
  __int64 v73; // [rsp+50h] [rbp-570h]
  __m128i v74; // [rsp+58h] [rbp-568h] BYREF
  __m128i v75; // [rsp+68h] [rbp-558h] BYREF
  __m128i v76; // [rsp+78h] [rbp-548h] BYREF
  __m128i v77; // [rsp+88h] [rbp-538h] BYREF
  __int16 v78; // [rsp+98h] [rbp-528h]
  unsigned __int8 v79; // [rsp+9Ah] [rbp-526h]
  __m128i v80; // [rsp+A0h] [rbp-520h]
  __m128i v81; // [rsp+B0h] [rbp-510h]
  _OWORD v82[5]; // [rsp+C0h] [rbp-500h] BYREF
  __m128i v83; // [rsp+110h] [rbp-4B0h] BYREF
  __m128i v84; // [rsp+120h] [rbp-4A0h]
  _BYTE v85[24]; // [rsp+130h] [rbp-490h] BYREF
  __m128i v86; // [rsp+148h] [rbp-478h] BYREF
  __m128i v87; // [rsp+158h] [rbp-468h] BYREF
  __m128i v88; // [rsp+168h] [rbp-458h] BYREF
  unsigned __int8 v89; // [rsp+178h] [rbp-448h]
  unsigned __int8 v90; // [rsp+179h] [rbp-447h]
  __m128i v91; // [rsp+190h] [rbp-430h] BYREF
  __m128i v92; // [rsp+1A0h] [rbp-420h] BYREF
  __m128i v93; // [rsp+1B0h] [rbp-410h] BYREF
  __m128i v94[64]; // [rsp+1C0h] [rbp-400h] BYREF

  if ( sub_CC23D0() << 10 < a2 )
  {
    _BitScanReverse64(&v15, ((a2 - 1) >> 10) | 1);
    v16 = a1 + (1024LL << v15);
    v62 = 1024LL << v15;
    v66 = a2 - (1024LL << v15);
    v64 = a4 + ((unsigned __int64)(1024LL << v15) >> 10);
    v17 = sub_CC23D0();
    if ( v62 <= 0x400 || (v18 = 64, v17 != 1) )
      v18 = 32 * v17;
    v60 = v18;
    v19 = sub_CBFA60(a1, v62, a3, a4, a5, &v91);
    v20 = sub_CBFA60(v16, v66, a3, v64, a5, &v91.m128i_i8[v60]);
    if ( v19 == 1 )
    {
      v57 = _mm_loadu_si128(&v92);
      v58 = _mm_loadu_si128(&v93);
      v59 = _mm_loadu_si128(v94);
      *(__m128i *)a6 = _mm_loadu_si128(&v91);
      *(__m128i *)(a6 + 16) = v57;
      *(__m128i *)(a6 + 32) = v58;
      *(__m128i *)(a6 + 48) = v59;
      return 2;
    }
    else
    {
      v21 = v20 + v19;
      if ( v21 <= 1 )
      {
        v26 = 0;
        v25 = 0;
      }
      else
      {
        v22 = &v91;
        v23 = &v83;
        v24 = (v21 - 2) >> 1;
        do
        {
          v23->m128i_i64[0] = (__int64)v22;
          v23 = (__m128i *)((char *)v23 + 8);
          v22 += 4;
        }
        while ( v23 != (__m128i *)&v83.m128i_u64[v24 + 1] );
        v25 = v24 + 1;
        v26 = ((v21 - 2) & 0xFFFFFFFFFFFFFFFELL) + 2;
      }
      v67 = v25;
      sub_CC2330((unsigned int)&v83, v25, 1, (_DWORD)a3, 0, 0, a5 | 4, 0, 0, a6);
      result = v67;
      if ( v21 > v26 )
      {
        v27 = (__m128i *)(a6 + 32 * v67);
        v28 = &v91 + 4 * v67;
        v29 = _mm_loadu_si128(v28 + 1);
        *v27 = _mm_loadu_si128(v28);
        v27[1] = v29;
        return v67 + 1;
      }
    }
  }
  else
  {
    if ( a2 <= 0x3FF )
    {
      v63 = a2;
      v13 = 0;
      v65 = 0;
    }
    else
    {
      v10 = a1;
      v11 = &v91;
      v12 = (a2 - 1024) >> 10;
      do
      {
        v11->m128i_i64[0] = v10;
        v11 = (__m128i *)((char *)v11 + 8);
        v10 += 1024;
      }
      while ( v11 != (__m128i *)&v91.m128i_u64[v12 + 1] );
      v13 = v12 + 1;
      v65 = ((a2 - 1024) & 0xFFFFFFFFFFFFFC00LL) + 1024;
      v63 = a2 & 0x3FF;
    }
    v61 = v13;
    sub_CC2330((unsigned int)&v91, v13, 16, (_DWORD)a3, a4, 1, a5, 1, 2, a6);
    result = v61;
    if ( a2 > v65 )
    {
      v30 = a4 + v61;
      v78 = 0;
      v79 = a5;
      v73 = a4 + v61;
      v31 = (__int64 *)(v65 + a1);
      v71 = _mm_loadu_si128(a3);
      v72 = _mm_loadu_si128(a3 + 1);
      v74 = 0;
      v75 = 0;
      v76 = 0;
      v77 = 0;
      if ( v63 <= 0x40 )
      {
        v37 = &v74;
        v36 = 64;
      }
      else
      {
        v32 = 0;
        v68 = (v63 - 65) >> 6;
        v33 = a5;
        v34 = v31;
        v35 = v33;
        while ( 1 )
        {
          sub_CC2280(&v71, v34, 64, v30, (unsigned __int8)(v35 | (v32 == 0)));
          v32 = ++HIBYTE(v78);
          if ( v34 == (__int64 *)((char *)v31 + ((v63 - 65) & 0xFFFFFFFFFFFFFFC0LL)) )
            break;
          v35 = v79;
          v30 = v73;
          v34 += 8;
        }
        result = v61;
        v31 += 8 * v68 + 8;
        v36 = 64LL - (unsigned __int8)v78;
        v37 = (__m128i *)((char *)&v74 + (unsigned __int8)v78);
        v63 = v63 - (v68 << 6) - 64;
      }
      if ( v63 <= v36 )
        LODWORD(v36) = v63;
      if ( (unsigned int)v36 >= 8 )
      {
        v50 = (unsigned __int64)&v37->m128i_u64[1] & 0xFFFFFFFFFFFFFFF8LL;
        v37->m128i_i64[0] = *v31;
        *(__int64 *)((char *)&v37->m128i_i64[-1] + (unsigned int)v36) = *(__int64 *)((char *)v31 + (unsigned int)v36 - 8);
        v51 = &v37->m128i_i8[-v50];
        v52 = (char *)((char *)v31 - v51);
        v53 = (v36 + (_DWORD)v51) & 0xFFFFFFF8;
        if ( v53 >= 8 )
        {
          v54 = v53 & 0xFFFFFFF8;
          v55 = 0;
          do
          {
            v56 = v55;
            v55 += 8;
            *(_QWORD *)(v50 + v56) = *(_QWORD *)&v52[v56];
          }
          while ( v55 < v54 );
        }
      }
      else if ( (v36 & 4) != 0 )
      {
        v37->m128i_i32[0] = *(_DWORD *)v31;
        *(__int32 *)((char *)&v37->m128i_i32[-1] + (unsigned int)v36) = *(_DWORD *)((char *)v31 + (unsigned int)v36 - 4);
      }
      else if ( (_DWORD)v36 )
      {
        v37->m128i_i8[0] = *(_BYTE *)v31;
        if ( (v36 & 2) != 0 )
          *(__int16 *)((char *)&v37->m128i_i16[-1] + (unsigned int)v36) = *(_WORD *)((char *)v31 + (unsigned int)v36 - 2);
      }
      v38 = _mm_loadu_si128(&v74);
      v39 = _mm_loadu_si128(&v75);
      v89 = v78 + v36;
      v40 = _mm_loadu_si128(&v71);
      *(__m128i *)&v85[8] = v38;
      v86 = v39;
      v41 = _mm_loadu_si128(&v76);
      v42 = _mm_loadu_si128(&v77);
      v43 = _mm_loadu_si128(&v72);
      *(_QWORD *)v85 = v73;
      v87 = v41;
      v44 = _mm_loadu_si128((const __m128i *)v85);
      v45 = _mm_loadu_si128((const __m128i *)&v85[16]);
      v88 = v42;
      v46 = _mm_loadu_si128((const __m128i *)&v86.m128i_u64[1]);
      v47 = _mm_loadu_si128((const __m128i *)&v87.m128i_u64[1]);
      v90 = v79 | (HIBYTE(v78) == 0) | 2;
      v83 = v40;
      v80 = v40;
      v48 = _mm_loadu_si128((const __m128i *)&v88.m128i_u64[1]);
      LOBYTE(v78) = v78 + v36;
      v84 = v43;
      v81 = v43;
      v82[0] = v44;
      v82[1] = v45;
      v82[2] = v46;
      v82[3] = v47;
      v82[4] = v48;
      v69 = result;
      v49 = (__m128i *)(a6 + 32 * result);
      sub_CC2280(&v83, (char *)v82 + 8, v89, v73, v90);
      *v49 = v83;
      v49[1] = v84;
      return v69 + 1;
    }
  }
  return result;
}
