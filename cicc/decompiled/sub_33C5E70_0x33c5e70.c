// Function: sub_33C5E70
// Address: 0x33c5e70
//
void __fastcall sub_33C5E70(__int64 a1, __int64 a2)
{
  int v3; // eax
  __int64 v4; // rdx
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // r15
  const __m128i *v8; // r13
  int v9; // eax
  __int64 v10; // r15
  int v11; // edx
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // r13
  int v15; // eax
  __int64 v16; // r9
  __m128i *v17; // rsi
  __int64 v18; // r12
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // r15
  __int64 v23; // rax
  __int64 v24; // r11
  __int64 v25; // rcx
  __int64 v26; // r9
  __m128i *v27; // rax
  void *v28; // rsi
  const __m128i *v29; // r13
  unsigned int v30; // eax
  __int64 v31; // r9
  __int64 v32; // rax
  unsigned __int64 v33; // r8
  __int64 v34; // rdx
  const __m128i *v35; // r13
  __m128i *v36; // rax
  unsigned __int32 v37; // edx
  __int64 v38; // rax
  __m128i v39; // xmm0
  __m128i v40; // xmm2
  __int64 v41; // r14
  __int64 v42; // rsi
  __int64 v43; // r12
  __int64 v44; // rdx
  __int64 v45; // r13
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // r9
  __int64 v49; // rax
  __int64 v50; // rdx
  __int64 v51; // r8
  __int64 v52; // r9
  __m128i *v53; // rax
  __int64 v54; // rsi
  __int64 v55; // rax
  int v56; // edx
  __int64 v57; // rbx
  int v58; // r12d
  char *v59; // r13
  __int128 v60; // [rsp-20h] [rbp-1E0h]
  __int128 v61; // [rsp-10h] [rbp-1D0h]
  __int64 v62; // [rsp+8h] [rbp-1B8h]
  int v63; // [rsp+10h] [rbp-1B0h]
  __int8 *v64; // [rsp+10h] [rbp-1B0h]
  int v65; // [rsp+18h] [rbp-1A8h]
  __int64 v66; // [rsp+28h] [rbp-198h]
  void *v67; // [rsp+28h] [rbp-198h]
  __int64 v68; // [rsp+30h] [rbp-190h]
  __int64 v69; // [rsp+30h] [rbp-190h]
  __int64 v70; // [rsp+38h] [rbp-188h]
  int v71; // [rsp+6Ch] [rbp-154h] BYREF
  const __m128i *v72; // [rsp+70h] [rbp-150h] BYREF
  __m128i *v73; // [rsp+78h] [rbp-148h]
  const __m128i *v74; // [rsp+80h] [rbp-140h]
  void *src[2]; // [rsp+90h] [rbp-130h] BYREF
  __int64 v76[2]; // [rsp+A0h] [rbp-120h]
  int v77[4]; // [rsp+B0h] [rbp-110h]
  __m128i v78; // [rsp+C0h] [rbp-100h] BYREF
  __m128i v79; // [rsp+D0h] [rbp-F0h] BYREF
  __int64 v80; // [rsp+E0h] [rbp-E0h]

  v3 = *(_DWORD *)(a2 + 4);
  v4 = *(_QWORD *)(a1 + 960);
  v72 = 0;
  v73 = 0;
  v5 = *(_QWORD *)(v4 + 32);
  v74 = 0;
  v6 = ((v3 & 0x7FFFFFFu) >> 1) - 1;
  if ( v6 )
  {
    v7 = 40 * v6;
    v8 = (const __m128i *)sub_22077B0(40 * v6);
    v9 = *(_DWORD *)(a2 + 4);
    v72 = v8;
    v73 = (__m128i *)v8;
    v74 = (const __m128i *)((char *)v8 + v7);
    v66 = ((v9 & 0x7FFFFFFu) >> 1) - 1;
    if ( (v9 & 0x7FFFFFFu) >> 1 == 1 )
    {
      v4 = *(_QWORD *)(a1 + 960);
    }
    else
    {
      v10 = 0;
      do
      {
        if ( (_DWORD)v10 == -2 )
        {
          v12 = 32;
          v11 = 0;
        }
        else
        {
          v11 = v10 + 1;
          v12 = 32LL * (unsigned int)(2 * v10 + 3);
        }
        v13 = *(_QWORD *)(a2 - 8);
        ++v10;
        v14 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 960) + 56LL)
                        + 8LL * *(unsigned int *)(*(_QWORD *)(v13 + v12) + 44LL));
        v68 = *(_QWORD *)(v13 + 32LL * (unsigned int)(2 * v10));
        if ( v5 )
        {
          v15 = sub_FF0300(v5, *(_QWORD *)(a2 + 40), v11);
          v16 = v68;
          LODWORD(src[0]) = v15;
        }
        else
        {
          sub_F02DB0(src, 1u, (*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) >> 1);
          v16 = v68;
        }
        v78.m128i_i64[1] = v16;
        v79.m128i_i64[0] = v16;
        v17 = v73;
        v78.m128i_i32[0] = 0;
        v79.m128i_i64[1] = v14;
        LODWORD(v80) = src[0];
        if ( v73 == v74 )
        {
          sub_337D270((unsigned __int64 *)&v72, v73, &v78);
        }
        else
        {
          if ( v73 )
          {
            *v73 = _mm_loadu_si128(&v78);
            v17[1] = _mm_loadu_si128(&v79);
            v17[2].m128i_i64[0] = v80;
            v17 = v73;
          }
          v73 = (__m128i *)((char *)v17 + 40);
        }
      }
      while ( v10 != v66 );
      v4 = *(_QWORD *)(a1 + 960);
    }
  }
  v18 = *(_QWORD *)(*(_QWORD *)(v4 + 56) + 8LL * *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(a2 - 8) + 32LL) + 44LL));
  sub_35D8F40(&v72);
  v71 = 0;
  v67 = (void *)sub_33C5C00(a1, a2, (__int64 *)&v72, (unsigned int *)&v71);
  v22 = *(_QWORD *)(*(_QWORD *)(a1 + 960) + 744LL);
  if ( v73 == v72 )
  {
    sub_2E33F80(*(_QWORD *)(*(_QWORD *)(a1 + 960) + 744LL), v18, -1, v19, v20, v21);
    if ( v18 != sub_3374B60(a1, v22) )
    {
      v41 = *(_QWORD *)(a1 + 864);
      v42 = v18;
      v43 = sub_33EEAD0(v41, v18);
      v45 = v44;
      v49 = sub_3373A60(a1, v42, v44, v46, v47, v48);
      v78.m128i_i64[0] = 0;
      v51 = v49;
      v52 = v50;
      v53 = *(__m128i **)a1;
      v78.m128i_i32[2] = *(_DWORD *)(a1 + 848);
      if ( v53 )
      {
        if ( &v78 != &v53[3] )
        {
          v54 = v53[3].m128i_i64[0];
          v78.m128i_i64[0] = v54;
          if ( v54 )
          {
            v69 = v51;
            v70 = v50;
            sub_B96E90((__int64)&v78, v54, 1);
            v51 = v69;
            v52 = v70;
          }
        }
      }
      *((_QWORD *)&v61 + 1) = v45;
      *(_QWORD *)&v61 = v43;
      *((_QWORD *)&v60 + 1) = v52;
      *(_QWORD *)&v60 = v51;
      v55 = sub_3406EB0(v41, 301, (unsigned int)&v78, 1, 0, v52, v60, v61);
      v57 = v55;
      v58 = v56;
      if ( v55 )
      {
        nullsub_1875(v55, v41, 0);
        *(_QWORD *)(v41 + 384) = v57;
        *(_DWORD *)(v41 + 392) = v58;
        sub_33E2B60(v41, 0);
      }
      else
      {
        *(_QWORD *)(v41 + 384) = 0;
        *(_DWORD *)(v41 + 392) = v56;
      }
      if ( v78.m128i_i64[0] )
        sub_B91220((__int64)&v78, v78.m128i_i64[0]);
    }
  }
  else
  {
    v23 = *(_QWORD *)(a1 + 864);
    v24 = *(_QWORD *)(a1 + 896);
    v25 = *(_QWORD *)(v23 + 112);
    v26 = *(_QWORD *)(v23 + 104);
    LODWORD(src[1]) = *(_DWORD *)(a1 + 848);
    src[0] = 0;
    v27 = *(__m128i **)a1;
    if ( *(_QWORD *)a1 )
    {
      if ( src == (void **)&v27[3] || (v28 = (void *)v27[3].m128i_i64[0], (src[0] = v28) == 0) )
      {
        v78.m128i_i64[0] = 0;
      }
      else
      {
        v62 = v25;
        v63 = v26;
        v65 = v24;
        sub_B96E90((__int64)src, (__int64)v28, 1);
        LODWORD(v24) = v65;
        LODWORD(v26) = v63;
        v25 = v62;
        v78.m128i_i64[0] = (__int64)src[0];
        if ( src[0] )
        {
          sub_B976B0((__int64)src, (unsigned __int8 *)src[0], (__int64)&v78);
          v25 = v62;
          src[0] = 0;
          LODWORD(v26) = v63;
          LODWORD(v24) = v65;
        }
      }
    }
    else
    {
      v78.m128i_i64[0] = 0;
    }
    v78.m128i_i32[2] = (__int32)src[1];
    v79.m128i_i8[0] = 1;
    sub_35DD2C0(v24, (unsigned int)&v72, a2, (unsigned int)&v78, v18, v26, v25);
    if ( v79.m128i_i8[0] )
    {
      v79.m128i_i8[0] = 0;
      if ( v78.m128i_i64[0] )
        sub_B91220((__int64)&v78, v78.m128i_i64[0]);
    }
    if ( src[0] )
      sub_B91220((__int64)src, (__int64)src[0]);
    sub_35DB0F0(*(_QWORD *)(a1 + 896), &v72, a2);
    v78.m128i_i64[0] = (__int64)&v79;
    v29 = v72;
    v78.m128i_i64[1] = 0x400000000LL;
    v64 = &v73[-3].m128i_i8[8];
    v30 = sub_3373D80(a1, (__int64)v67, v18);
    if ( v71
      && v18 == *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 960) + 56LL)
                          + 8LL * *(unsigned int *)(*(_QWORD *)(*(_QWORD *)(a2 - 8) + 32LL) + 44LL)) )
    {
      v30 = sub_33652A0(v30, v71);
    }
    v77[2] = v30;
    v32 = v78.m128i_u32[2];
    src[1] = (void *)v29;
    src[0] = v67;
    v33 = v78.m128i_u32[2] + 1LL;
    v34 = v78.m128i_i64[0];
    v76[1] = 0;
    v76[0] = (__int64)v64;
    *(_QWORD *)v77 = 0;
    v35 = (const __m128i *)src;
    if ( v33 > v78.m128i_u32[3] )
    {
      if ( v78.m128i_i64[0] > (unsigned __int64)src
        || (unsigned __int64)src >= v78.m128i_i64[0] + 48 * (unsigned __int64)v78.m128i_u32[2] )
      {
        sub_C8D5F0((__int64)&v78, &v79, v78.m128i_u32[2] + 1LL, 0x30u, v33, v31);
        v34 = v78.m128i_i64[0];
        v32 = v78.m128i_u32[2];
      }
      else
      {
        v59 = (char *)src - v78.m128i_i64[0];
        sub_C8D5F0((__int64)&v78, &v79, v78.m128i_u32[2] + 1LL, 0x30u, v33, v31);
        v34 = v78.m128i_i64[0];
        v32 = v78.m128i_u32[2];
        v35 = (const __m128i *)&v59[v78.m128i_i64[0]];
      }
    }
    v36 = (__m128i *)(v34 + 48 * v32);
    *v36 = _mm_loadu_si128(v35);
    v36[1] = _mm_loadu_si128(v35 + 1);
    v36[2] = _mm_loadu_si128(v35 + 2);
    v37 = v78.m128i_i32[2] + 1;
    for ( v78.m128i_i32[2] = v37; v78.m128i_i32[2]; v37 = v78.m128i_u32[2] )
    {
      while ( 1 )
      {
        v38 = v78.m128i_i64[0] + 48LL * v37;
        v39 = _mm_loadu_si128((const __m128i *)(v38 - 48));
        *(__m128i *)src = v39;
        *(__m128i *)v76 = _mm_loadu_si128((const __m128i *)(v38 - 32));
        v40 = _mm_loadu_si128((const __m128i *)(v38 - 16));
        v78.m128i_i32[2] = v37 - 1;
        *(__m128i *)v77 = v40;
        if ( -858993459 * (unsigned int)((v76[0] - v39.m128i_i64[1]) >> 3) + 1 > 3
          && *(_DWORD *)(*(_QWORD *)(a1 + 856) + 648LL)
          && !(unsigned __int8)sub_B2D610(**(_QWORD **)(v18 + 32), 18) )
        {
          break;
        }
        sub_33C4220(
          a1,
          **(_BYTE ***)(a2 - 8),
          v22,
          v18,
          v33,
          v31,
          v39,
          (__int64)src[0],
          (__m128i *)src[1],
          v76[0],
          v76[1],
          v77[0],
          v77[2]);
        v37 = v78.m128i_u32[2];
        if ( !v78.m128i_i32[2] )
          goto LABEL_35;
      }
      sub_33C5190(a1, (__int64)&v78, (__int64 *)src, **(_BYTE ***)(a2 - 8), v22);
    }
LABEL_35:
    if ( (__m128i *)v78.m128i_i64[0] != &v79 )
      _libc_free(v78.m128i_u64[0]);
  }
  if ( v72 )
    j_j___libc_free_0((unsigned __int64)v72);
}
