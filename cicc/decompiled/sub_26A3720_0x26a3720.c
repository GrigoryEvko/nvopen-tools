// Function: sub_26A3720
// Address: 0x26a3720
//
__int64 __fastcall sub_26A3720(__int64 a1, signed int a2, __int64 a3, __int64 a4, __int64 a5, __int64 i)
{
  __int64 v6; // r14
  __int64 v7; // r13
  __int64 v8; // r12
  int v9; // eax
  __int64 v10; // rcx
  int v11; // esi
  unsigned int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rax
  unsigned int v17; // eax
  __int64 v18; // rcx
  unsigned int v19; // eax
  __int64 *v20; // rdx
  __int64 v21; // rbx
  __int64 *v22; // rax
  char v23; // dl
  __int64 v24; // r15
  __int64 v25; // rax
  __int64 v26; // r14
  __int64 v27; // r12
  __int64 v28; // r15
  unsigned __int64 v29; // rbx
  int v30; // eax
  __int64 v31; // rsi
  int v32; // ecx
  unsigned int v33; // edx
  __int64 *v34; // rax
  __int64 v35; // rdi
  __int64 v36; // rsi
  __m128i v37; // rax
  __int64 v38; // rbx
  __int64 v39; // rdx
  __int64 v40; // rax
  unsigned __int64 v41; // r15
  __int64 v42; // rax
  unsigned __int64 v43; // rdx
  __int64 v44; // r10
  int v45; // eax
  __int64 v46; // rsi
  __int64 v47; // rax
  int v48; // eax
  int v49; // r11d
  __int64 v50; // rax
  __int64 v51; // rax
  unsigned __int64 v52; // rbx
  int v53; // eax
  int v54; // r9d
  __int64 v55; // rsi
  _QWORD *v56; // [rsp+8h] [rbp-1F8h]
  __int64 v57; // [rsp+20h] [rbp-1E0h]
  __int64 v59; // [rsp+30h] [rbp-1D0h]
  __int8 v60; // [rsp+3Fh] [rbp-1C1h]
  signed int v61; // [rsp+5Ch] [rbp-1A4h] BYREF
  __m128i v62; // [rsp+60h] [rbp-1A0h] BYREF
  __m128i v63; // [rsp+70h] [rbp-190h] BYREF
  __m128i v64; // [rsp+80h] [rbp-180h]
  __m128i v65; // [rsp+90h] [rbp-170h]
  __int64 *v66; // [rsp+A0h] [rbp-160h] BYREF
  __int64 v67; // [rsp+A8h] [rbp-158h]
  _QWORD v68[16]; // [rsp+B0h] [rbp-150h] BYREF
  __int64 v69; // [rsp+130h] [rbp-D0h] BYREF
  __int64 *v70; // [rsp+138h] [rbp-C8h]
  __int64 v71; // [rsp+140h] [rbp-C0h]
  int v72; // [rsp+148h] [rbp-B8h]
  unsigned __int8 v73; // [rsp+14Ch] [rbp-B4h]
  char v74; // [rsp+150h] [rbp-B0h] BYREF

  v6 = a1;
  v7 = a1 + 32LL * a2 + 104;
  v8 = a3;
  v9 = *(_DWORD *)(a1 + 32LL * a2 + 128);
  v61 = a2;
  v10 = *(_QWORD *)(a1 + 32LL * a2 + 112);
  if ( v9 )
  {
    v11 = v9 - 1;
    v12 = (v9 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
    v13 = (__int64 *)(v10 + 16LL * v12);
    v14 = *v13;
    if ( v8 == *v13 )
    {
LABEL_3:
      v15 = v13[1];
    }
    else
    {
      a5 = *v13;
      v17 = v12;
      for ( i = 1; ; i = (unsigned int)(i + 1) )
      {
        if ( a5 == -4096 )
          goto LABEL_8;
        v17 = v11 & (i + v17);
        a5 = *(_QWORD *)(v10 + 16LL * v17);
        if ( v8 == a5 )
          break;
      }
      v48 = 1;
      while ( v14 != -4096 )
      {
        v54 = v48 + 1;
        v12 = v11 & (v48 + v12);
        v13 = (__int64 *)(v10 + 16LL * v12);
        v14 = *v13;
        if ( a5 == *v13 )
          goto LABEL_3;
        v48 = v54;
      }
      v15 = 0;
    }
    v64.m128i_i64[0] = v15;
    v64.m128i_i8[8] = 1;
    return v64.m128i_i64[0];
  }
LABEL_8:
  v69 = 0;
  v18 = 1;
  v66 = v68;
  v71 = 16;
  v72 = 0;
  v73 = 1;
  v68[0] = v8;
  v60 = 0;
  v57 = 0;
  v70 = (__int64 *)&v74;
  v67 = 0x1000000001LL;
  v19 = 1;
  v62 = 0;
LABEL_9:
  while ( v19 )
  {
    v20 = v66;
    v21 = v66[v19 - 1];
    LODWORD(v67) = v19 - 1;
    if ( !(_BYTE)v18 )
      goto LABEL_16;
    v22 = v70;
    v20 = &v70[HIDWORD(v71)];
    if ( v70 != v20 )
    {
      while ( v21 != *v22 )
      {
        if ( v20 == ++v22 )
          goto LABEL_48;
      }
      goto LABEL_15;
    }
LABEL_48:
    if ( HIDWORD(v71) < (unsigned int)v71 )
    {
      ++HIDWORD(v71);
      *v20 = v21;
      ++v69;
LABEL_17:
      v24 = *(_QWORD *)(v21 + 40);
      if ( *(_QWORD *)(v24 + 56) == v21 + 24 )
      {
LABEL_33:
        if ( v24 != *(_QWORD *)(v8 + 40) )
          goto LABEL_34;
      }
      else
      {
        v25 = v6;
        v59 = v8;
        v26 = *(_QWORD *)(v21 + 40);
        v27 = v21;
        v28 = v25;
        do
        {
          v29 = *(_QWORD *)(v27 + 24) & 0xFFFFFFFFFFFFFFF8LL;
          if ( !v29 )
            break;
          v30 = *(_DWORD *)(v7 + 24);
          v31 = *(_QWORD *)(v7 + 8);
          v27 = v29 - 24;
          if ( v30 )
          {
            v32 = v30 - 1;
            v33 = (v30 - 1) & (((unsigned int)v27 >> 9) ^ ((unsigned int)v27 >> 4));
            v34 = (__int64 *)(v31 + 16LL * v33);
            v35 = *v34;
            if ( v27 == *v34 )
            {
              v36 = v28;
              v8 = v59;
              v24 = v26;
              v6 = v36;
LABEL_23:
              v37.m128i_i64[0] = v34[1];
LABEL_24:
              if ( v60 )
              {
                if ( v37.m128i_i64[0] != v57 )
                {
LABEL_26:
                  v64.m128i_i8[8] = 1;
                  LOBYTE(v18) = v73;
                  v64.m128i_i64[0] = 0;
                  goto LABEL_27;
                }
              }
              else
              {
                v60 = 1;
              }
              goto LABEL_32;
            }
            v44 = *v34;
            a5 = v33;
            v45 = 1;
            while ( v44 != -4096 )
            {
              v49 = v45 + 1;
              v50 = v32 & (unsigned int)(a5 + v45);
              a5 = (unsigned int)v50;
              v44 = *(_QWORD *)(v31 + 16 * v50);
              if ( v27 == v44 )
              {
                v51 = v28;
                v52 = v29 - 24;
                v8 = v59;
                v24 = v26;
                v6 = v51;
                v53 = 1;
                while ( v35 != -4096 )
                {
                  a5 = (unsigned int)(v53 + 1);
                  v33 = v32 & (v53 + v33);
                  v34 = (__int64 *)(v31 + 16LL * v33);
                  v35 = *v34;
                  if ( v52 == *v34 )
                    goto LABEL_23;
                  v53 = a5;
                }
                v37.m128i_i64[0] = 0;
                goto LABEL_24;
              }
              v45 = v49;
            }
          }
          if ( (unsigned __int8)(*(_BYTE *)(v29 - 24) - 34) <= 0x33u )
          {
            v46 = 0x8000000000041LL;
            if ( _bittest64(&v46, (unsigned int)*(unsigned __int8 *)(v29 - 24) - 34) )
            {
              v56 = (_QWORD *)(v29 + 48);
              if ( !(unsigned __int8)sub_A747A0((_QWORD *)(v29 + 48), "no_openmp", 9u)
                && !(unsigned __int8)sub_B49590(v29 - 24, "no_openmp", 9u)
                && !(unsigned __int8)sub_A747A0(v56, "no_openmp_routines", 0x12u)
                && !(unsigned __int8)sub_B49590(v29 - 24, "no_openmp_routines", 0x12u)
                && !(unsigned __int8)sub_A747A0(v56, "no_openmp_constructs", 0x14u)
                && !(unsigned __int8)sub_B49590(v29 - 24, "no_openmp_constructs", 0x14u) )
              {
                v37.m128i_i64[0] = sub_26A3510(v28, a4, v29 - 24, (unsigned int *)&v61);
                v65 = v37;
                v63 = v37;
                if ( v37.m128i_i8[8] )
                {
                  if ( !v60 )
                  {
                    v55 = v28;
                    v60 = v37.m128i_i8[8];
                    v24 = v26;
                    v8 = v59;
                    v6 = v55;
                    v62 = _mm_loadu_si128(&v63);
LABEL_32:
                    v57 = v37.m128i_i64[0];
                    goto LABEL_33;
                  }
                  if ( v57 != v37.m128i_i64[0] )
                    goto LABEL_26;
                }
              }
            }
          }
        }
        while ( *(_QWORD *)(*(_QWORD *)(v29 + 16) + 56LL) != v29 );
        v8 = v59;
        v47 = v28;
        v24 = v26;
        v6 = v47;
        if ( v24 != *(_QWORD *)(v59 + 40) )
          goto LABEL_34;
      }
      if ( v60 )
      {
        LOBYTE(v18) = v73;
        v62.m128i_i64[0] = v57;
        v62.m128i_i8[8] = v60;
        v64 = _mm_loadu_si128(&v62);
        goto LABEL_27;
      }
LABEL_34:
      v38 = *(_QWORD *)(v24 + 16);
      if ( v38 )
      {
        while ( 1 )
        {
          v39 = *(_QWORD *)(v38 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v39 - 30) <= 0xAu )
            break;
          v38 = *(_QWORD *)(v38 + 8);
          if ( !v38 )
          {
            v19 = v67;
            v18 = v73;
            goto LABEL_9;
          }
        }
LABEL_38:
        v40 = *(_QWORD *)(v39 + 40);
        v41 = *(_QWORD *)(v40 + 48) & 0xFFFFFFFFFFFFFFF8LL;
        if ( v41 != v40 + 48 )
        {
          if ( !v41 )
            BUG();
          if ( (unsigned int)*(unsigned __int8 *)(v41 - 24) - 30 <= 0xA )
          {
            v42 = (unsigned int)v67;
            v43 = (unsigned int)v67 + 1LL;
            if ( v43 > HIDWORD(v67) )
            {
              sub_C8D5F0((__int64)&v66, v68, v43, 8u, a5, i);
              v42 = (unsigned int)v67;
            }
            v66[v42] = v41 - 24;
            LODWORD(v67) = v67 + 1;
          }
        }
        while ( 1 )
        {
          v38 = *(_QWORD *)(v38 + 8);
          if ( !v38 )
            break;
          v39 = *(_QWORD *)(v38 + 24);
          if ( (unsigned __int8)(*(_BYTE *)v39 - 30) <= 0xAu )
            goto LABEL_38;
        }
      }
      v19 = v67;
      v18 = v73;
    }
    else
    {
LABEL_16:
      sub_C8CC70((__int64)&v69, v21, (__int64)v20, v18, a5, i);
      v18 = v73;
      if ( v23 )
        goto LABEL_17;
LABEL_15:
      v19 = v67;
    }
  }
  v62.m128i_i64[0] = v57;
  v62.m128i_i8[8] = v60;
  v64 = _mm_loadu_si128(&v62);
LABEL_27:
  if ( !(_BYTE)v18 )
    _libc_free((unsigned __int64)v70);
  if ( v66 != v68 )
    _libc_free((unsigned __int64)v66);
  return v64.m128i_i64[0];
}
