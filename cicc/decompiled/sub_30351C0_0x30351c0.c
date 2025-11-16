// Function: sub_30351C0
// Address: 0x30351c0
//
void __fastcall sub_30351C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v9; // r8
  __int64 v10; // r9
  char v11; // al
  __int64 v12; // rax
  __int64 *v13; // rdx
  unsigned __int64 *v14; // r15
  __int64 *v15; // rbx
  __int64 v16; // rdx
  unsigned __int64 v17; // rcx
  char v18; // al
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rax
  _BYTE *v23; // r15
  __int64 v24; // r14
  __int64 v25; // rbx
  __int64 v26; // r9
  __int64 v27; // rax
  _QWORD *v28; // rax
  unsigned __int64 v29; // rcx
  __int64 v30; // rax
  _QWORD *v31; // rax
  __int64 v32; // rax
  __int64 v33; // rsi
  unsigned __int64 v34; // rcx
  __int64 v35; // rax
  int v36; // edx
  __int64 v37; // r15
  __int64 v38; // r13
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v41; // rsi
  unsigned __int16 *v42; // rcx
  __int64 v43; // rdx
  __int64 v44; // rax
  unsigned int v45; // r12d
  __int16 v46; // ax
  __int64 v47; // rdx
  unsigned int v48; // edx
  __int64 v49; // rax
  __m128i v50; // xmm0
  __int64 v51; // rax
  unsigned int v52; // r13d
  __int64 v53; // r14
  __int64 v54; // rax
  __int64 v55; // rdx
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rax
  __int64 v59; // rax
  __m128i v60; // xmm0
  __int64 v61; // [rsp-10h] [rbp-270h]
  __int64 v62; // [rsp-8h] [rbp-268h]
  __int64 v63; // [rsp+8h] [rbp-258h]
  const void *v64; // [rsp+8h] [rbp-258h]
  __m128i v65; // [rsp+10h] [rbp-250h] BYREF
  __int64 v66; // [rsp+20h] [rbp-240h]
  _BYTE *v67; // [rsp+28h] [rbp-238h]
  _BYTE *v68; // [rsp+30h] [rbp-230h]
  _BYTE **v69; // [rsp+38h] [rbp-228h]
  __m128i v70; // [rsp+40h] [rbp-220h] BYREF
  __m128i v71; // [rsp+50h] [rbp-210h] BYREF
  __m128i v72; // [rsp+60h] [rbp-200h] BYREF
  unsigned __int64 v73; // [rsp+70h] [rbp-1F0h] BYREF
  char v74; // [rsp+78h] [rbp-1E8h]
  unsigned __int64 v75; // [rsp+80h] [rbp-1E0h] BYREF
  __int64 v76; // [rsp+88h] [rbp-1D8h]
  unsigned __int64 v77[2]; // [rsp+90h] [rbp-1D0h] BYREF
  _BYTE v78[128]; // [rsp+A0h] [rbp-1C0h] BYREF
  _BYTE *v79; // [rsp+120h] [rbp-140h] BYREF
  __int64 v80; // [rsp+128h] [rbp-138h]
  _BYTE v81[304]; // [rsp+130h] [rbp-130h] BYREF

  v69 = &v79;
  v65.m128i_i64[0] = (__int64)v77;
  v67 = v81;
  v79 = v81;
  v68 = v78;
  v77[0] = (unsigned __int64)v78;
  v70.m128i_i64[0] = a6;
  v80 = 0x1000000000LL;
  v77[1] = 0x1000000000LL;
  if ( !sub_BCAC40(a3, 128) )
  {
    v11 = *(_BYTE *)(a3 + 8);
    if ( v11 != 5 )
    {
      if ( v11 == 15 )
      {
        v12 = sub_AE4AC0(a2, a3);
        v13 = *(__int64 **)(a3 + 16);
        v65.m128i_i64[0] = (__int64)&v13[*(unsigned int *)(a3 + 12)];
        if ( v13 != (__int64 *)v65.m128i_i64[0] )
        {
          v14 = (unsigned __int64 *)(v12 + 24);
          v63 = a4;
          v15 = v13;
          v66 = (__int64)&v75;
          do
          {
            v16 = *v15;
            v17 = *v14;
            ++v15;
            v14 += 2;
            v18 = *((_BYTE *)v14 - 8);
            v69 = (_BYTE **)v16;
            v75 = v17;
            LOBYTE(v76) = v18;
            v19 = sub_CA1930((_BYTE *)v66);
            sub_30351C0(a1, a2, v69, v63, a5, v19 + v70.m128i_i64[0]);
          }
          while ( (__int64 *)v65.m128i_i64[0] != v15 );
        }
        goto LABEL_7;
      }
      if ( v11 == 16 )
      {
        v69 = *(_BYTE ***)(a3 + 24);
        v65.m128i_i8[0] = sub_AE5020(a2, (__int64)v69);
        v20 = sub_9208B0(a2, (__int64)v69);
        v76 = v21;
        v75 = ((1LL << v65.m128i_i8[0]) + ((unsigned __int64)(v20 + 7) >> 3) - 1) >> v65.m128i_i8[0] << v65.m128i_i8[0];
        v65.m128i_i64[0] = sub_CA1930(&v75);
        v66 = *(int *)(a3 + 32);
        if ( v66 )
        {
          v22 = a5;
          v23 = 0;
          v24 = a4;
          v25 = v22;
          do
          {
            v26 = v65.m128i_i64[0] * (int)v23++;
            sub_30351C0(a1, a2, v69, v24, v25, v70.m128i_i64[0] + v26);
          }
          while ( (_BYTE *)v66 != v23 );
        }
        goto LABEL_7;
      }
      v36 = a3;
      v37 = a5;
      v38 = 0;
      sub_34B8FD0(a1, a2, v36, (_DWORD)v69, 0, v65.m128i_i32[0], v70.m128i_i64[0]);
      v39 = v61;
      v40 = v62;
      v69 = (_BYTE **)(8LL * (unsigned int)v80);
      v41 = a5 + 16;
      v64 = (const void *)(a5 + 16);
      if ( !(_DWORD)v80 )
        goto LABEL_7;
      while ( 1 )
      {
        v71 = _mm_loadu_si128((const __m128i *)&v79[2 * v38]);
        v70.m128i_i64[0] = *(_QWORD *)(v77[0] + v38);
        if ( v71.m128i_i16[0] )
        {
          if ( (unsigned __int16)(v71.m128i_i16[0] - 17) > 0xD3u )
            goto LABEL_44;
          if ( (unsigned __int16)(v71.m128i_i16[0] - 176) > 0x34u )
            goto LABEL_31;
        }
        else
        {
          if ( !sub_30070B0((__int64)&v71) )
          {
LABEL_44:
            v49 = *(unsigned int *)(a4 + 8);
            v50 = _mm_load_si128(&v71);
            if ( v49 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
            {
              v41 = a4 + 16;
              v65 = v50;
              sub_C8D5F0(a4, (const void *)(a4 + 16), v49 + 1, 0x10u, v39, v40);
              v49 = *(unsigned int *)(a4 + 8);
              v50 = _mm_load_si128(&v65);
            }
            *(__m128i *)(*(_QWORD *)a4 + 16 * v49) = v50;
            ++*(_DWORD *)(a4 + 8);
            if ( v37 )
            {
              v51 = *(unsigned int *)(v37 + 8);
              if ( v51 + 1 > (unsigned __int64)*(unsigned int *)(v37 + 12) )
              {
                sub_C8D5F0(v37, v64, v51 + 1, 8u, v39, v40);
                v51 = *(unsigned int *)(v37 + 8);
              }
              v41 = v70.m128i_i64[0];
              *(_QWORD *)(*(_QWORD *)v37 + 8 * v51) = v70.m128i_i64[0];
              ++*(_DWORD *)(v37 + 8);
            }
            goto LABEL_50;
          }
          if ( !sub_3007100((__int64)&v71) )
          {
LABEL_54:
            v45 = sub_3007130((__int64)&v71, v41);
            goto LABEL_55;
          }
        }
        sub_CA17B0(
          "Possible incorrect use of EVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use E"
          "VT::getVectorElementCount() instead");
        if ( !v71.m128i_i16[0] )
          goto LABEL_54;
        if ( (unsigned __int16)(v71.m128i_i16[0] - 176) > 0x34u )
        {
          v44 = v71.m128i_u16[0] - 1;
          v45 = word_4456340[v44];
LABEL_32:
          v46 = word_4456580[v44];
          v47 = 0;
          goto LABEL_33;
        }
        sub_CA17B0(
          "Possible incorrect use of MVT::getVectorNumElements() for scalable vector. Scalable flag may be dropped, use M"
          "VT::getVectorElementCount() instead");
LABEL_31:
        v42 = word_4456340;
        v43 = v71.m128i_u16[0];
        v44 = v71.m128i_u16[0] - 1;
        v45 = word_4456340[v44];
        if ( v71.m128i_i16[0] )
          goto LABEL_32;
LABEL_55:
        v46 = sub_3009970((__int64)&v71, v41, v43, (__int64)v42, v39);
LABEL_33:
        v72.m128i_i64[1] = v47;
        v72.m128i_i16[0] = v46;
        if ( ((unsigned __int16)(v46 - 10) <= 1u || v46 == 6) && v45 && (v45 & 1) == 0 )
        {
          v48 = v45 - 1;
          if ( (v45 & (v45 - 1)) != 0 )
          {
            if ( v46 != 5 )
              goto LABEL_58;
LABEL_76:
            if ( (v45 & (v48 | 3)) != 0 && v45 != 3 )
            {
              if ( v45 == 2 )
              {
                v45 = 1;
                v72.m128i_i64[1] = 0;
                v72.m128i_i16[0] = 47;
              }
LABEL_58:
              v65.m128i_i64[0] = v38;
              v66 = a4 + 16;
              v52 = 0;
              v53 = v70.m128i_i64[0];
              do
              {
                v59 = *(unsigned int *)(a4 + 8);
                v60 = _mm_load_si128(&v72);
                if ( v59 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
                {
                  v41 = v66;
                  v70 = v60;
                  sub_C8D5F0(a4, (const void *)v66, v59 + 1, 0x10u, v39, v40);
                  v59 = *(unsigned int *)(a4 + 8);
                  v60 = _mm_load_si128(&v70);
                }
                *(__m128i *)(*(_QWORD *)a4 + 16 * v59) = v60;
                ++*(_DWORD *)(a4 + 8);
                if ( v37 )
                {
                  if ( v72.m128i_i16[0] )
                  {
                    if ( v72.m128i_i16[0] == 1 || (unsigned __int16)(v72.m128i_i16[0] - 504) <= 7u )
                      goto LABEL_89;
                    v54 = *(_QWORD *)&byte_444C4A0[16 * v72.m128i_u16[0] - 16];
                    LOBYTE(v55) = byte_444C4A0[16 * v72.m128i_u16[0] - 8];
                  }
                  else
                  {
                    v54 = sub_3007260((__int64)&v72);
                    v75 = v54;
                    v76 = v55;
                  }
                  v74 = v55;
                  v73 = v52 * ((unsigned __int64)(v54 + 7) >> 3);
                  v56 = sub_CA1930(&v73);
                  v57 = *(unsigned int *)(v37 + 8);
                  v58 = v53 + v56;
                  v40 = v57 + 1;
                  if ( v57 + 1 > (unsigned __int64)*(unsigned int *)(v37 + 12) )
                  {
                    v41 = (__int64)v64;
                    v70.m128i_i64[0] = v58;
                    sub_C8D5F0(v37, v64, v57 + 1, 8u, v39, v40);
                    v57 = *(unsigned int *)(v37 + 8);
                    v58 = v70.m128i_i64[0];
                  }
                  *(_QWORD *)(*(_QWORD *)v37 + 8 * v57) = v58;
                  ++*(_DWORD *)(v37 + 8);
                }
                ++v52;
              }
              while ( v52 != v45 );
              v38 = v65.m128i_i64[0];
              goto LABEL_50;
            }
            v72.m128i_i64[1] = 0;
            v72.m128i_i16[0] = 37;
            v45 = (v45 + 3) >> 2;
            goto LABEL_57;
          }
          switch ( v46 )
          {
            case 10:
              v72.m128i_i64[1] = 0;
              v41 = 138;
              v72.m128i_i16[0] = 138;
              break;
            case 11:
              v72.m128i_i64[1] = 0;
              v72.m128i_i16[0] = 127;
              break;
            case 6:
              v72.m128i_i64[1] = 0;
              v72.m128i_i16[0] = 47;
              break;
            default:
LABEL_89:
              BUG();
          }
          v45 >>= 1;
        }
        else if ( v46 == 5 )
        {
          if ( !v45 )
            goto LABEL_50;
          v48 = v45 - 1;
          goto LABEL_76;
        }
LABEL_57:
        if ( v45 )
          goto LABEL_58;
LABEL_50:
        v38 += 8;
        if ( (_BYTE **)v38 == v69 )
          goto LABEL_7;
      }
    }
  }
  v27 = *(unsigned int *)(a4 + 8);
  if ( v27 + 1 > (unsigned __int64)*(unsigned int *)(a4 + 12) )
  {
    sub_C8D5F0(a4, (const void *)(a4 + 16), v27 + 1, 0x10u, v9, v10);
    v27 = *(unsigned int *)(a4 + 8);
  }
  v28 = (_QWORD *)(*(_QWORD *)a4 + 16 * v27);
  *v28 = 8;
  v28[1] = 0;
  v29 = *(unsigned int *)(a4 + 12);
  v30 = (unsigned int)(*(_DWORD *)(a4 + 8) + 1);
  *(_DWORD *)(a4 + 8) = v30;
  if ( v30 + 1 > v29 )
  {
    sub_C8D5F0(a4, (const void *)(a4 + 16), v30 + 1, 0x10u, v9, v10);
    v30 = *(unsigned int *)(a4 + 8);
  }
  v31 = (_QWORD *)(*(_QWORD *)a4 + 16 * v30);
  *v31 = 8;
  v31[1] = 0;
  ++*(_DWORD *)(a4 + 8);
  if ( a5 )
  {
    v32 = *(unsigned int *)(a5 + 8);
    if ( v32 + 1 > (unsigned __int64)*(unsigned int *)(a5 + 12) )
    {
      sub_C8D5F0(a5, (const void *)(a5 + 16), v32 + 1, 8u, v9, v10);
      v32 = *(unsigned int *)(a5 + 8);
    }
    v33 = v70.m128i_i64[0];
    *(_QWORD *)(*(_QWORD *)a5 + 8 * v32) = v70.m128i_i64[0];
    v34 = *(unsigned int *)(a5 + 12);
    v35 = (unsigned int)(*(_DWORD *)(a5 + 8) + 1);
    *(_DWORD *)(a5 + 8) = v35;
    if ( v35 + 1 > v34 )
    {
      sub_C8D5F0(a5, (const void *)(a5 + 16), v35 + 1, 8u, v9, v10);
      v35 = *(unsigned int *)(a5 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a5 + 8 * v35) = v33 + 8;
    ++*(_DWORD *)(a5 + 8);
  }
LABEL_7:
  if ( (_BYTE *)v77[0] != v68 )
    _libc_free(v77[0]);
  if ( v79 != v67 )
    _libc_free((unsigned __int64)v79);
}
