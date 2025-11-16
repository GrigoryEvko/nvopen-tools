// Function: sub_EAC980
// Address: 0xeac980
//
__int64 __fastcall sub_EAC980(__int64 a1, unsigned __int8 a2, unsigned int a3)
{
  __int64 v6; // rax
  bool v7; // zf
  unsigned __int64 v8; // r13
  unsigned int v9; // r15d
  unsigned __int64 v11; // rsi
  signed __int64 v12; // rdx
  unsigned __int64 v13; // rcx
  __int64 v14; // rax
  signed __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // r13
  __int64 v18; // rdi
  __int64 v19; // rax
  _QWORD *v20; // rdx
  __m128i v21; // rax
  char v22; // al
  _QWORD *v23; // rcx
  char v24; // al
  __m128i *v25; // rcx
  char v26; // dl
  _QWORD *v27; // rsi
  char v28; // al
  __m128i *v29; // rsi
  __m128i *v30; // rcx
  __int64 v31; // rax
  int v32; // eax
  __int64 v33; // rsi
  unsigned __int64 v34; // rax
  __int64 v35; // r12
  unsigned int v36; // ebx
  void (__fastcall *v37)(__int64, __int64, __int64, _QWORD); // r13
  __int64 v38; // rsi
  __int64 v39; // rdx
  unsigned __int64 v40; // rax
  int v41; // eax
  unsigned __int8 v42; // al
  unsigned __int64 v43; // rcx
  __m128i v44; // xmm0
  __m128i v45; // xmm1
  __m128i v46; // xmm3
  __m128i v47; // xmm5
  __m128i v48; // xmm7
  __int64 v49; // [rsp+0h] [rbp-1E0h]
  __int64 v50; // [rsp+8h] [rbp-1D8h]
  __int64 v51; // [rsp+10h] [rbp-1D0h]
  __int64 v52; // [rsp+18h] [rbp-1C8h]
  __int64 v53; // [rsp+20h] [rbp-1C0h]
  char v54; // [rsp+2Eh] [rbp-1B2h]
  unsigned __int8 v55; // [rsp+2Fh] [rbp-1B1h]
  signed __int64 v56; // [rsp+38h] [rbp-1A8h] BYREF
  unsigned __int64 v57; // [rsp+40h] [rbp-1A0h] BYREF
  __int64 v58; // [rsp+48h] [rbp-198h] BYREF
  __int64 v59; // [rsp+50h] [rbp-190h] BYREF
  unsigned __int64 v60; // [rsp+58h] [rbp-188h] BYREF
  __m128i v61; // [rsp+60h] [rbp-180h] BYREF
  __m128i v62; // [rsp+70h] [rbp-170h] BYREF
  __int64 v63; // [rsp+80h] [rbp-160h]
  _QWORD v64[4]; // [rsp+90h] [rbp-150h] BYREF
  char v65; // [rsp+B0h] [rbp-130h]
  char v66; // [rsp+B1h] [rbp-12Fh]
  __m128i v67; // [rsp+C0h] [rbp-120h] BYREF
  __m128i v68; // [rsp+D0h] [rbp-110h] BYREF
  __int64 v69; // [rsp+E0h] [rbp-100h]
  _QWORD *v70; // [rsp+F0h] [rbp-F0h] BYREF
  __int64 v71; // [rsp+F8h] [rbp-E8h]
  __int16 v72; // [rsp+110h] [rbp-D0h]
  __m128i v73; // [rsp+120h] [rbp-C0h] BYREF
  __m128i v74; // [rsp+130h] [rbp-B0h] BYREF
  __int64 v75; // [rsp+140h] [rbp-A0h]
  __m128i v76; // [rsp+150h] [rbp-90h] BYREF
  __m128i v77; // [rsp+160h] [rbp-80h] BYREF
  __int64 v78; // [rsp+170h] [rbp-70h]
  __m128i v79; // [rsp+180h] [rbp-60h] BYREF
  __m128i v80; // [rsp+190h] [rbp-50h]
  __int64 v81; // [rsp+1A0h] [rbp-40h]

  v55 = a2;
  v6 = sub_ECD690(a1 + 40);
  v7 = *(_BYTE *)(a1 + 869) == 0;
  v57 = 0;
  v58 = 0;
  v8 = v6;
  v59 = 0;
  v60 = 0;
  if ( v7 && (unsigned __int8)sub_EA2540(a1) )
    return 1;
  if ( a3 == 1 && a2 && *(_DWORD *)sub_ECD7B0(a1) == 9 )
  {
    v79.m128i_i64[0] = (__int64)"p2align directive with no operand(s) is ignored";
    LOWORD(v81) = 259;
    sub_EA8060((_QWORD *)a1, v8, (__int64)&v79, 0, 0);
    return (unsigned int)sub_ECE000(a1);
  }
  if ( (unsigned __int8)sub_EAC8B0(a1, &v56) )
    return 1;
  v11 = 26;
  v54 = sub_ECE2A0(a1, 26);
  if ( v54 )
  {
    if ( *(_DWORD *)sub_ECD7B0(a1) == 26 )
    {
      v54 = 0;
    }
    else if ( (unsigned __int8)sub_ECD7C0(a1, &v60) || (unsigned __int8)sub_EAC8B0(a1, &v58) )
    {
      return 1;
    }
    v11 = 26;
    if ( (unsigned __int8)sub_ECE2A0(a1, 26) )
    {
      if ( (unsigned __int8)sub_ECD7C0(a1, &v57) )
        return 1;
      v11 = (unsigned __int64)&v59;
      if ( (unsigned __int8)sub_EAC8B0(a1, &v59) )
        return 1;
    }
  }
  v9 = sub_ECE000(a1);
  if ( (_BYTE)v9 )
    return 1;
  v13 = v56;
  if ( a2 )
  {
    if ( v56 > 31 )
    {
      v11 = v8;
      v79.m128i_i64[0] = (__int64)"invalid alignment value";
      LOWORD(v81) = 259;
      v9 = sub_ECDA70(a1, v8, &v79, 0, 0);
      v14 = 0x80000000LL;
    }
    else
    {
      v14 = 1LL << v56;
    }
    v56 = v14;
  }
  else if ( v56 )
  {
    if ( (v56 & (v56 - 1)) != 0 )
    {
      v11 = v8;
      v79.m128i_i64[0] = (__int64)"alignment must be a power of 2";
      LOWORD(v81) = 259;
      v42 = sub_ECDA70(a1, v8, &v79, 0, 0);
      v13 = v56;
      v55 = v42;
      if ( v56 )
      {
        _BitScanReverse64(&v43, v56);
        v13 = 0x8000000000000000LL >> ((unsigned __int8)v43 ^ 0x3Fu);
        v12 = v13;
        v31 = (unsigned int)v13;
      }
      else
      {
        v31 = 0;
        v12 = 0;
      }
      v56 = v12;
    }
    else
    {
      v31 = (unsigned int)v56;
    }
    if ( v13 == v31 )
    {
      v9 = v55;
    }
    else
    {
      v11 = v8;
      v79.m128i_i64[0] = (__int64)"alignment must be smaller than 2**32";
      LOWORD(v81) = 259;
      v9 = sub_ECDA70(a1, v8, &v79, 0, 0) | v55;
      v56 = 0x80000000LL;
    }
  }
  else
  {
    v56 = 1;
  }
  if ( v57 )
  {
    v15 = v59;
    if ( v59 <= 0 )
    {
      v11 = v57;
      v79.m128i_i64[0] = (__int64)"alignment directive can never be satisfied in this many bytes, ignoring maximum bytes expression";
      LOWORD(v81) = 259;
      v41 = sub_ECDA70(a1, v57, &v79, 0, 0);
      v59 = 0;
      v9 |= v41;
      v15 = 0;
    }
    if ( v56 <= v15 )
    {
      v11 = v57;
      v79.m128i_i64[0] = (__int64)"maximum bytes expression exceeds alignment and has no effect";
      LOWORD(v81) = 259;
      sub_EA8060((_QWORD *)a1, v57, (__int64)&v79, 0, 0);
      v59 = 0;
    }
  }
  v16 = *(_QWORD *)(*(_QWORD *)(a1 + 232) + 288LL);
  v17 = *(_QWORD *)(v16 + 8);
  if ( v54 )
  {
    if ( !v58 || (*(_BYTE *)(v17 + 48) & 0x20) == 0 )
      goto LABEL_55;
    v18 = *(_QWORD *)(v16 + 8);
    v76.m128i_i64[0] = (__int64)"'";
    v19 = *(_QWORD *)(v17 + 136);
    v20 = *(_QWORD **)(v17 + 128);
    LOWORD(v78) = 259;
    v71 = v19;
    v72 = 261;
    v70 = v20;
    v66 = 1;
    v64[0] = " section '";
    v65 = 3;
    v21.m128i_i64[0] = (*(__int64 (__fastcall **)(__int64, __int64, _QWORD *, unsigned __int64))(*(_QWORD *)v17 + 16LL))(
                         v18,
                         261,
                         v20,
                         v13);
    v61.m128i_i64[0] = (__int64)"ignoring non-zero fill value in ";
    v62 = v21;
    v22 = v65;
    LOWORD(v63) = 1283;
    if ( v65 )
    {
      if ( v65 == 1 )
      {
        v44 = _mm_loadu_si128(&v61);
        v45 = _mm_loadu_si128(&v62);
        v69 = v63;
        v24 = v72;
        v67 = v44;
        v68 = v45;
        if ( (_BYTE)v72 )
        {
          if ( (_BYTE)v72 != 1 )
          {
            if ( BYTE1(v69) == 1 )
            {
              v26 = 3;
              v52 = v67.m128i_i64[1];
              v25 = (__m128i *)v67.m128i_i64[0];
LABEL_34:
              if ( HIBYTE(v72) == 1 )
              {
                v27 = v70;
                v51 = v71;
              }
              else
              {
                v27 = &v70;
                v24 = 2;
              }
              v73.m128i_i64[0] = (__int64)v25;
              BYTE1(v75) = v24;
              v28 = v78;
              v73.m128i_i64[1] = v52;
              v74.m128i_i64[0] = (__int64)v27;
              v74.m128i_i64[1] = v51;
              LOBYTE(v75) = v26;
              if ( (_BYTE)v78 )
                goto LABEL_37;
              goto LABEL_53;
            }
LABEL_33:
            v25 = &v67;
            v26 = 2;
            goto LABEL_34;
          }
          goto LABEL_75;
        }
      }
      else
      {
        if ( v66 == 1 )
        {
          v53 = v64[1];
          v23 = (_QWORD *)v64[0];
        }
        else
        {
          v23 = v64;
          v22 = 2;
        }
        v68.m128i_i64[0] = (__int64)v23;
        BYTE1(v69) = v22;
        v24 = v72;
        v67.m128i_i64[0] = (__int64)&v61;
        v68.m128i_i64[1] = v53;
        LOBYTE(v69) = 2;
        if ( (_BYTE)v72 )
        {
          if ( (_BYTE)v72 != 1 )
            goto LABEL_33;
LABEL_75:
          v46 = _mm_loadu_si128(&v68);
          v26 = v69;
          v73 = _mm_loadu_si128(&v67);
          v75 = v69;
          v74 = v46;
          if ( (_BYTE)v69 )
          {
            v28 = v78;
            if ( (_BYTE)v78 )
            {
              if ( (_BYTE)v69 != 1 )
              {
LABEL_37:
                if ( v28 == 1 )
                {
                  v48 = _mm_loadu_si128(&v74);
                  v79 = _mm_loadu_si128(&v73);
                  v81 = v75;
                  v80 = v48;
                }
                else
                {
                  if ( BYTE1(v75) == 1 )
                  {
                    v50 = v73.m128i_i64[1];
                    v29 = (__m128i *)v73.m128i_i64[0];
                  }
                  else
                  {
                    v29 = &v73;
                    v26 = 2;
                  }
                  if ( BYTE1(v78) == 1 )
                  {
                    v49 = v76.m128i_i64[1];
                    v30 = (__m128i *)v76.m128i_i64[0];
                  }
                  else
                  {
                    v30 = &v76;
                    v28 = 2;
                  }
                  v80.m128i_i64[0] = (__int64)v30;
                  v79.m128i_i64[0] = (__int64)v29;
                  v79.m128i_i64[1] = v50;
                  v80.m128i_i64[1] = v49;
                  LOBYTE(v81) = v26;
                  BYTE1(v81) = v28;
                }
                goto LABEL_54;
              }
              v47 = _mm_loadu_si128(&v77);
              v79 = _mm_loadu_si128(&v76);
              v81 = v78;
              v80 = v47;
LABEL_54:
              v11 = v60;
              v32 = sub_EA8060((_QWORD *)a1, v60, (__int64)&v79, 0, 0);
              v58 = 0;
              v9 |= v32;
LABEL_55:
              (*(void (__fastcall **)(__int64, unsigned __int64, signed __int64, unsigned __int64))(*(_QWORD *)v17 + 8LL))(
                v17,
                v11,
                v12,
                v13);
LABEL_56:
              v33 = 0xFFFFFFFFLL;
              if ( v56 )
              {
                _BitScanReverse64(&v34, v56);
                v33 = 63 - ((unsigned int)v34 ^ 0x3F);
              }
              (*(void (__fastcall **)(_QWORD, __int64, __int64, _QWORD, __int64))(**(_QWORD **)(a1 + 232) + 608LL))(
                *(_QWORD *)(a1 + 232),
                v33,
                v58,
                a3,
                v59);
              return v9;
            }
          }
LABEL_53:
          LOWORD(v81) = 256;
          goto LABEL_54;
        }
      }
    }
    else
    {
      LOWORD(v69) = 256;
    }
    LOWORD(v75) = 256;
    goto LABEL_53;
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64, unsigned __int64, signed __int64, unsigned __int64))(*(_QWORD *)v17 + 8LL))(
          v17,
          v11,
          v12,
          v13) )
    goto LABEL_56;
  v35 = *(_QWORD *)(a1 + 232);
  v36 = v59;
  v37 = *(void (__fastcall **)(__int64, __int64, __int64, _QWORD))(*(_QWORD *)v35 + 616LL);
  v38 = 0xFFFFFFFFLL;
  v39 = sub_ECE6C0(*(_QWORD *)(a1 + 8));
  if ( v56 )
  {
    _BitScanReverse64(&v40, v56);
    v38 = 63 - ((unsigned int)v40 ^ 0x3F);
  }
  v37(v35, v38, v39, v36);
  return v9;
}
