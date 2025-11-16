// Function: sub_3034720
// Address: 0x3034720
//
__int64 __fastcall sub_3034720(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  unsigned __int16 *v3; // rdx
  unsigned __int16 v4; // ax
  __int64 v6; // rsi
  __int64 v7; // rax
  __int64 v8; // rdx
  unsigned int v9; // ecx
  unsigned __int64 v10; // rbx
  __int64 v11; // rax
  __m128i v12; // xmm0
  __m128i v13; // xmm1
  __int64 v14; // r13
  __int64 v15; // r15
  int v16; // eax
  int v17; // eax
  int v19; // edx
  int v20; // eax
  __int64 v21; // rax
  unsigned __int64 v22; // rbx
  unsigned int v23; // ecx
  int v24; // r9d
  int v25; // r10d
  int v26; // eax
  bool v27; // al
  int v28; // ebx
  __int128 v29; // rax
  __int64 v30; // rdi
  int v31; // r9d
  __int128 v32; // rax
  int v33; // r9d
  __int64 v34; // rsi
  __int64 v35; // rdx
  unsigned __int64 v36; // rdx
  char v37; // al
  unsigned int v38; // eax
  unsigned int v39; // edx
  __int64 v40; // rax
  __int64 v41; // rcx
  __int64 v42; // rax
  unsigned int v43; // r14d
  __int64 v44; // rdi
  unsigned __int64 v45; // rax
  __int64 v46; // rdx
  unsigned int v47; // edx
  unsigned __int64 v48; // rax
  unsigned int v49; // r14d
  int v50; // eax
  __int64 v51; // rcx
  __int64 *v52; // rsi
  __int64 v53; // rax
  __int64 v54; // rdx
  __int64 v55; // rax
  int v56; // eax
  int v57; // eax
  unsigned __int64 v58; // rdx
  int v59; // eax
  unsigned __int64 v60; // rax
  __int128 v61; // [rsp-20h] [rbp-130h]
  __int64 *v62; // [rsp+8h] [rbp-108h]
  unsigned int v63; // [rsp+10h] [rbp-100h]
  int v64; // [rsp+10h] [rbp-100h]
  int v65; // [rsp+18h] [rbp-F8h]
  int v66; // [rsp+18h] [rbp-F8h]
  int v67; // [rsp+18h] [rbp-F8h]
  __int128 v68; // [rsp+20h] [rbp-F0h]
  __int128 v69; // [rsp+20h] [rbp-F0h]
  __m128i v70; // [rsp+30h] [rbp-E0h] BYREF
  __int64 v71; // [rsp+40h] [rbp-D0h]
  __int64 v72; // [rsp+48h] [rbp-C8h]
  __m128i v73; // [rsp+50h] [rbp-C0h]
  __m128i v74; // [rsp+60h] [rbp-B0h]
  int v75; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v76; // [rsp+78h] [rbp-98h]
  __int64 v77; // [rsp+80h] [rbp-90h] BYREF
  int v78; // [rsp+88h] [rbp-88h]
  unsigned __int64 v79; // [rsp+90h] [rbp-80h] BYREF
  unsigned int v80; // [rsp+98h] [rbp-78h]
  unsigned __int64 v81; // [rsp+A0h] [rbp-70h] BYREF
  unsigned int v82; // [rsp+A8h] [rbp-68h]
  unsigned __int64 v83; // [rsp+B0h] [rbp-60h] BYREF
  unsigned int v84; // [rsp+B8h] [rbp-58h]
  __int64 v85; // [rsp+C0h] [rbp-50h]
  __int64 v86; // [rsp+C8h] [rbp-48h]
  __int64 v87; // [rsp+D0h] [rbp-40h] BYREF
  __int64 v88; // [rsp+D8h] [rbp-38h]

  v2 = 0;
  v3 = *(unsigned __int16 **)(a1 + 48);
  v4 = *v3;
  v76 = *((_QWORD *)v3 + 1);
  LOWORD(v75) = v4;
  if ( (unsigned __int16)(v4 - 7) > 1u )
    return v2;
  v6 = *(_QWORD *)(a1 + 80);
  v77 = v6;
  if ( v6 )
  {
    sub_B96E90((__int64)&v77, v6, 1);
    v4 = v75;
    v78 = *(_DWORD *)(a1 + 72);
    if ( !(_WORD)v75 )
    {
      v7 = sub_3007260((__int64)&v75);
      v85 = v7;
      v86 = v8;
      goto LABEL_5;
    }
    v19 = (unsigned __int16)v75;
    if ( (_WORD)v75 == 1 )
      goto LABEL_86;
  }
  else
  {
    v78 = *(_DWORD *)(a1 + 72);
    v19 = v4;
  }
  if ( (unsigned __int16)(v4 - 504) <= 7u )
    goto LABEL_86;
  v8 = 16LL * (v19 - 1);
  v7 = *(_QWORD *)&byte_444C4A0[v8];
  LOBYTE(v8) = byte_444C4A0[v8 + 8];
LABEL_5:
  v87 = v7;
  LOBYTE(v88) = v8;
  v10 = sub_CA1930(&v87);
  v11 = *(_QWORD *)(a1 + 40);
  v12 = _mm_loadu_si128((const __m128i *)v11);
  v13 = _mm_loadu_si128((const __m128i *)(v11 + 40));
  v14 = *(_QWORD *)v11;
  v15 = *(_QWORD *)(v11 + 40);
  v16 = *(_DWORD *)(a1 + 24);
  *((_QWORD *)&v68 + 1) = v12.m128i_i64[1];
  v70 = v13;
  if ( v16 != 58 )
  {
    if ( v16 != 190 )
      goto LABEL_19;
    v17 = *(_DWORD *)(v15 + 24);
    if ( v17 != 11 && v17 != 35 )
      goto LABEL_9;
    v34 = *(_QWORD *)(v15 + 96);
    v80 = *(_DWORD *)(v34 + 32);
    if ( v80 > 0x40 )
      sub_C43780((__int64)&v79, (const void **)(v34 + 24));
    else
      v79 = *(_QWORD *)(v34 + 24);
    if ( !(_WORD)v75 )
    {
      v87 = sub_3007260((__int64)&v75);
      v88 = v35;
      v36 = v87;
      v37 = v88;
      goto LABEL_31;
    }
    if ( (_WORD)v75 != 1 && (unsigned __int16)(v75 - 504) > 7u )
    {
      v55 = 16LL * ((unsigned __int16)v75 - 1);
      v36 = *(_QWORD *)&byte_444C4A0[v55];
      v37 = byte_444C4A0[v55 + 8];
LABEL_31:
      v83 = v36;
      LOBYTE(v84) = v37;
      v38 = sub_CA1930(&v83);
      v39 = v38;
      if ( v80 <= 0x40 )
      {
        if ( v80 )
        {
          v40 = (__int64)(v79 << (64 - (unsigned __int8)v80)) >> (64 - (unsigned __int8)v80);
          v41 = v39;
          if ( v40 < 0 )
            goto LABEL_9;
        }
        else
        {
          v41 = v38;
          v40 = 0;
        }
        if ( v41 <= v40 )
          goto LABEL_9;
LABEL_48:
        v84 = v39;
        if ( v39 > 0x40 )
        {
          sub_C43690((__int64)&v83, 1, 0);
          v82 = v84;
          if ( v84 > 0x40 )
          {
            sub_C43780((__int64)&v81, (const void **)&v83);
LABEL_51:
            sub_C47AC0((__int64)&v81, (__int64)&v79);
            if ( v84 > 0x40 && v83 )
              j_j___libc_free_0_0(v83);
            v53 = sub_34007B0(*(_QWORD *)(a2 + 16), (unsigned int)&v81, (unsigned int)&v77, v75, v76, 0, 0);
            v9 = 0;
            v71 = v53;
            v15 = v53;
            v70.m128i_i64[0] = v53;
            v72 = v54;
            v70.m128i_i64[1] = (unsigned int)v54 | v70.m128i_i64[1] & 0xFFFFFFFF00000000LL;
            if ( v82 > 0x40 && v81 )
              j_j___libc_free_0_0(v81);
            if ( v80 > 0x40 && v79 )
              j_j___libc_free_0_0(v79);
            goto LABEL_19;
          }
        }
        else
        {
          v83 = 1;
          v82 = v39;
        }
        v81 = v83;
        goto LABEL_51;
      }
      v49 = v80 + 1;
      v63 = v38;
      v62 = (__int64 *)v79;
      if ( (*(_QWORD *)(v79 + 8LL * ((v80 - 1) >> 6)) & (1LL << ((unsigned __int8)v80 - 1))) != 0 )
      {
        if ( v49 - (unsigned int)sub_C44500((__int64)&v79) > 0x40 || *v62 < 0 )
          goto LABEL_61;
        v50 = sub_C44500((__int64)&v79);
        v51 = v63;
        v39 = v63;
        v52 = v62;
        if ( v49 - v50 > 0x40 )
          goto LABEL_48;
      }
      else
      {
        if ( v49 - (unsigned int)sub_C444A0((__int64)&v79) > 0x40 )
          goto LABEL_61;
        if ( *v62 < 0 )
          goto LABEL_61;
        v57 = sub_C444A0((__int64)&v79);
        v51 = v63;
        v39 = v63;
        v52 = v62;
        if ( v49 - v57 > 0x40 )
          goto LABEL_61;
      }
      if ( v51 > *v52 )
        goto LABEL_48;
LABEL_61:
      v2 = 0;
      if ( v79 )
        j_j___libc_free_0_0(v79);
      goto LABEL_10;
    }
LABEL_86:
    BUG();
  }
  v20 = *(_DWORD *)(v14 + 24);
  if ( v20 == 35 || v20 == 11 )
  {
    v9 = 0;
    v74 = _mm_load_si128(&v70);
    v73 = v12;
    v70.m128i_i64[0] = v12.m128i_i64[0];
    *((_QWORD *)&v68 + 1) = v74.m128i_u32[2] | v12.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    v70.m128i_i64[1] = v12.m128i_u32[2] | v70.m128i_i64[1] & 0xFFFFFFFF00000000LL;
    v21 = v15;
    v15 = v14;
    v14 = v21;
  }
LABEL_19:
  v22 = v10 >> 1;
  if ( !(unsigned __int8)sub_3033330(v14, v22, &v81, v9) )
    goto LABEL_9;
  v25 = v81;
  if ( (_DWORD)v81 == 2 )
    goto LABEL_9;
  v26 = *(_DWORD *)(v15 + 24);
  if ( v26 == 11 || v26 == 35 )
  {
    v42 = *(_QWORD *)(v15 + 96);
    v43 = *(_DWORD *)(v42 + 32);
    v44 = v42 + 24;
    if ( (_DWORD)v81 == 1 )
    {
      if ( v43 > 0x40 )
      {
        v67 = v81;
        v59 = sub_C444A0(v44);
        v25 = v67;
      }
      else
      {
        v58 = *(_QWORD *)(v42 + 24);
        v59 = *(_DWORD *)(v42 + 32);
        if ( v58 )
        {
          _BitScanReverse64(&v60, v58);
          v59 = v43 - 64 + (v60 ^ 0x3F);
        }
      }
      v27 = (unsigned int)v22 >= v43 - v59;
    }
    else
    {
      v45 = *(_QWORD *)(v42 + 24);
      v46 = 1LL << ((unsigned __int8)v43 - 1);
      if ( v43 > 0x40 )
      {
        v64 = v81;
        if ( (*(_QWORD *)(v45 + 8LL * ((v43 - 1) >> 6)) & v46) != 0 )
          v56 = sub_C44500(v44);
        else
          v56 = sub_C444A0(v44);
        v25 = v64;
        v47 = v43 + 1 - v56;
      }
      else if ( (v46 & v45) != 0 )
      {
        if ( v43 )
        {
          v47 = v43 - 63;
          v48 = ~(v45 << (64 - (unsigned __int8)v43));
          if ( v48 )
          {
            _BitScanReverse64(&v48, v48);
            v47 = v43 + 1 - (v48 ^ 0x3F);
          }
        }
        else
        {
          v47 = 1;
        }
      }
      else
      {
        v47 = 1;
        if ( v45 )
        {
          _BitScanReverse64(&v45, v45);
          v47 = 65 - (v45 ^ 0x3F);
        }
      }
      v27 = (unsigned int)v22 >= v47;
    }
  }
  else
  {
    v65 = v81;
    if ( !(unsigned __int8)sub_3033330(v15, v22, &v83, v23) )
    {
LABEL_9:
      v2 = 0;
      goto LABEL_10;
    }
    v25 = v65;
    v27 = (_DWORD)v81 == (_DWORD)v83;
  }
  if ( !v27 )
    goto LABEL_9;
  *(_QWORD *)&v68 = v14;
  v28 = (unsigned __int16)(((_WORD)v75 != 7) + 6);
  v66 = v25;
  *(_QWORD *)&v29 = sub_33FAF80(*(_QWORD *)(a2 + 16), 216, (unsigned int)&v77, v28, 0, v24, v68);
  v30 = *(_QWORD *)(a2 + 16);
  *((_QWORD *)&v61 + 1) = v70.m128i_i64[1];
  v70.m128i_i64[0] = v15;
  *(_QWORD *)&v61 = v15;
  v69 = v29;
  *(_QWORD *)&v32 = sub_33FAF80(v30, 216, (unsigned int)&v77, v28, 0, v31, v61);
  v2 = sub_3406EB0(
         *(_QWORD *)(a2 + 16),
         531 - ((unsigned int)(v66 == 0) - 1),
         (unsigned int)&v77,
         v75,
         v76,
         v33,
         v69,
         v32);
LABEL_10:
  if ( v77 )
    sub_B91220((__int64)&v77, v77);
  return v2;
}
