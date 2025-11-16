// Function: sub_84C4B0
// Address: 0x84c4b0
//
__int64 __fastcall sub_84C4B0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        _BOOL4 a4,
        __m128i *a5,
        __int64 **a6,
        unsigned int a7,
        int a8,
        unsigned int a9,
        int a10,
        int a11,
        unsigned int a12,
        int a13,
        __int64 a14,
        FILE *a15,
        unsigned int a16,
        __int64 *a17,
        _DWORD *a18,
        __int64 a19,
        __int64 *a20)
{
  const __m128i *v20; // rbx
  FILE *v21; // r15
  __int64 result; // rax
  __int64 *v23; // rcx
  int *v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __int64 v30; // r11
  _BYTE *v31; // r10
  __int64 v32; // r13
  int v33; // eax
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // r11
  __m128i *j; // r12
  char v39; // al
  __int64 v40; // rdx
  __int64 v41; // rax
  char v42; // al
  __int64 v43; // rsi
  __int64 i; // rdi
  __int64 v45; // r9
  __int64 v46; // rax
  __int64 k; // rdi
  __int64 v48; // rsi
  __int64 v49; // rax
  __int64 v50; // rax
  int v51; // eax
  __int8 v52; // al
  _BOOL4 v53; // eax
  __int64 *v54; // rax
  __int8 v55; // al
  __int8 v56; // al
  __int64 v57; // rsi
  __int64 v58; // r8
  __int64 v59; // r9
  __int64 v60; // rax
  _BOOL4 v61; // eax
  unsigned int v63; // [rsp+8h] [rbp-C8h]
  __int64 v64; // [rsp+8h] [rbp-C8h]
  __int64 v65; // [rsp+8h] [rbp-C8h]
  __int64 v66; // [rsp+8h] [rbp-C8h]
  __int64 v67; // [rsp+8h] [rbp-C8h]
  __int64 v68; // [rsp+8h] [rbp-C8h]
  __int64 v69; // [rsp+8h] [rbp-C8h]
  __int64 v70; // [rsp+8h] [rbp-C8h]
  _BOOL4 v72; // [rsp+1Ch] [rbp-B4h] BYREF
  int v73; // [rsp+20h] [rbp-B0h] BYREF
  unsigned int v74; // [rsp+24h] [rbp-ACh] BYREF
  __int64 v75; // [rsp+28h] [rbp-A8h] BYREF
  __m128i *v76; // [rsp+30h] [rbp-A0h] BYREF
  __int64 *v77; // [rsp+38h] [rbp-98h] BYREF
  __int64 v78[2]; // [rsp+40h] [rbp-90h] BYREF
  char v79; // [rsp+51h] [rbp-7Fh]
  __int64 v80; // [rsp+60h] [rbp-70h]
  __int64 v81; // [rsp+90h] [rbp-40h]
  _BYTE *v82; // [rsp+118h] [rbp+48h]
  _BYTE *v83; // [rsp+118h] [rbp+48h]
  _BYTE *v84; // [rsp+118h] [rbp+48h]
  _BYTE *v85; // [rsp+118h] [rbp+48h]
  _BYTE *v86; // [rsp+118h] [rbp+48h]

  v20 = a5;
  v21 = a15;
  v72 = a4;
  v73 = 0;
  v76 = 0;
  if ( !a15 )
    v21 = (FILE *)(a14 + 68);
  if ( unk_4F04C48 == -1 || (result = 0, *(char *)(qword_4F04C68[0] + 776LL * unk_4F04C48 + 12) >= 0) )
  {
    v23 = (__int64 *)&v76;
    if ( !a10 )
      v23 = 0;
    v24 = &v73;
    if ( !a14 )
      v24 = 0;
    v25 = sub_84AC10(a1, a2, a3, v72, a5, *a6, 0, 0, a7, a8, a9, a13, v21, a16, v24, 0, &v74, a18, v23, &v75);
    v30 = a1;
    v31 = (_BYTE *)a14;
    v32 = v25;
    *a20 = 0;
    v33 = v74;
    if ( v74 )
    {
      if ( a10 )
      {
        *(__m128i *)a19 = _mm_loadu_si128(v20);
        *(__m128i *)(a19 + 16) = _mm_loadu_si128(v20 + 1);
        *(__m128i *)(a19 + 32) = _mm_loadu_si128(v20 + 2);
        *(__m128i *)(a19 + 48) = _mm_loadu_si128(v20 + 3);
        *(__m128i *)(a19 + 64) = _mm_loadu_si128(v20 + 4);
        *(__m128i *)(a19 + 80) = _mm_loadu_si128(v20 + 5);
        *(__m128i *)(a19 + 96) = _mm_loadu_si128(v20 + 6);
        *(__m128i *)(a19 + 112) = _mm_loadu_si128(v20 + 7);
        *(__m128i *)(a19 + 128) = _mm_loadu_si128(v20 + 8);
        v55 = v20[1].m128i_i8[0];
        if ( v55 == 2 )
        {
          *(__m128i *)(a19 + 144) = _mm_loadu_si128(v20 + 9);
          *(__m128i *)(a19 + 160) = _mm_loadu_si128(v20 + 10);
          *(__m128i *)(a19 + 176) = _mm_loadu_si128(v20 + 11);
          *(__m128i *)(a19 + 192) = _mm_loadu_si128(v20 + 12);
          *(__m128i *)(a19 + 208) = _mm_loadu_si128(v20 + 13);
          *(__m128i *)(a19 + 224) = _mm_loadu_si128(v20 + 14);
          *(__m128i *)(a19 + 240) = _mm_loadu_si128(v20 + 15);
          *(__m128i *)(a19 + 256) = _mm_loadu_si128(v20 + 16);
          *(__m128i *)(a19 + 272) = _mm_loadu_si128(v20 + 17);
          *(__m128i *)(a19 + 288) = _mm_loadu_si128(v20 + 18);
          *(__m128i *)(a19 + 304) = _mm_loadu_si128(v20 + 19);
          *(__m128i *)(a19 + 320) = _mm_loadu_si128(v20 + 20);
          *(__m128i *)(a19 + 336) = _mm_loadu_si128(v20 + 21);
        }
        else if ( v55 == 5 || v55 == 1 )
        {
          *(_QWORD *)(a19 + 144) = v20[9].m128i_i64[0];
        }
        j = 0;
        sub_6F40C0(a19, a2, v26, v27, v28, v29);
        v33 = 1;
        v72 = 0;
      }
      else
      {
        if ( a14 )
        {
          sub_6F3AD0(
            a1,
            (*(_BYTE *)(a14 + 19) & 8) != 0,
            *(_QWORD *)(a14 + 104),
            (*(_BYTE *)(a14 + 18) & 0x40) != 0,
            a19);
          sub_6E4BC0(a19, a14);
          *(_BYTE *)(a19 + 18) &= 0xFCu;
          sub_6E5010((_BYTE *)a19, (_BYTE *)a14);
          sub_6E5070(a19, a14);
          v37 = a1;
        }
        else
        {
          sub_6F3AD0(a1, 0, 0, 0, a19);
          v37 = a1;
          *(_QWORD *)(a19 + 68) = *(_QWORD *)&v21->_flags;
          *(_QWORD *)(a19 + 76) = *(_QWORD *)&v21->_flags;
        }
        if ( v72 )
        {
          j = 0;
          sub_82F8F0(v20, (v20[1].m128i_i8[2] & 2) != 0, (__m128i *)a19, v34, v35, v36);
          v33 = 1;
          if ( !v73 )
          {
LABEL_15:
            v63 = v33;
            sub_84A700(v32, v74, (__int64)j, v72, (__int64)v20, *a6, v75, a20);
            return v63;
          }
          goto LABEL_28;
        }
        v69 = v37;
        if ( (unsigned int)sub_693580()
          && (*(_BYTE *)(v69 + 81) & 0x10) != 0
          && (unsigned int)sub_878150(v69)
          && (unsigned int)sub_830940(&v77, v78) )
        {
          sub_5F7420(v77, v21, 0, 1);
        }
        j = 0;
        v33 = 1;
      }
LABEL_27:
      if ( !v73 )
        goto LABEL_15;
LABEL_28:
      v42 = *(_BYTE *)(v32 + 80);
      if ( v42 == 16 )
      {
        v32 = **(_QWORD **)(v32 + 88);
        v42 = *(_BYTE *)(v32 + 80);
      }
      if ( v42 == 24 )
        v32 = *(_QWORD *)(v32 + 88);
      v43 = *(_QWORD *)(v32 + 88);
      for ( i = *(_QWORD *)(v43 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      sub_831320(i, v43, (__int64)v78);
      if ( a17 )
        v81 = *a17;
      sub_849040(*a6, (__int64)v78);
      *a20 = v80;
      return v79 == 0;
    }
    if ( !v32 )
    {
      j = v76;
      if ( v76 )
      {
        j = sub_73D790(*(_QWORD *)(*(_QWORD *)(v75 + 48) + 152LL));
        *(__m128i *)a19 = _mm_loadu_si128(v20);
        *(__m128i *)(a19 + 16) = _mm_loadu_si128(v20 + 1);
        *(__m128i *)(a19 + 32) = _mm_loadu_si128(v20 + 2);
        *(__m128i *)(a19 + 48) = _mm_loadu_si128(v20 + 3);
        *(__m128i *)(a19 + 64) = _mm_loadu_si128(v20 + 4);
        *(__m128i *)(a19 + 80) = _mm_loadu_si128(v20 + 5);
        *(__m128i *)(a19 + 96) = _mm_loadu_si128(v20 + 6);
        *(__m128i *)(a19 + 112) = _mm_loadu_si128(v20 + 7);
        *(__m128i *)(a19 + 128) = _mm_loadu_si128(v20 + 8);
        v56 = v20[1].m128i_i8[0];
        if ( v56 == 2 )
        {
          *(__m128i *)(a19 + 144) = _mm_loadu_si128(v20 + 9);
          *(__m128i *)(a19 + 160) = _mm_loadu_si128(v20 + 10);
          *(__m128i *)(a19 + 176) = _mm_loadu_si128(v20 + 11);
          *(__m128i *)(a19 + 192) = _mm_loadu_si128(v20 + 12);
          *(__m128i *)(a19 + 208) = _mm_loadu_si128(v20 + 13);
          *(__m128i *)(a19 + 224) = _mm_loadu_si128(v20 + 14);
          *(__m128i *)(a19 + 240) = _mm_loadu_si128(v20 + 15);
          *(__m128i *)(a19 + 256) = _mm_loadu_si128(v20 + 16);
          *(__m128i *)(a19 + 272) = _mm_loadu_si128(v20 + 17);
          *(__m128i *)(a19 + 288) = _mm_loadu_si128(v20 + 18);
          *(__m128i *)(a19 + 304) = _mm_loadu_si128(v20 + 19);
          *(__m128i *)(a19 + 320) = _mm_loadu_si128(v20 + 20);
          *(__m128i *)(a19 + 336) = _mm_loadu_si128(v20 + 21);
        }
        else if ( v56 == 5 || v56 == 1 )
        {
          *(_QWORD *)(a19 + 144) = v20[9].m128i_i64[0];
        }
        v57 = (__int64)j;
        sub_8449E0((_QWORD *)a19, j, v75 + 48, 0, 0);
        if ( (*(_BYTE *)(v75 + 64) & 4) != 0 )
        {
          if ( (unsigned int)sub_8D2E30(j) )
          {
            j = (__m128i *)sub_8D46C0(j);
            sub_6FA3A0((__m128i *)a19, v57);
          }
          else
          {
            sub_6F5FA0((const __m128i *)a19, 0, 0, 1u, v58, v59);
          }
        }
        else
        {
          j = (__m128i *)sub_8D46C0(j);
        }
        while ( j[8].m128i_i8[12] == 12 )
          j = (__m128i *)j[10].m128i_i64[0];
        v72 = 0;
        v33 = 1;
      }
      goto LABEL_27;
    }
    v39 = *(_BYTE *)(v32 + 80);
    v40 = v32;
    if ( v39 == 16 )
    {
      v40 = **(_QWORD **)(v32 + 88);
      v39 = *(_BYTE *)(v40 + 80);
    }
    if ( v39 == 24 )
      v40 = *(_QWORD *)(v40 + 88);
    for ( j = *(__m128i **)(*(_QWORD *)(v40 + 88) + 152LL); j[8].m128i_i8[12] == 12; j = (__m128i *)j[10].m128i_i64[0] )
      ;
    v64 = v40;
    v41 = *(_QWORD *)j[10].m128i_i64[1];
    if ( !v41 || (*(_BYTE *)(v41 + 35) & 1) == 0 )
    {
LABEL_26:
      sub_8310F0(v32, v30, v31, (__int64 *)&v21->_flags, (int *)&v72, (__int64)v20, a11, a12, (__m128i *)a19);
      v33 = 1;
      goto LABEL_27;
    }
    v45 = v72;
    if ( !v72 )
    {
      if ( !(unsigned int)sub_830940(&v77, v78) )
      {
        sub_6851C0(0xC8Eu, v21);
        return 0;
      }
      v46 = sub_8D46C0(v78[0]);
      v30 = a1;
      v31 = (_BYTE *)a14;
      for ( k = v46; *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
        ;
      v48 = *(_QWORD *)(v64 + 64);
      if ( v48 != k )
      {
        if ( !v48 || !dword_4F07588 || (v49 = *(_QWORD *)(k + 32), *(_QWORD *)(v48 + 32) != v49) || !v49 )
        {
          v50 = sub_8D5CE0(k, v48);
          v30 = a1;
          v31 = (_BYTE *)a14;
          if ( !v50 && dword_4F04C44 == -1 )
          {
            v60 = qword_4F04C68[0] + 776LL * dword_4F04C64;
            if ( (*(_BYTE *)(v60 + 6) & 6) == 0 && *(_BYTE *)(v60 + 4) != 12 )
              goto LABEL_48;
          }
        }
      }
      a2 = v30;
      v82 = v31;
      v65 = v30;
      v51 = sub_830D50(v32, v30, (__int64 *)&v21->_flags, 0, (__int64)v20);
      v30 = v65;
      v31 = v82;
      if ( !v51 )
      {
LABEL_48:
        v20 = 0;
        goto LABEL_26;
      }
      v20[1].m128i_i8[2] |= 2u;
      v72 = 1;
    }
    if ( (v20[1].m128i_i8[2] & 2) != 0 )
    {
      v52 = v20[1].m128i_i8[1];
      if ( v52 != 1 )
      {
LABEL_51:
        if ( v52 == 2 || (v83 = v31, v66 = v30, v53 = sub_6ED0A0((__int64)v20), v30 = v66, v31 = v83, v53) )
        {
          v84 = v31;
          v67 = v30;
          sub_6F9270(v20, a2, v40, v27, v28, v45);
          v30 = v67;
          v31 = v84;
        }
        goto LABEL_54;
      }
      v86 = v31;
      v70 = v30;
      v61 = sub_6ED0A0((__int64)v20);
      v30 = v70;
      v31 = v86;
      if ( v61 )
      {
        v52 = v20[1].m128i_i8[1];
        goto LABEL_51;
      }
      sub_6ECF90((__int64)v20, 0);
      v30 = v70;
      v31 = v86;
    }
LABEL_54:
    v85 = v31;
    v68 = v30;
    v54 = (__int64 *)sub_6E3060(v20);
    v31 = v85;
    v30 = v68;
    *v54 = (__int64)*a6;
    *a6 = v54;
    v72 = 0;
    goto LABEL_26;
  }
  return result;
}
