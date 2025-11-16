// Function: sub_337F9F0
// Address: 0x337f9f0
//
char __fastcall sub_337F9F0(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4, __int64 a5, int a6, __int64 *a7)
{
  __int64 v8; // r14
  __int64 (*v9)(void); // rdx
  __int64 v10; // rax
  __int64 v11; // r13
  char result; // al
  int v13; // edx
  unsigned __int8 v14; // cl
  __int64 v15; // rax
  int v16; // eax
  char v17; // r13
  __int64 v18; // rax
  char v19; // cl
  __int64 v20; // rsi
  int v21; // edx
  __int64 v22; // rdx
  __int64 v23; // rsi
  __int64 v24; // rdx
  unsigned int v25; // ecx
  __int64 *v26; // r15
  __int64 v27; // r9
  __int64 v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rax
  _DWORD *v31; // rcx
  _DWORD *v32; // rax
  int v33; // edx
  _QWORD *v34; // rdx
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rdx
  __int64 v38; // r13
  __int64 v39; // rbx
  __int64 v40; // rax
  unsigned int v41; // eax
  unsigned int v42; // esi
  __int64 v43; // r10
  __int64 v44; // r9
  __int64 *v45; // r9
  __int64 v46; // rax
  __int64 v47; // r13
  __int64 v48; // rdx
  unsigned int v49; // edx
  unsigned __int64 v50; // rcx
  unsigned int v51; // eax
  int v52; // edx
  __int32 v53; // eax
  int v54; // edx
  int v55; // eax
  int v56; // edx
  int v57; // r10d
  unsigned __int64 v58; // r11
  __int64 v59; // [rsp+10h] [rbp-310h]
  __int64 v60; // [rsp+10h] [rbp-310h]
  __int64 v61; // [rsp+10h] [rbp-310h]
  __int64 v62; // [rsp+10h] [rbp-310h]
  __int64 v63; // [rsp+18h] [rbp-308h]
  __int64 v64; // [rsp+18h] [rbp-308h]
  __int64 v65; // [rsp+18h] [rbp-308h]
  __int64 v66; // [rsp+18h] [rbp-308h]
  char v67; // [rsp+20h] [rbp-300h]
  int v68; // [rsp+20h] [rbp-300h]
  char v69; // [rsp+20h] [rbp-300h]
  int v70; // [rsp+20h] [rbp-300h]
  int v71; // [rsp+20h] [rbp-300h]
  unsigned __int64 v72; // [rsp+20h] [rbp-300h]
  int v73; // [rsp+2Ch] [rbp-2F4h] BYREF
  __int64 v74; // [rsp+30h] [rbp-2F0h] BYREF
  _QWORD *v75; // [rsp+38h] [rbp-2E8h] BYREF
  __int64 v76; // [rsp+40h] [rbp-2E0h] BYREF
  __int64 v77[2]; // [rsp+48h] [rbp-2D8h] BYREF
  __int64 v78; // [rsp+58h] [rbp-2C8h] BYREF
  __int64 v79[4]; // [rsp+60h] [rbp-2C0h] BYREF
  __m128i v80; // [rsp+80h] [rbp-2A0h] BYREF
  __int128 v81; // [rsp+90h] [rbp-290h]
  __int128 v82; // [rsp+A0h] [rbp-280h]
  __int64 *v83[8]; // [rsp+B0h] [rbp-270h] BYREF
  char *v84; // [rsp+F0h] [rbp-230h] BYREF
  unsigned int v85; // [rsp+F8h] [rbp-228h]
  char v86; // [rsp+100h] [rbp-220h] BYREF
  unsigned __int8 *v87[2]; // [rsp+160h] [rbp-1C0h] BYREF
  _BYTE v88[64]; // [rsp+170h] [rbp-1B0h] BYREF
  char *v89; // [rsp+1B0h] [rbp-170h]
  char v90; // [rsp+1C8h] [rbp-158h] BYREF
  _BYTE *v91; // [rsp+1D0h] [rbp-150h]
  _BYTE v92[16]; // [rsp+1E0h] [rbp-140h] BYREF
  _BYTE *v93; // [rsp+1F0h] [rbp-130h]
  int v94; // [rsp+1F8h] [rbp-128h]
  _BYTE v95[32]; // [rsp+200h] [rbp-120h] BYREF
  _BYTE *v96; // [rsp+220h] [rbp-100h] BYREF
  __int64 v97; // [rsp+228h] [rbp-F8h]
  _BYTE v98[240]; // [rsp+230h] [rbp-F0h] BYREF

  v77[0] = a2;
  v76 = a3;
  v75 = a4;
  v74 = a5;
  v73 = a6;
  if ( *(_BYTE *)a2 != 22 )
    return 0;
  v8 = *(_QWORD *)(*(_QWORD *)(a1 + 864) + 40LL);
  v9 = *(__int64 (**)(void))(**(_QWORD **)(v8 + 16) + 128LL);
  v10 = 0;
  if ( v9 != sub_2DAC790 )
    v10 = v9();
  v78 = v10;
  v79[0] = v8;
  v11 = *(_QWORD *)(a1 + 960);
  v79[1] = (__int64)&v78;
  v79[2] = (__int64)&v74;
  v79[3] = (__int64)&v76;
  if ( v73 )
    goto LABEL_14;
  if ( *(_QWORD *)(v11 + 744) != *(_QWORD *)(*(_QWORD *)(v11 + 8) + 328LL) )
    return 0;
  v13 = *(_DWORD *)(a1 + 848);
  if ( !*(_WORD *)(v76 + 20) )
    goto LABEL_13;
  v14 = *(_BYTE *)(v74 - 16);
  if ( (v14 & 2) != 0 )
  {
    if ( *(_DWORD *)(v74 - 24) == 2 )
    {
      v15 = *(_QWORD *)(v74 - 32);
      goto LABEL_12;
    }
  }
  else if ( ((*(_WORD *)(v74 - 16) >> 6) & 0xF) == 2 )
  {
    v15 = v74 - 16 - 8LL * ((v14 >> 2) & 0xF);
LABEL_12:
    if ( *(_QWORD *)(v15 + 8) )
    {
LABEL_13:
      if ( v13 != 1 )
        return 0;
      goto LABEL_14;
    }
  }
  v41 = *(_DWORD *)(a2 + 32);
  v42 = *(_DWORD *)(v11 + 456);
  v43 = 1LL << *(_DWORD *)(a2 + 32);
  v44 = 8LL * (v41 >> 6);
  if ( v41 >= v42 )
  {
    v49 = v41 + 1;
    if ( (v42 & 0x3F) != 0 )
      *(_QWORD *)(*(_QWORD *)(v11 + 392) + 8LL * *(unsigned int *)(v11 + 400) - 8) &= ~(-1LL << (v42 & 0x3F));
    v50 = *(unsigned int *)(v11 + 400);
    *(_DWORD *)(v11 + 456) = v49;
    v51 = (v41 + 64) >> 6;
    if ( v51 != v50 )
    {
      if ( v51 >= v50 )
      {
        v58 = v51 - v50;
        if ( v51 > (unsigned __int64)*(unsigned int *)(v11 + 404) )
        {
          v62 = v43;
          v66 = v44;
          v72 = v51 - v50;
          sub_C8D5F0(v11 + 392, (const void *)(v11 + 408), v51, 8u, (__int64)&v76, v44);
          v50 = *(unsigned int *)(v11 + 400);
          v43 = v62;
          v44 = v66;
          v58 = v72;
        }
        if ( 8 * v58 )
        {
          v61 = v43;
          v65 = v44;
          v71 = v58;
          memset((void *)(*(_QWORD *)(v11 + 392) + 8 * v50), 0, 8 * v58);
          LODWORD(v50) = *(_DWORD *)(v11 + 400);
          LODWORD(v58) = v71;
          v44 = v65;
          v43 = v61;
        }
        v49 = *(_DWORD *)(v11 + 456);
        *(_DWORD *)(v11 + 400) = v58 + v50;
      }
      else
      {
        *(_DWORD *)(v11 + 400) = v51;
      }
    }
    v52 = v49 & 0x3F;
    if ( v52 )
      *(_QWORD *)(*(_QWORD *)(v11 + 392) + 8LL * *(unsigned int *)(v11 + 400) - 8) &= ~(-1LL << v52);
    v45 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(a1 + 960) + 392LL) + v44);
    v46 = *v45;
  }
  else
  {
    v45 = (__int64 *)(*(_QWORD *)(v11 + 392) + v44);
    v46 = *v45;
    if ( v13 != 1 && (v43 & v46) != 0 )
      return *sub_337DC20(a1 + 8, v77) == 0;
  }
  *v45 = v43 | v46;
  v11 = *(_QWORD *)(a1 + 960);
LABEL_14:
  v80 = 0;
  v81 = 0;
  v82 = 0;
  v16 = sub_374D370(v11, a2);
  if ( v16 != 0x7FFFFFFF )
  {
    v19 = BYTE8(v82);
    *(_QWORD *)&v81 = 0;
    DWORD2(v81) = v16;
    v80.m128i_i32[0] = v80.m128i_i32[0] & 0xFFF00000 | 5;
    if ( BYTE8(v82) )
      v19 = 0;
    else
      BYTE8(v82) = 1;
    v96 = v98;
    v97 = 0x800000000LL;
    goto LABEL_44;
  }
  result = BYTE8(v82);
  v96 = v98;
  v97 = 0x800000000LL;
  v17 = BYTE8(v82);
  if ( BYTE8(v82) )
    goto LABEL_106;
  if ( !*a7 )
    goto LABEL_24;
  sub_3367F80((__int64)&v96, a7);
  if ( (_DWORD)v97 == 1 )
  {
    v54 = *(_DWORD *)v96;
    if ( *(_DWORD *)v96 )
    {
      if ( v54 < 0 )
      {
        v70 = *(_DWORD *)v96;
        v55 = sub_2EBF1A0(*(_QWORD *)(v8 + 32), v54);
        v54 = v70;
        if ( v55 )
          v54 = v55;
      }
      v80.m128i_i64[0] &= 0xFFFFFFF000000000LL;
      v80.m128i_i32[2] = v54;
      v81 = 0u;
      v19 = v73 != 0;
      *(_QWORD *)&v82 = 0;
      if ( !BYTE8(v82) )
        BYTE8(v82) = 1;
      goto LABEL_44;
    }
  }
  result = BYTE8(v82);
  if ( BYTE8(v82) )
  {
LABEL_106:
    v19 = 0;
    goto LABEL_44;
  }
  if ( *a7 )
  {
    v67 = BYTE8(v82);
    v18 = sub_33CF5B0(*a7, a7[1]);
    v19 = v67;
    if ( *(_DWORD *)(v18 + 24) == 298 )
    {
      v20 = *(_QWORD *)(*(_QWORD *)(v18 + 40) + 40LL);
      result = BYTE8(v82);
      v21 = *(_DWORD *)(v20 + 24);
      if ( v21 == 15 || v21 == 39 )
      {
        v56 = *(_DWORD *)(v20 + 96);
        if ( !BYTE8(v82) )
        {
          BYTE8(v82) = 1;
          DWORD2(v81) = v56;
          v34 = v75;
          v80.m128i_i32[0] = v80.m128i_i32[0] & 0xFFF00000 | 5;
          *(_QWORD *)&v81 = 0;
          goto LABEL_57;
        }
        DWORD2(v81) = *(_DWORD *)(v20 + 96);
        *(_QWORD *)&v81 = 0;
        v80.m128i_i32[0] = v80.m128i_i32[0] & 0xFFF00000 | 5;
LABEL_44:
        v34 = v75;
        if ( !v80.m128i_i8[0] )
        {
          sub_3367790(v79, v80.m128i_i32[2], v75, v19);
          v38 = v37;
LABEL_46:
          v39 = *(_QWORD *)(a1 + 960);
          v40 = *(unsigned int *)(v39 + 320);
          if ( v40 + 1 > (unsigned __int64)*(unsigned int *)(v39 + 324) )
          {
            sub_C8D5F0(v39 + 312, (const void *)(v39 + 328), v40 + 1, 8u, v35, v36);
            v40 = *(unsigned int *)(v39 + 320);
          }
          *(_QWORD *)(*(_QWORD *)(v39 + 312) + 8 * v40) = v38;
          ++*(_DWORD *)(v39 + 320);
          goto LABEL_49;
        }
LABEL_57:
        v60 = (__int64)v34;
        v47 = *(_QWORD *)(v78 + 8);
        v64 = v76;
        sub_B10CB0(v87, v74);
        sub_2E908B0((_QWORD *)v8, v87, (_WORD *)(v47 - 560), 1, &v80, 1, v64, v60);
        v38 = v48;
        if ( v87[0] )
          sub_B91220((__int64)v87, (__int64)v87[0]);
        goto LABEL_46;
      }
    }
    else
    {
      result = BYTE8(v82);
    }
    if ( result )
      goto LABEL_44;
  }
LABEL_24:
  v83[1] = (__int64 *)a1;
  v83[0] = (__int64 *)&v75;
  v83[3] = v77;
  v83[5] = v79;
  v83[6] = (__int64 *)&v73;
  v22 = *(_QWORD *)(a1 + 960);
  v83[2] = &v76;
  v23 = *(_QWORD *)(v22 + 128);
  v24 = *(unsigned int *)(v22 + 144);
  v83[4] = &v74;
  if ( (_DWORD)v24 )
  {
    v25 = (v24 - 1) & ((LODWORD(v77[0]) >> 9) ^ (LODWORD(v77[0]) >> 4));
    v26 = (__int64 *)(v23 + 16LL * v25);
    v27 = *v26;
    if ( v77[0] == *v26 )
    {
LABEL_26:
      if ( v26 != (__int64 *)(v23 + 16 * v24) )
      {
        v28 = *(_QWORD *)(a1 + 864);
        v29 = *(_QWORD *)(v28 + 16);
        BYTE4(v84) = 0;
        v59 = v29;
        v63 = *(_QWORD *)(v77[0] + 8);
        v68 = sub_2E79000(*(__int64 **)(v28 + 40));
        v30 = sub_BD5C60(v77[0]);
        sub_336FEE0((__int64)v87, v30, v59, v68, *((_DWORD *)v26 + 2), v63, (__int64)v84);
        v31 = &v93[4 * v94];
        if ( v93 != (_BYTE *)v31 )
        {
          v32 = v93;
          v33 = 0;
          do
            v33 += *v32++;
          while ( v31 != v32 );
          if ( v33 > 1 )
          {
            sub_3372B70((__int64)&v84, (__int64)v87);
            sub_33679D0(v83, (__int64)v84, v85);
            if ( v84 != &v86 )
              _libc_free((unsigned __int64)v84);
            if ( v93 != v95 )
              _libc_free((unsigned __int64)v93);
            if ( v91 != v92 )
              _libc_free((unsigned __int64)v91);
            if ( v89 != &v90 )
              _libc_free((unsigned __int64)v89);
            if ( v87[0] != v88 )
              _libc_free((unsigned __int64)v87[0]);
            goto LABEL_49;
          }
        }
        v53 = *((_DWORD *)v26 + 2);
        v80.m128i_i64[0] &= 0xFFFFFFF000000000LL;
        v80.m128i_i32[2] = v53;
        v81 = 0u;
        *(_QWORD *)&v82 = 0;
        if ( !BYTE8(v82) )
          BYTE8(v82) = 1;
        v17 = v73 != 0;
        if ( v93 != v95 )
          _libc_free((unsigned __int64)v93);
        if ( v91 != v92 )
          _libc_free((unsigned __int64)v91);
        if ( v89 != &v90 )
          _libc_free((unsigned __int64)v89);
        if ( v87[0] != v88 )
          _libc_free((unsigned __int64)v87[0]);
        result = BYTE8(v82);
        goto LABEL_80;
      }
    }
    else
    {
      v57 = 1;
      while ( v27 != -4096 )
      {
        v25 = (v24 - 1) & (v57 + v25);
        v26 = (__int64 *)(v23 + 16LL * v25);
        v27 = *v26;
        if ( v77[0] == *v26 )
          goto LABEL_26;
        ++v57;
      }
    }
  }
  if ( (unsigned int)v97 > 1uLL )
  {
    sub_33679D0(v83, (__int64)v96, (unsigned int)v97);
LABEL_49:
    result = 1;
    goto LABEL_50;
  }
LABEL_80:
  if ( result )
  {
    v19 = v17;
    goto LABEL_44;
  }
LABEL_50:
  if ( v96 != v98 )
  {
    v69 = result;
    _libc_free((unsigned __int64)v96);
    return v69;
  }
  return result;
}
