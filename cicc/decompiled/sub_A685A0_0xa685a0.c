// Function: sub_A685A0
// Address: 0xa685a0
//
__int64 __fastcall sub_A685A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, char a6, char a7)
{
  __int64 result; // rax
  _BYTE *v9; // r13
  _BYTE *v10; // r15
  __m128i *v11; // rdi
  __int64 (__fastcall *v12)(__int64); // rax
  __int64 v13; // rax
  __int64 v14; // rdi
  __int64 v15; // r13
  unsigned int v16; // esi
  int v17; // r10d
  __int64 v18; // r8
  _QWORD *v19; // r11
  unsigned int v20; // edi
  _QWORD *v21; // rcx
  __int64 v22; // rdx
  __m128i *v23; // r13
  __m128i *v24; // r15
  __m128i *v25; // rdi
  __int64 (__fastcall *v26)(__int64); // rax
  __int64 v27; // rdi
  int v28; // ecx
  int v29; // edx
  __int64 v30; // rax
  int v31; // ecx
  int v32; // ecx
  __int64 v33; // rdi
  unsigned int v34; // eax
  __int64 v35; // rsi
  int v36; // r10d
  _QWORD *v37; // r8
  int v38; // ecx
  int v39; // ecx
  __int64 v40; // rdi
  int v41; // r10d
  unsigned int v42; // eax
  __int64 v43; // rsi
  unsigned int v44; // [rsp+4h] [rbp-10Ch]
  __int64 v45; // [rsp+8h] [rbp-108h]
  __int64 v46; // [rsp+10h] [rbp-100h]
  __int64 v47; // [rsp+18h] [rbp-F8h]
  __int64 v48; // [rsp+20h] [rbp-F0h]
  __m128i v49; // [rsp+28h] [rbp-E8h]
  __m128i v50; // [rsp+40h] [rbp-D0h] BYREF
  __m128i v51; // [rsp+50h] [rbp-C0h]
  _BYTE v52[16]; // [rsp+60h] [rbp-B0h] BYREF
  __int64 (__fastcall *v53)(__int64 *); // [rsp+70h] [rbp-A0h]
  __int64 v54; // [rsp+78h] [rbp-98h]
  _BYTE v55[16]; // [rsp+80h] [rbp-90h] BYREF
  __int64 (__fastcall *v56)(_QWORD *); // [rsp+90h] [rbp-80h]
  __int64 v57; // [rsp+98h] [rbp-78h]
  __m128i v58; // [rsp+A0h] [rbp-70h] BYREF
  __m128i v59[2]; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v60; // [rsp+D0h] [rbp-40h]
  __int64 v61; // [rsp+D8h] [rbp-38h]

  *(_QWORD *)(a1 + 32) = a3;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = a4;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 40) = a4;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_DWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_DWORD *)(a1 + 104) = 0;
  *(_QWORD *)(a1 + 112) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  *(_DWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = 0;
  *(_QWORD *)(a1 + 152) = 0;
  *(_QWORD *)(a1 + 160) = 0;
  *(_DWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_BYTE *)(a1 + 200) = 0;
  *(_QWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 224) = 0;
  *(_DWORD *)(a1 + 232) = 0;
  *(_QWORD *)(a1 + 240) = 0;
  *(_QWORD *)(a1 + 248) = 0;
  *(_QWORD *)(a1 + 256) = 0;
  *(_QWORD *)(a1 + 264) = a5;
  *(_QWORD *)(a1 + 272) = 0;
  *(_QWORD *)(a1 + 280) = 0;
  *(_QWORD *)(a1 + 288) = 0;
  *(_DWORD *)(a1 + 296) = 0;
  v47 = a1 + 320;
  *(_QWORD *)(a1 + 304) = a1 + 320;
  *(_QWORD *)(a1 + 312) = 0;
  *(_BYTE *)(a1 + 320) = a6;
  *(_QWORD *)(a1 + 328) = 0;
  *(_BYTE *)(a1 + 321) = a7;
  *(_QWORD *)(a1 + 360) = a1 + 376;
  result = 0x800000000LL;
  *(_QWORD *)(a1 + 336) = 0;
  *(_QWORD *)(a1 + 344) = 0;
  *(_DWORD *)(a1 + 352) = 0;
  *(_QWORD *)(a1 + 368) = 0x800000000LL;
  *(_QWORD *)(a1 + 504) = a1 + 520;
  *(_QWORD *)(a1 + 512) = 0x800000000LL;
  *(_QWORD *)(a1 + 648) = 0;
  *(_QWORD *)(a1 + 656) = 0;
  *(_QWORD *)(a1 + 664) = 0;
  *(_DWORD *)(a1 + 672) = 0;
  *(_QWORD *)(a1 + 680) = 0;
  if ( a4 )
  {
    v46 = a1 + 272;
    sub_BA9640(&v58, a4);
    v49 = v59[1];
    v50 = _mm_loadu_si128(&v58);
    v51 = _mm_loadu_si128(v59);
    v45 = v60;
    v48 = v61;
    while ( 1 )
    {
      if ( *(_OWORD *)&v50 == *(_OWORD *)&v49 && v51.m128i_i64[1] == v48 )
      {
        result = v45;
        if ( v51.m128i_i64[0] == v45 )
          return result;
      }
      v9 = v52;
      v54 = 0;
      v10 = v52;
      v11 = &v50;
      v53 = sub_A4F570;
      v12 = sub_A4F550;
      if ( ((unsigned __int8)sub_A4F550 & 1) != 0 )
        goto LABEL_5;
LABEL_6:
      v13 = v12((__int64)v11);
      if ( !v13 )
        break;
LABEL_10:
      v15 = *(_QWORD *)(v13 + 48);
      if ( !v15 )
        goto LABEL_13;
      v16 = *(_DWORD *)(a1 + 296);
      if ( v16 )
      {
        v17 = 1;
        v18 = *(_QWORD *)(a1 + 280);
        v19 = 0;
        v20 = (v16 - 1) & (((unsigned int)v15 >> 4) ^ ((unsigned int)v15 >> 9));
        v21 = (_QWORD *)(v18 + 8LL * v20);
        v22 = *v21;
        if ( v15 == *v21 )
          goto LABEL_13;
        while ( v22 != -4096 )
        {
          if ( v22 != -8192 || v19 )
            v21 = v19;
          v20 = (v16 - 1) & (v17 + v20);
          v22 = *(_QWORD *)(v18 + 8LL * v20);
          if ( v15 == v22 )
            goto LABEL_13;
          ++v17;
          v19 = v21;
          v21 = (_QWORD *)(v18 + 8LL * v20);
        }
        if ( !v19 )
          v19 = v21;
        v28 = *(_DWORD *)(a1 + 288);
        ++*(_QWORD *)(a1 + 272);
        v29 = v28 + 1;
        if ( 4 * (v28 + 1) < 3 * v16 )
        {
          if ( v16 - *(_DWORD *)(a1 + 292) - v29 > v16 >> 3 )
            goto LABEL_31;
          v44 = ((unsigned int)v15 >> 4) ^ ((unsigned int)v15 >> 9);
          sub_A683D0(v46, v16);
          v38 = *(_DWORD *)(a1 + 296);
          if ( !v38 )
          {
LABEL_58:
            ++*(_DWORD *)(a1 + 288);
            BUG();
          }
          v39 = v38 - 1;
          v40 = *(_QWORD *)(a1 + 280);
          v37 = 0;
          v41 = 1;
          v42 = v39 & v44;
          v19 = (_QWORD *)(v40 + 8LL * (v39 & v44));
          v43 = *v19;
          v29 = *(_DWORD *)(a1 + 288) + 1;
          if ( v15 == *v19 )
            goto LABEL_31;
          while ( v43 != -4096 )
          {
            if ( v43 == -8192 && !v37 )
              v37 = v19;
            v42 = v39 & (v41 + v42);
            v19 = (_QWORD *)(v40 + 8LL * v42);
            v43 = *v19;
            if ( v15 == *v19 )
              goto LABEL_31;
            ++v41;
          }
          goto LABEL_41;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 272);
      }
      sub_A683D0(v46, 2 * v16);
      v31 = *(_DWORD *)(a1 + 296);
      if ( !v31 )
        goto LABEL_58;
      v32 = v31 - 1;
      v33 = *(_QWORD *)(a1 + 280);
      v34 = v32 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v19 = (_QWORD *)(v33 + 8LL * v34);
      v35 = *v19;
      v29 = *(_DWORD *)(a1 + 288) + 1;
      if ( *v19 == v15 )
        goto LABEL_31;
      v36 = 1;
      v37 = 0;
      while ( v35 != -4096 )
      {
        if ( !v37 && v35 == -8192 )
          v37 = v19;
        v34 = v32 & (v36 + v34);
        v19 = (_QWORD *)(v33 + 8LL * v34);
        v35 = *v19;
        if ( v15 == *v19 )
          goto LABEL_31;
        ++v36;
      }
LABEL_41:
      if ( v37 )
        v19 = v37;
LABEL_31:
      *(_DWORD *)(a1 + 288) = v29;
      if ( *v19 != -4096 )
        --*(_DWORD *)(a1 + 292);
      *v19 = v15;
      v30 = *(unsigned int *)(a1 + 312);
      if ( v30 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 316) )
      {
        sub_C8D5F0(a1 + 304, v47, v30 + 1, 8);
        v30 = *(unsigned int *)(a1 + 312);
      }
      *(_QWORD *)(*(_QWORD *)(a1 + 304) + 8 * v30) = v15;
      ++*(_DWORD *)(a1 + 312);
LABEL_13:
      v23 = (__m128i *)v55;
      v57 = 0;
      v24 = (__m128i *)v55;
      v25 = &v50;
      v56 = sub_A4F520;
      v26 = sub_A4F4F0;
      if ( ((unsigned __int8)sub_A4F4F0 & 1) == 0 )
        goto LABEL_15;
LABEL_14:
      v26 = *(__int64 (__fastcall **)(__int64))((char *)v26 + v25->m128i_i64[0] - 1);
LABEL_15:
      while ( !(unsigned __int8)v26((__int64)v25) )
      {
        if ( &v58 == ++v23 )
          goto LABEL_59;
        v27 = v24[1].m128i_i64[1];
        v26 = (__int64 (__fastcall *)(__int64))v24[1].m128i_i64[0];
        v24 = v23;
        v25 = (__m128i *)((char *)&v50 + v27);
        if ( ((unsigned __int8)v26 & 1) != 0 )
          goto LABEL_14;
      }
    }
    while ( 1 )
    {
      v9 += 16;
      if ( v55 == v9 )
LABEL_59:
        BUG();
      v14 = *((_QWORD *)v10 + 3);
      v12 = (__int64 (__fastcall *)(__int64))*((_QWORD *)v10 + 2);
      v10 = v9;
      v11 = (__m128i *)((char *)&v50 + v14);
      if ( ((unsigned __int8)v12 & 1) != 0 )
        break;
      v13 = v12((__int64)v11);
      if ( v13 )
        goto LABEL_10;
    }
LABEL_5:
    v12 = *(__int64 (__fastcall **)(__int64))((char *)v12 + v11->m128i_i64[0] - 1);
    goto LABEL_6;
  }
  return result;
}
