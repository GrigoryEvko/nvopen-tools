// Function: sub_327C770
// Address: 0x327c770
//
__int64 __fastcall sub_327C770(__int64 *a1, __int64 a2)
{
  const __m128i *v2; // rax
  __int64 v3; // r15
  __int64 v5; // rsi
  __m128i v7; // xmm0
  __int64 v8; // r13
  unsigned __int16 v9; // ax
  __int64 v10; // rdi
  __int64 v11; // rsi
  __int64 *v12; // rsi
  __int64 v13; // rsi
  int v14; // r15d
  int v15; // edx
  int v16; // r9d
  __int64 result; // rax
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 *v20; // rsi
  __int64 v21; // rsi
  int v22; // edx
  __int64 v23; // r14
  __int64 (*v24)(); // rax
  __int64 *v25; // rsi
  __int64 v26; // r11
  int v27; // edx
  __int64 v28; // r11
  int v29; // edx
  int v30; // ecx
  int v31; // edx
  __int64 v32; // rax
  __int64 v33; // rdi
  int v34; // r9d
  __int16 v35; // r10
  __int128 v36; // rax
  __int64 v37; // rdi
  unsigned int v38; // edx
  __int64 v39; // rsi
  char v40; // bl
  __int128 v41; // rax
  int v42; // r9d
  __int16 v43; // [rsp+Ch] [rbp-124h]
  int v44; // [rsp+10h] [rbp-120h]
  int v45; // [rsp+10h] [rbp-120h]
  int v46; // [rsp+18h] [rbp-118h]
  int v47; // [rsp+18h] [rbp-118h]
  int v48; // [rsp+20h] [rbp-110h]
  int v49; // [rsp+20h] [rbp-110h]
  __int64 v50; // [rsp+20h] [rbp-110h]
  __int64 v51; // [rsp+28h] [rbp-108h]
  int v52; // [rsp+28h] [rbp-108h]
  int v53; // [rsp+28h] [rbp-108h]
  __int64 v54; // [rsp+28h] [rbp-108h]
  __int64 v55; // [rsp+28h] [rbp-108h]
  int v56; // [rsp+28h] [rbp-108h]
  __int64 v57; // [rsp+30h] [rbp-100h]
  __int128 v58; // [rsp+30h] [rbp-100h]
  __int64 v59; // [rsp+40h] [rbp-F0h]
  __int64 v60; // [rsp+40h] [rbp-F0h]
  unsigned __int64 v61; // [rsp+48h] [rbp-E8h]
  __int64 v62; // [rsp+A0h] [rbp-90h] BYREF
  int v63; // [rsp+A8h] [rbp-88h]
  __int64 v64; // [rsp+B0h] [rbp-80h]
  __int64 v65; // [rsp+B8h] [rbp-78h]
  __int128 v66; // [rsp+C0h] [rbp-70h] BYREF
  __int64 v67; // [rsp+D0h] [rbp-60h]
  _OWORD v68[5]; // [rsp+E0h] [rbp-50h] BYREF

  v2 = *(const __m128i **)(a2 + 40);
  v3 = v2[2].m128i_i64[1];
  if ( *(_DWORD *)(v3 + 24) == 36
    || *(_DWORD *)(a2 + 24) != 299
    || (*(_BYTE *)(a2 + 33) & 4) != 0
    || (*(_WORD *)(a2 + 32) & 0x380) != 0 )
  {
    return 0;
  }
  v5 = *(_QWORD *)(a2 + 80);
  v62 = v5;
  if ( v5 )
  {
    sub_B96E90((__int64)&v62, v5, 1);
    v2 = *(const __m128i **)(a2 + 40);
  }
  v7 = _mm_loadu_si128(v2 + 5);
  v63 = *(_DWORD *)(a2 + 72);
  v8 = v2->m128i_i64[0];
  v57 = v2->m128i_i64[1];
  v9 = **(_WORD **)(v3 + 48);
  if ( v9 == 12 )
  {
    if ( !*((_BYTE *)a1 + 34) || *(_QWORD *)(a1[1] + 168) )
    {
      if ( !*((_BYTE *)a1 + 33)
        && (*(_BYTE *)(*(_QWORD *)(a2 + 112) + 37LL) & 0xF) == 0
        && (*(_BYTE *)(a2 + 32) & 8) == 0
        || (v18 = a1[1], *(_QWORD *)(v18 + 168)) && (*(_BYTE *)(v18 + 10213) & 0xFB) == 0 )
      {
        v19 = *(_QWORD *)(v3 + 80);
        v54 = *a1;
        *(_QWORD *)&v68[0] = v19;
        if ( v19 )
          sub_B96E90((__int64)v68, v19, 1);
        DWORD2(v68[0]) = *(_DWORD *)(v3 + 72);
        v20 = (__int64 *)(*(_QWORD *)(v3 + 96) + 24LL);
        if ( (void *)*v20 == sub_C33340() )
          sub_C3E660((__int64)&v66, (__int64)v20);
        else
          sub_C3A850((__int64)&v66, v20);
        LODWORD(v21) = v66;
        if ( DWORD2(v66) > 0x40 )
          v21 = *(_QWORD *)v66;
        v14 = sub_3400BD0(v54, v21, (unsigned int)v68, 7, 0, 0, 0);
        v16 = v22;
        if ( DWORD2(v66) <= 0x40 )
          goto LABEL_23;
        goto LABEL_21;
      }
    }
  }
  else if ( v9 <= 0xCu )
  {
    if ( (unsigned __int16)(v9 - 10) > 1u )
      goto LABEL_76;
  }
  else
  {
    if ( v9 == 13 )
    {
      v10 = a1[1];
      if ( !*(_QWORD *)(v10 + 176)
        || (*((_BYTE *)a1 + 33)
         || (*(_BYTE *)(*(_QWORD *)(a2 + 112) + 37LL) & 0xF) != 0
         || (*(_BYTE *)(a2 + 32) & 8) != 0)
        && (*(_BYTE *)(v10 + 10713) & 0xFB) != 0 )
      {
        if ( (*(_BYTE *)(*(_QWORD *)(a2 + 112) + 37LL) & 0xF) == 0
          && (*(_BYTE *)(a2 + 32) & 8) == 0
          && *(_QWORD *)(v10 + 168)
          && (*(_BYTE *)(v10 + 10213) & 0xFB) == 0 )
        {
          v23 = *(_QWORD *)(v3 + 96);
          v24 = *(__int64 (**)())(*(_QWORD *)v10 + 616LL);
          v25 = (__int64 *)(v23 + 24);
          if ( v24 == sub_2FE3170 )
          {
LABEL_55:
            if ( *(void **)(v23 + 24) == sub_C33340() )
              sub_C3E660((__int64)v68, (__int64)v25);
            else
              sub_C3A850((__int64)v68, v25);
            if ( DWORD2(v68[0]) <= 0x40 )
            {
              v55 = *(_QWORD *)&v68[0];
            }
            else
            {
              v55 = **(_QWORD **)&v68[0];
              j_j___libc_free_0_0(*(unsigned __int64 *)&v68[0]);
            }
            v26 = *a1;
            *(_QWORD *)&v68[0] = *(_QWORD *)(v3 + 80);
            if ( *(_QWORD *)&v68[0] )
            {
              v48 = v26;
              sub_325F5D0((__int64 *)v68);
              LODWORD(v26) = v48;
            }
            DWORD2(v68[0]) = *(_DWORD *)(v3 + 72);
            v49 = sub_3400BD0(v26, v55, (unsigned int)v68, 7, 0, 0, 0);
            v46 = v27;
            if ( *(_QWORD *)&v68[0] )
              sub_B91220((__int64)v68, *(__int64 *)&v68[0]);
            v28 = *a1;
            *(_QWORD *)&v68[0] = *(_QWORD *)(v3 + 80);
            if ( *(_QWORD *)&v68[0] )
            {
              v44 = v28;
              sub_325F5D0((__int64 *)v68);
              LODWORD(v28) = v44;
            }
            DWORD2(v68[0]) = *(_DWORD *)(v3 + 72);
            v56 = sub_3400BD0(v28, HIDWORD(v55), (unsigned int)v68, 7, 0, 0, 0);
            v45 = v29;
            if ( *(_QWORD *)&v68[0] )
              sub_B91220((__int64)v68, *(__int64 *)&v68[0]);
            if ( *(_BYTE *)sub_2E79000(*(__int64 **)(*a1 + 40)) )
            {
              v30 = v49;
              v49 = v56;
              v31 = v46;
              v46 = v45;
              v56 = v30;
              v45 = v31;
            }
            v32 = *(_QWORD *)(a2 + 112);
            v33 = *a1;
            v34 = v46;
            v35 = *(_WORD *)(v32 + 32);
            v47 = v57;
            v68[0] = _mm_loadu_si128((const __m128i *)(v32 + 40));
            v43 = v35;
            v68[1] = _mm_loadu_si128((const __m128i *)(v32 + 56));
            *(_QWORD *)&v36 = sub_33F4560(
                                v33,
                                v8,
                                v57,
                                (unsigned int)&v62,
                                v49,
                                v34,
                                v7.m128i_i64[0],
                                v7.m128i_i64[1],
                                *(_OWORD *)v32,
                                *(_QWORD *)(v32 + 16),
                                *(_BYTE *)(v32 + 34),
                                v35,
                                (__int64)v68);
            v37 = *a1;
            LOBYTE(v65) = 0;
            v58 = v36;
            v64 = 4;
            v60 = sub_3409320(v37, v7.m128i_i32[0], v7.m128i_i32[2], 4, v65, (unsigned int)&v62, 0);
            v39 = *(_QWORD *)(a2 + 112);
            v61 = v38 | v7.m128i_i64[1] & 0xFFFFFFFF00000000LL;
            v40 = *(_BYTE *)(v39 + 34);
            v50 = *a1;
            sub_327C6E0((__int64)&v66, (__int64 *)v39, 4);
            *(_QWORD *)&v41 = sub_33F4560(
                                v50,
                                v8,
                                v47,
                                (unsigned int)&v62,
                                v56,
                                v45,
                                v60,
                                v61,
                                v66,
                                v67,
                                v40,
                                v43,
                                (__int64)v68);
            result = sub_3406EB0(*a1, 2, (unsigned int)&v62, 1, 0, v42, v58, v41);
            goto LABEL_30;
          }
          if ( !((unsigned __int8 (__fastcall *)(__int64, __int64 *, __int64, _QWORD, _QWORD))v24)(v10, v25, 13, 0, 0) )
          {
            v23 = *(_QWORD *)(v3 + 96);
            v25 = (__int64 *)(v23 + 24);
            goto LABEL_55;
          }
        }
        goto LABEL_29;
      }
      v11 = *(_QWORD *)(v3 + 80);
      v51 = *a1;
      *(_QWORD *)&v68[0] = v11;
      if ( v11 )
        sub_B96E90((__int64)v68, v11, 1);
      DWORD2(v68[0]) = *(_DWORD *)(v3 + 72);
      v12 = (__int64 *)(*(_QWORD *)(v3 + 96) + 24LL);
      if ( (void *)*v12 == sub_C33340() )
        sub_C3E660((__int64)&v66, (__int64)v12);
      else
        sub_C3A850((__int64)&v66, v12);
      LODWORD(v13) = v66;
      if ( DWORD2(v66) > 0x40 )
        v13 = *(_QWORD *)v66;
      v14 = sub_3400BD0(v51, v13, (unsigned int)v68, 8, 0, 0, 0);
      v16 = v15;
      if ( DWORD2(v66) <= 0x40 )
      {
LABEL_23:
        if ( *(_QWORD *)&v68[0] )
        {
          v53 = v16;
          sub_B91220((__int64)v68, *(__int64 *)&v68[0]);
          v16 = v53;
        }
        result = sub_33F3F90(
                   *a1,
                   v8,
                   v57,
                   (unsigned int)&v62,
                   v14,
                   v16,
                   v7.m128i_i64[0],
                   v7.m128i_i64[1],
                   *(_QWORD *)(a2 + 112));
        goto LABEL_30;
      }
LABEL_21:
      if ( (_QWORD)v66 )
      {
        v52 = v16;
        j_j___libc_free_0_0(v66);
        v16 = v52;
      }
      goto LABEL_23;
    }
    if ( (unsigned __int16)(v9 - 14) > 2u )
LABEL_76:
      BUG();
  }
LABEL_29:
  result = 0;
LABEL_30:
  if ( v62 )
  {
    v59 = result;
    sub_B91220((__int64)&v62, v62);
    return v59;
  }
  return result;
}
