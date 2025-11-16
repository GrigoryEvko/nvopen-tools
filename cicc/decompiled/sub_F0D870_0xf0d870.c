// Function: sub_F0D870
// Address: 0xf0d870
//
unsigned __int8 *__fastcall sub_F0D870(const __m128i *a1, unsigned __int8 *a2, __int64 a3, __int64 a4)
{
  char v6; // r13
  __int64 *v7; // rdx
  __int64 v8; // rax
  __int64 *v9; // rdx
  __int64 v10; // rax
  __int64 v11; // r15
  char v12; // al
  __int64 v13; // rcx
  __int64 v14; // r11
  unsigned int v15; // eax
  __int64 v16; // rax
  int v17; // edi
  __m128i v18; // xmm0
  __m128i v19; // xmm1
  unsigned __int64 v20; // xmm2_8
  __int64 v21; // rdi
  __m128i v22; // xmm3
  __int64 v23; // rax
  __int64 v24; // rax
  unsigned __int8 *v25; // r13
  __int64 v27; // rdx
  __int64 v28; // rcx
  unsigned int **v29; // rdi
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rdx
  __int64 *v33; // rdi
  __int64 *v34; // rdi
  _BYTE *v35; // [rsp+0h] [rbp-160h]
  _BYTE *v36; // [rsp+8h] [rbp-158h]
  _BYTE *v37; // [rsp+10h] [rbp-150h]
  _BYTE *v38; // [rsp+18h] [rbp-148h]
  __int64 v39; // [rsp+20h] [rbp-140h]
  __int64 v40; // [rsp+28h] [rbp-138h]
  unsigned int v42; // [rsp+30h] [rbp-130h]
  __int64 v43; // [rsp+30h] [rbp-130h]
  char v44; // [rsp+38h] [rbp-128h]
  __int64 v45; // [rsp+38h] [rbp-128h]
  unsigned int v46; // [rsp+38h] [rbp-128h]
  unsigned int v47; // [rsp+38h] [rbp-128h]
  __int64 v48; // [rsp+40h] [rbp-120h]
  int v49; // [rsp+48h] [rbp-118h]
  char v50; // [rsp+4Ch] [rbp-114h]
  char v51; // [rsp+4Dh] [rbp-113h]
  __int16 v52; // [rsp+4Eh] [rbp-112h]
  unsigned int v53; // [rsp+5Ch] [rbp-104h] BYREF
  __int64 v54; // [rsp+60h] [rbp-100h] BYREF
  __int64 v55; // [rsp+68h] [rbp-F8h] BYREF
  __int64 v56; // [rsp+70h] [rbp-F0h] BYREF
  int v57; // [rsp+78h] [rbp-E8h]
  _BYTE v58[32]; // [rsp+80h] [rbp-E0h] BYREF
  __int16 v59; // [rsp+A0h] [rbp-C0h]
  _QWORD v60[6]; // [rsp+B0h] [rbp-B0h] BYREF
  _OWORD v61[2]; // [rsp+E0h] [rbp-80h] BYREF
  unsigned __int64 v62; // [rsp+100h] [rbp-60h]
  unsigned __int8 *v63; // [rsp+108h] [rbp-58h]
  __m128i v64; // [rsp+110h] [rbp-50h]
  __int64 v65; // [rsp+120h] [rbp-40h]

  if ( *(_BYTE *)a3 == 86 )
  {
    if ( (*(_BYTE *)(a3 + 7) & 0x40) != 0 )
    {
      v9 = *(__int64 **)(a3 - 8);
      v10 = *v9;
      if ( !*v9 )
        goto LABEL_2;
    }
    else
    {
      v9 = (__int64 *)(a3 - 32LL * (*(_DWORD *)(a3 + 4) & 0x7FFFFFF));
      v10 = *v9;
      if ( !*v9 )
        goto LABEL_2;
    }
    if ( v9[4] )
    {
      v36 = (_BYTE *)v9[8];
      if ( v36 )
      {
        v38 = (_BYTE *)v9[4];
        v40 = v10;
        if ( *(_BYTE *)a4 != 86 )
        {
          v44 = 0;
          v6 = 1;
          v50 = 0;
          goto LABEL_14;
        }
        v6 = 1;
        goto LABEL_3;
      }
      v38 = (_BYTE *)v9[4];
    }
    v40 = v10;
  }
LABEL_2:
  v6 = 0;
  if ( *(_BYTE *)a4 != 86 )
    return 0;
LABEL_3:
  if ( (*(_BYTE *)(a4 + 7) & 0x40) != 0 )
  {
    v7 = *(__int64 **)(a4 - 8);
    v8 = *v7;
    if ( !*v7 )
      goto LABEL_28;
  }
  else
  {
    v7 = (__int64 *)(a4 - 32LL * (*(_DWORD *)(a4 + 4) & 0x7FFFFFF));
    v8 = *v7;
    if ( !*v7 )
      goto LABEL_28;
  }
  if ( v7[4] )
  {
    v35 = (_BYTE *)v7[8];
    if ( v35 )
    {
      v44 = v6;
      v37 = (_BYTE *)v7[4];
      v39 = v8;
      v50 = 1;
      goto LABEL_14;
    }
    v37 = (_BYTE *)v7[4];
  }
  v39 = v8;
LABEL_28:
  if ( !v6 )
    return 0;
  v44 = 0;
  v50 = 0;
LABEL_14:
  v11 = a1[2].m128i_i64[0];
  v49 = *(_DWORD *)(v11 + 104);
  v48 = *(_QWORD *)(v11 + 96);
  v51 = *(_BYTE *)(v11 + 110);
  v52 = *(_WORD *)(v11 + 108);
  v12 = sub_920620((__int64)a2);
  v13 = 0;
  v14 = a4;
  if ( v12 )
  {
    v15 = sub_B45210((__int64)a2);
    v14 = a4;
    v13 = v15;
    *(_DWORD *)(a1[2].m128i_i64[0] + 104) = v15;
  }
  v16 = a1[10].m128i_i64[0];
  v17 = *a2;
  v55 = 0;
  v18 = _mm_loadu_si128(a1 + 6);
  v19 = _mm_loadu_si128(a1 + 7);
  v56 = 0;
  v65 = v16;
  v20 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
  v21 = (unsigned int)(v17 - 29);
  v60[0] = &v53;
  v22 = _mm_loadu_si128(a1 + 9);
  v60[1] = &v55;
  v60[2] = &v56;
  v62 = v20;
  v53 = v21;
  v63 = a2;
  v60[3] = a1;
  v60[4] = &v54;
  v60[5] = a2;
  v61[0] = v18;
  v61[1] = v19;
  v64 = v22;
  if ( v44 )
  {
    if ( v40 == v39 )
    {
      v43 = v14;
      v47 = v13;
      v54 = v40;
      v55 = sub_101E830(v21, v38, v37, v13, v61);
      v30 = sub_101E830(v53, v36, v35, v47, v61);
      v31 = *(_QWORD *)(a3 + 16);
      v56 = v30;
      if ( !v31 )
        goto LABEL_38;
      if ( *(_QWORD *)(v31 + 8) )
        goto LABEL_38;
      v32 = *(_QWORD *)(v43 + 16);
      if ( !v32 || *(_QWORD *)(v32 + 8) )
        goto LABEL_38;
      v27 = v55;
      if ( v30 )
      {
        v28 = v30;
        if ( v55 )
        {
LABEL_40:
          v29 = (unsigned int **)a1[2].m128i_i64[0];
          v59 = 257;
          v25 = (unsigned __int8 *)sub_B36550(v29, v54, v27, v28, (__int64)v58, 0);
          sub_BD6B90(v25, a2);
          goto LABEL_22;
        }
        v33 = (__int64 *)a1[2].m128i_i64[0];
        v59 = 257;
        v55 = sub_F0A990(v33, v53, (__int64)v38, (__int64)v37, v57, 0, (__int64)v58, 0);
        goto LABEL_38;
      }
      if ( v55 )
      {
        v34 = (__int64 *)a1[2].m128i_i64[0];
        v59 = 257;
        v56 = sub_F0A990(v34, v53, (__int64)v36, (__int64)v35, v57, 0, (__int64)v58, 0);
        goto LABEL_38;
      }
      goto LABEL_21;
    }
  }
  else if ( !v6 )
  {
    goto LABEL_19;
  }
  v23 = *(_QWORD *)(a3 + 16);
  if ( v23 && !*(_QWORD *)(v23 + 8) )
  {
    v42 = v13;
    v45 = v14;
    v54 = v40;
    v55 = sub_101E830(v21, v38, v14, v13, v61);
    v56 = sub_101E830(v53, v36, v45, v42, v61);
    v25 = (unsigned __int8 *)sub_F09E10((__int64)v60, v38, v36, v45);
    if ( v25 )
      goto LABEL_22;
    goto LABEL_38;
  }
LABEL_19:
  if ( v50 )
  {
    v24 = *(_QWORD *)(v14 + 16);
    if ( v24 )
    {
      if ( !*(_QWORD *)(v24 + 8) )
      {
        v46 = v13;
        v54 = v39;
        v55 = sub_101E830(v21, a3, v37, v13, v61);
        v56 = sub_101E830(v53, a3, v35, v46, v61);
        v25 = (unsigned __int8 *)sub_F09E10((__int64)v60, v37, v35, a3);
        if ( v25 )
          goto LABEL_22;
LABEL_38:
        v27 = v55;
        if ( v55 )
        {
          v28 = v56;
          if ( v56 )
            goto LABEL_40;
        }
      }
    }
  }
LABEL_21:
  v25 = 0;
LABEL_22:
  *(_QWORD *)(v11 + 96) = v48;
  *(_DWORD *)(v11 + 104) = v49;
  *(_WORD *)(v11 + 108) = v52;
  *(_BYTE *)(v11 + 110) = v51;
  return v25;
}
