// Function: sub_3326BF0
// Address: 0x3326bf0
//
__int64 __fastcall sub_3326BF0(__int64 a1, __int64 a2, int a3, __int64 a4, __int64 a5, int a6)
{
  __int64 v8; // r13
  unsigned __int16 *v9; // rdx
  int v10; // eax
  __int64 v11; // r14
  __int64 v12; // r14
  __int64 v13; // rdx
  __int64 v14; // rdx
  __int64 v15; // rdi
  int v16; // eax
  int v17; // ecx
  int v18; // edx
  int v19; // r8d
  __int64 v20; // rax
  unsigned int v21; // edx
  unsigned int v22; // ecx
  unsigned int v23; // edx
  __m128i v24; // xmm0
  __int64 v25; // r8
  __int64 v26; // r9
  char v27; // al
  __int64 v28; // rdi
  unsigned __int16 *v29; // r13
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rsi
  char v33; // al
  __int64 v34; // rax
  __int64 v35; // r11
  unsigned __int16 v36; // r10
  int v37; // edx
  __m128i v38; // xmm2
  __int64 v39; // r15
  char v40; // al
  __int64 v41; // rdi
  __int64 v42; // rax
  unsigned __int16 v43; // r10
  int v44; // edx
  unsigned __int64 v45; // rdi
  __int64 v46; // rax
  __int64 result; // rax
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // rdx
  __int64 v52; // rsi
  __int64 v53; // rax
  __int64 v54; // r13
  int v55; // edx
  unsigned __int64 v56; // rdi
  __int64 v57; // rax
  __int64 v58; // rdi
  __int64 v59; // rax
  unsigned int v60; // edx
  __m128i v61; // xmm3
  __int64 v62; // [rsp-60h] [rbp-190h]
  __int64 v63; // [rsp-58h] [rbp-188h]
  __int128 v64; // [rsp-40h] [rbp-170h]
  __int64 v65; // [rsp-30h] [rbp-160h]
  __int64 v66; // [rsp+0h] [rbp-130h]
  unsigned int v67; // [rsp+8h] [rbp-128h]
  unsigned int v68; // [rsp+10h] [rbp-120h]
  unsigned __int16 v69; // [rsp+14h] [rbp-11Ch]
  __int64 v70; // [rsp+18h] [rbp-118h]
  __int64 v71; // [rsp+20h] [rbp-110h]
  __int64 v72; // [rsp+28h] [rbp-108h]
  _BYTE *v73; // [rsp+38h] [rbp-F8h]
  __int128 v76; // [rsp+50h] [rbp-E0h]
  unsigned __int16 v77; // [rsp+50h] [rbp-E0h]
  __int64 v78; // [rsp+50h] [rbp-E0h]
  __m128i v79; // [rsp+A0h] [rbp-90h] BYREF
  __int64 v80; // [rsp+B0h] [rbp-80h]
  __int64 v81; // [rsp+B8h] [rbp-78h]
  __int128 v82; // [rsp+C0h] [rbp-70h]
  __int64 v83; // [rsp+D0h] [rbp-60h]
  __m128i v84; // [rsp+E0h] [rbp-50h] BYREF
  __int64 v85; // [rsp+F0h] [rbp-40h]
  __int64 v86; // [rsp+F8h] [rbp-38h]

  *(_QWORD *)&v76 = a4;
  *((_QWORD *)&v76 + 1) = a5;
  v8 = 16LL * (unsigned int)a5;
  v9 = (unsigned __int16 *)(v8 + *(_QWORD *)(a4 + 48));
  v10 = *v9;
  v11 = *((_QWORD *)v9 + 1);
  v79.m128i_i16[0] = v10;
  v79.m128i_i64[1] = v11;
  if ( (_WORD)v10 )
  {
    if ( (unsigned __int16)(v10 - 17) > 0xD3u )
    {
      v84.m128i_i16[0] = v10;
      v84.m128i_i64[1] = v11;
      goto LABEL_4;
    }
    LOWORD(v10) = word_4456580[v10 - 1];
    v13 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v79) )
    {
      v84.m128i_i64[1] = v11;
      v84.m128i_i16[0] = 0;
      goto LABEL_9;
    }
    LOWORD(v10) = sub_3009970((__int64)&v79, a2, v48, v49, v50);
  }
  v84.m128i_i16[0] = v10;
  v84.m128i_i64[1] = v13;
  if ( !(_WORD)v10 )
  {
LABEL_9:
    v80 = sub_3007260((__int64)&v84);
    LODWORD(v12) = v80;
    v81 = v14;
    goto LABEL_10;
  }
LABEL_4:
  if ( (_WORD)v10 == 1 || (unsigned __int16)(v10 - 504) <= 7u )
    goto LABEL_59;
  v12 = *(_QWORD *)&byte_444C4A0[16 * (unsigned __int16)v10 - 16];
LABEL_10:
  *(__m128i *)a2 = _mm_loadu_si128(&v79);
  v15 = *(_QWORD *)(a1 + 16);
  switch ( (_DWORD)v12 )
  {
    case 1:
      v51 = 2;
      LOWORD(v16) = 2;
LABEL_36:
      v52 = *(_QWORD *)(a1 + 8);
      v17 = (unsigned __int16)v16;
      v19 = 0;
      goto LABEL_37;
    case 2:
      v51 = 3;
      LOWORD(v16) = 3;
      goto LABEL_36;
    case 4:
      v51 = 4;
      LOWORD(v16) = 4;
      goto LABEL_36;
    case 8:
      v51 = 5;
      LOWORD(v16) = 5;
      goto LABEL_36;
    case 0x10:
      v51 = 6;
      LOWORD(v16) = 6;
      goto LABEL_36;
    case 0x20:
      v51 = 7;
      LOWORD(v16) = 7;
      goto LABEL_36;
    case 0x40:
      v51 = 8;
      LOWORD(v16) = 8;
      goto LABEL_36;
    case 0x80:
      v51 = 9;
      LOWORD(v16) = 9;
      goto LABEL_36;
  }
  v16 = sub_3007020(*(_QWORD **)(v15 + 64), v12);
  HIWORD(v17) = HIWORD(v16);
  v19 = v18;
  if ( !(_WORD)v16 )
  {
    v15 = *(_QWORD *)(a1 + 16);
    goto LABEL_20;
  }
  v52 = *(_QWORD *)(a1 + 8);
  v15 = *(_QWORD *)(a1 + 16);
  v51 = (unsigned __int16)v16;
LABEL_37:
  if ( *(_QWORD *)(v52 + 8 * v51 + 112) )
  {
    LOWORD(v17) = v16;
    v53 = sub_33FAF80(v15, 234, a3, v17, v19, a6, v76);
    v84.m128i_i32[2] = v12;
    v54 = 1LL << ((unsigned __int8)v12 - 1);
    *(_QWORD *)(a2 + 112) = v53;
    *(_DWORD *)(a2 + 120) = v55;
    if ( (unsigned int)v12 > 0x40 )
    {
      sub_C43690((__int64)&v84, 0, 0);
      if ( v84.m128i_i32[2] > 0x40u )
      {
        *(_QWORD *)(v84.m128i_i64[0] + 8LL * ((unsigned int)(v12 - 1) >> 6)) |= v54;
        if ( *(_DWORD *)(a2 + 136) > 0x40u )
        {
LABEL_41:
          v56 = *(_QWORD *)(a2 + 128);
          if ( v56 )
            j_j___libc_free_0_0(v56);
        }
LABEL_43:
        v57 = v84.m128i_i64[0];
        *(_BYTE *)(a2 + 144) = v12 - 1;
        *(_QWORD *)(a2 + 128) = v57;
        result = v84.m128i_u32[2];
        *(_DWORD *)(a2 + 136) = v84.m128i_i32[2];
        return result;
      }
    }
    else
    {
      v84.m128i_i64[0] = 0;
    }
    v84.m128i_i64[0] |= v54;
    if ( *(_DWORD *)(a2 + 136) > 0x40u )
      goto LABEL_41;
    goto LABEL_43;
  }
LABEL_20:
  v73 = (_BYTE *)sub_2E79000(*(__int64 **)(v15 + 40));
  v69 = *(_WORD *)(*(_QWORD *)(a1 + 8) + 2862LL);
  v20 = sub_33EE0D0(*(_QWORD *)(a1 + 16), v79.m128i_u32[0], v79.m128i_i64[1], v69, 0);
  v22 = v21;
  v23 = *(_DWORD *)(v20 + 96);
  *(_QWORD *)(a2 + 32) = v20;
  *(_DWORD *)(a2 + 40) = v22;
  v67 = v22;
  v70 = v20;
  v68 = v23;
  v66 = *(_QWORD *)(*(_QWORD *)(a1 + 16) + 40LL);
  sub_2EAC300((__int64)&v84, v66, v23, 0);
  v24 = _mm_loadu_si128(&v84);
  v86 = 0;
  v25 = *(_QWORD *)(a2 + 32);
  v26 = *(_QWORD *)(a2 + 40);
  v84.m128i_i64[0] = 0;
  *(_DWORD *)(a2 + 104) = v85;
  v27 = BYTE4(v85);
  *(__m128i *)(a2 + 88) = v24;
  *(_BYTE *)(a2 + 108) = v27;
  v28 = *(_QWORD *)(a1 + 16);
  v29 = (unsigned __int16 *)(*(_QWORD *)(a4 + 48) + v8);
  v30 = *(_QWORD *)(a2 + 104);
  v82 = (__int128)v24;
  v84.m128i_i64[1] = 0;
  v85 = 0;
  v31 = *((_QWORD *)v29 + 1);
  v32 = *v29;
  v71 = v25;
  v72 = v26;
  v83 = v30;
  v33 = sub_33CC4A0(v28, v32, v31);
  v34 = sub_33F4560(v28, (int)v28 + 288, 0, a3, v76, DWORD2(v76), v71, v72, v82, v83, v33, 0, (__int64)&v84);
  v35 = v70;
  v36 = v69;
  *(_QWORD *)(a2 + 16) = v34;
  *(_DWORD *)(a2 + 24) = v37;
  if ( *v73 )
  {
    v38 = _mm_loadu_si128((const __m128i *)(a2 + 88));
    v39 = v67;
    *(_DWORD *)(a2 + 80) = *(_DWORD *)(a2 + 104);
    v40 = *(_BYTE *)(a2 + 108);
    *(__m128i *)(a2 + 64) = v38;
  }
  else
  {
    v58 = *(_QWORD *)(a1 + 16);
    BYTE8(v82) = 0;
    *(_QWORD *)&v82 = ((unsigned int)v12 >> 3) - 1;
    v59 = sub_3409320(v58, v70, v67, ((unsigned int)v12 >> 3) - 1, DWORD2(v82), a3, 0);
    v39 = v60;
    v78 = v59;
    sub_2EAC300((__int64)&v84, v66, v68, ((unsigned int)v12 >> 3) - 1);
    v61 = _mm_loadu_si128(&v84);
    v36 = v69;
    v35 = v78;
    *(_DWORD *)(a2 + 80) = v85;
    v40 = BYTE4(v85);
    *(__m128i *)(a2 + 64) = v61;
  }
  *(_BYTE *)(a2 + 84) = v40;
  *(_QWORD *)(a2 + 48) = v35;
  *(_DWORD *)(a2 + 56) = v39;
  v41 = *(_QWORD *)(a1 + 16);
  v65 = *(_QWORD *)(a2 + 80);
  v64 = *(_OWORD *)(a2 + 64);
  v63 = *(_QWORD *)(a2 + 24);
  v62 = *(_QWORD *)(a2 + 16);
  v77 = v36;
  v84 = 0u;
  v85 = 0;
  v86 = 0;
  v42 = sub_33F1DB0(v41, 1, a3, v36, 0, 0, v62, v63, v35, v39, v64, v65, 5, 0, 0, (__int64)&v84);
  v43 = v77;
  *(_QWORD *)(a2 + 112) = v42;
  *(_DWORD *)(a2 + 120) = v44;
  if ( (unsigned __int16)(v77 - 17) <= 0xD3u )
    v43 = word_4456580[v77 - 1];
  if ( v43 <= 1u || (unsigned __int16)(v43 - 504) <= 7u )
LABEL_59:
    BUG();
  v84.m128i_i32[2] = *(_QWORD *)&byte_444C4A0[16 * v43 - 16];
  if ( v84.m128i_i32[2] > 0x40u )
  {
    sub_C43690((__int64)&v84, 0, 0);
    if ( v84.m128i_i32[2] > 0x40u )
    {
      *(_QWORD *)v84.m128i_i64[0] |= 0x80uLL;
      goto LABEL_29;
    }
  }
  else
  {
    v84.m128i_i64[0] = 0;
  }
  v84.m128i_i64[0] |= 0x80uLL;
LABEL_29:
  if ( *(_DWORD *)(a2 + 136) > 0x40u )
  {
    v45 = *(_QWORD *)(a2 + 128);
    if ( v45 )
      j_j___libc_free_0_0(v45);
  }
  v46 = v84.m128i_i64[0];
  *(_BYTE *)(a2 + 144) = 7;
  *(_QWORD *)(a2 + 128) = v46;
  result = v84.m128i_u32[2];
  *(_DWORD *)(a2 + 136) = v84.m128i_i32[2];
  return result;
}
