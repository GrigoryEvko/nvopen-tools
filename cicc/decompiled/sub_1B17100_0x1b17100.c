// Function: sub_1B17100
// Address: 0x1b17100
//
__int64 __fastcall sub_1B17100(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        char a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  __int64 *v12; // rsi
  __int64 *v13; // rax
  __int64 v15; // r14
  __int64 v16; // r12
  unsigned __int64 v17; // rdi
  __int64 v18; // rax
  _QWORD *v19; // r13
  _QWORD *v20; // rax
  __int64 v21; // rax
  _QWORD *v22; // rdx
  __int64 v23; // r12
  __int64 *v24; // rax
  char v25; // dl
  __int64 v26; // r13
  _QWORD *v27; // rax
  double v28; // xmm4_8
  double v29; // xmm5_8
  _QWORD *v30; // r12
  _QWORD *v31; // rax
  __int64 v32; // rax
  int v33; // r8d
  int v34; // r9d
  __int64 v35; // rax
  __int64 v36; // rbx
  _QWORD *v37; // rdx
  __int64 v38; // rdx
  __int64 v39; // rsi
  __int64 v40; // rdx
  __int64 v41; // rcx
  __int64 *v42; // rsi
  __int64 *v43; // rcx
  _QWORD *v45; // rdx
  _QWORD *v46; // rdx
  __int64 *v50; // [rsp+20h] [rbp-F0h]
  __int64 *v51; // [rsp+38h] [rbp-D8h]
  __int64 v52; // [rsp+40h] [rbp-D0h]
  unsigned __int8 v53; // [rsp+4Bh] [rbp-C5h]
  int v54; // [rsp+4Ch] [rbp-C4h]
  unsigned __int64 v55; // [rsp+50h] [rbp-C0h]
  char v56; // [rsp+5Bh] [rbp-B5h]
  unsigned int v57; // [rsp+5Ch] [rbp-B4h]
  __int64 *v58; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v59; // [rsp+68h] [rbp-A8h]
  _BYTE v60[32]; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v61; // [rsp+90h] [rbp-80h] BYREF
  __int64 *v62; // [rsp+98h] [rbp-78h]
  __int64 *v63; // [rsp+A0h] [rbp-70h]
  __int64 v64; // [rsp+A8h] [rbp-68h]
  int v65; // [rsp+B0h] [rbp-60h]
  _BYTE v66[88]; // [rsp+B8h] [rbp-58h] BYREF

  v58 = (__int64 *)v60;
  v59 = 0x400000000LL;
  v12 = *(__int64 **)(a1 + 40);
  v62 = (__int64 *)v66;
  v63 = (__int64 *)v66;
  v13 = *(__int64 **)(a1 + 32);
  v61 = 0;
  v64 = 4;
  v65 = 0;
  v50 = v12;
  if ( v13 == v12 )
    return 0;
  v51 = v13;
  v15 = a1 + 56;
  v53 = 0;
  do
  {
    v16 = *v51;
    v17 = sub_157EBA0(*v51);
    if ( !v17 )
      goto LABEL_69;
    v54 = sub_15F4D60(v17);
    v55 = sub_157EBA0(v16);
    if ( !v54 )
      goto LABEL_69;
    v57 = 0;
    v18 = sub_15F4DF0(v55, 0);
    while ( 2 )
    {
      v22 = *(_QWORD **)(a1 + 72);
      v23 = v18;
      v20 = *(_QWORD **)(a1 + 64);
      if ( v22 == v20 )
      {
        v19 = &v20[*(unsigned int *)(a1 + 84)];
        if ( v20 == v19 )
        {
          v46 = *(_QWORD **)(a1 + 64);
        }
        else
        {
          do
          {
            if ( v23 == *v20 )
              break;
            ++v20;
          }
          while ( v19 != v20 );
          v46 = v19;
        }
LABEL_20:
        while ( v46 != v20 )
        {
          if ( *v20 < 0xFFFFFFFFFFFFFFFELL )
            goto LABEL_9;
          ++v20;
        }
        if ( v19 != v20 )
          goto LABEL_10;
      }
      else
      {
        v19 = &v22[*(unsigned int *)(a1 + 80)];
        v20 = sub_16CC9F0(v15, v23);
        if ( v23 == *v20 )
        {
          v40 = *(_QWORD *)(a1 + 72);
          if ( v40 == *(_QWORD *)(a1 + 64) )
            v41 = *(unsigned int *)(a1 + 84);
          else
            v41 = *(unsigned int *)(a1 + 80);
          v46 = (_QWORD *)(v40 + 8 * v41);
          goto LABEL_20;
        }
        v21 = *(_QWORD *)(a1 + 72);
        if ( v21 == *(_QWORD *)(a1 + 64) )
        {
          v20 = (_QWORD *)(v21 + 8LL * *(unsigned int *)(a1 + 84));
          v46 = v20;
          goto LABEL_20;
        }
        v20 = (_QWORD *)(v21 + 8LL * *(unsigned int *)(a1 + 80));
LABEL_9:
        if ( v19 != v20 )
          goto LABEL_10;
      }
      v24 = v62;
      if ( v63 == v62 )
      {
        v42 = &v62[HIDWORD(v64)];
        if ( v62 != v42 )
        {
          v43 = 0;
          do
          {
            if ( v23 == *v24 )
              goto LABEL_10;
            if ( *v24 == -2 )
              v43 = v24;
            ++v24;
          }
          while ( v42 != v24 );
          if ( v43 )
          {
            *v43 = v23;
            --v65;
            ++v61;
            goto LABEL_24;
          }
        }
        if ( HIDWORD(v64) < (unsigned int)v64 )
        {
          ++HIDWORD(v64);
          *v42 = v23;
          ++v61;
          goto LABEL_24;
        }
      }
      sub_16CCBA0((__int64)&v61, v23);
      if ( !v25 )
        goto LABEL_10;
LABEL_24:
      v26 = *(_QWORD *)(v23 + 8);
      if ( !v26 )
        goto LABEL_50;
      while ( 1 )
      {
        v27 = sub_1648700(v26);
        if ( (unsigned __int8)(*((_BYTE *)v27 + 16) - 25) <= 9u )
          break;
        v26 = *(_QWORD *)(v26 + 8);
        if ( !v26 )
        {
          LODWORD(v59) = 0;
          goto LABEL_10;
        }
      }
      v56 = 1;
      v52 = v23;
      while ( 1 )
      {
        v36 = v27[5];
        v37 = *(_QWORD **)(a1 + 72);
        v31 = *(_QWORD **)(a1 + 64);
        if ( v37 == v31 )
        {
          v30 = &v31[*(unsigned int *)(a1 + 84)];
          if ( v31 == v30 )
          {
            v45 = *(_QWORD **)(a1 + 64);
          }
          else
          {
            do
            {
              if ( v36 == *v31 )
                break;
              ++v31;
            }
            while ( v30 != v31 );
            v45 = v30;
          }
        }
        else
        {
          v30 = &v37[*(unsigned int *)(a1 + 80)];
          v31 = sub_16CC9F0(v15, v36);
          if ( v36 == *v31 )
          {
            v38 = *(_QWORD *)(a1 + 72);
            v39 = v38 == *(_QWORD *)(a1 + 64) ? *(unsigned int *)(a1 + 84) : *(unsigned int *)(a1 + 80);
            v45 = (_QWORD *)(v38 + 8 * v39);
          }
          else
          {
            v32 = *(_QWORD *)(a1 + 72);
            if ( v32 != *(_QWORD *)(a1 + 64) )
            {
              v31 = (_QWORD *)(v32 + 8LL * *(unsigned int *)(a1 + 80));
              goto LABEL_30;
            }
            v31 = (_QWORD *)(v32 + 8LL * *(unsigned int *)(a1 + 84));
            v45 = v31;
          }
        }
        if ( v31 != v45 )
        {
          while ( *v31 >= 0xFFFFFFFFFFFFFFFELL )
          {
            if ( v45 == ++v31 )
            {
              if ( v30 != v31 )
                goto LABEL_31;
              goto LABEL_47;
            }
          }
        }
LABEL_30:
        if ( v30 != v31 )
          break;
LABEL_47:
        v56 = 0;
        v26 = *(_QWORD *)(v26 + 8);
        if ( !v26 )
        {
LABEL_48:
          if ( !v56 )
          {
            sub_1AAB350(v52, v58, (unsigned int)v59, ".loopexit", a2, a3, a5, a6, a7, a8, v28, v29, a11, a12, a4);
            v53 = 1;
          }
          goto LABEL_50;
        }
        while ( 1 )
        {
          v27 = sub_1648700(v26);
          if ( (unsigned __int8)(*((_BYTE *)v27 + 16) - 25) <= 9u )
            break;
LABEL_35:
          v26 = *(_QWORD *)(v26 + 8);
          if ( !v26 )
            goto LABEL_48;
        }
      }
LABEL_31:
      if ( *(_BYTE *)(sub_157EBA0(v36) + 16) != 28 )
      {
        v35 = (unsigned int)v59;
        if ( (unsigned int)v59 >= HIDWORD(v59) )
        {
          sub_16CD150((__int64)&v58, v60, 0, 8, v33, v34);
          v35 = (unsigned int)v59;
        }
        v58[v35] = v36;
        LODWORD(v59) = v59 + 1;
        goto LABEL_35;
      }
LABEL_50:
      LODWORD(v59) = 0;
LABEL_10:
      if ( v54 != ++v57 )
      {
        v18 = sub_15F4DF0(v55, v57);
        continue;
      }
      break;
    }
LABEL_69:
    ++v51;
  }
  while ( v50 != v51 );
  if ( v63 != v62 )
    _libc_free((unsigned __int64)v63);
  if ( v58 != (__int64 *)v60 )
    _libc_free((unsigned __int64)v58);
  return v53;
}
