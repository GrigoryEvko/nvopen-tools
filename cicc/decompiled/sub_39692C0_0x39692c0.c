// Function: sub_39692C0
// Address: 0x39692c0
//
__int64 __fastcall sub_39692C0(
        __int64 a1,
        double a2,
        double a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        __m128 a9)
{
  _QWORD *v9; // r14
  _BYTE *v11; // rdx
  _BYTE *v12; // rsi
  _BYTE *v13; // rsi
  unsigned __int64 v14; // rdi
  _QWORD *v15; // r8
  __int64 v16; // r12
  _QWORD *v17; // r15
  bool v18; // zf
  unsigned __int16 v19; // ax
  __int64 v20; // rsi
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 *v23; // r8
  __int64 v24; // r9
  double v25; // xmm4_8
  double v26; // xmm5_8
  __int64 *v27; // rax
  __int64 v28; // rdx
  __int64 *v29; // r12
  __int64 v30; // rsi
  __int64 *v31; // rbx
  __int64 *v32; // rbx
  __int64 *v33; // r12
  __int64 v34; // rdi
  __int64 v35; // rax
  unsigned __int16 v36; // ax
  unsigned __int64 v37; // rdi
  __int64 v39; // rcx
  __int64 **v40; // r8
  __int64 v41; // rcx
  __int64 v42; // rax
  unsigned int **v43; // rbx
  unsigned int *v44; // rax
  unsigned int *v45; // rax
  unsigned int **v46; // r10
  int v47; // r11d
  unsigned int *v48; // rax
  int v49; // r11d
  unsigned int *v50; // rax
  int v51; // r11d
  __int64 **v52; // r15
  __int64 *v53; // rdx
  char v54; // al
  unsigned __int64 v55; // rdi
  __int64 *v56; // rax
  _QWORD *v57; // rax
  char v58; // al
  char v59; // al
  char v60; // al
  __int64 **v61; // [rsp+0h] [rbp-120h]
  __int64 **v62; // [rsp+0h] [rbp-120h]
  __int64 v63; // [rsp+8h] [rbp-118h]
  __int64 **v64; // [rsp+8h] [rbp-118h]
  __int64 **v65; // [rsp+10h] [rbp-110h]
  unsigned __int8 v66; // [rsp+1Eh] [rbp-102h]
  char v67; // [rsp+1Fh] [rbp-101h]
  __int64 v68; // [rsp+28h] [rbp-F8h] BYREF
  _QWORD *v69; // [rsp+30h] [rbp-F0h] BYREF
  _BYTE *v70; // [rsp+38h] [rbp-E8h]
  _BYTE *v71; // [rsp+40h] [rbp-E0h]
  __int64 v72; // [rsp+50h] [rbp-D0h] BYREF
  _BYTE *v73; // [rsp+58h] [rbp-C8h]
  _BYTE *v74; // [rsp+60h] [rbp-C0h]
  __int64 v75; // [rsp+68h] [rbp-B8h]
  int v76; // [rsp+70h] [rbp-B0h]
  _BYTE v77[40]; // [rsp+78h] [rbp-A8h] BYREF
  _QWORD *v78; // [rsp+A0h] [rbp-80h] BYREF
  _BYTE *v79; // [rsp+A8h] [rbp-78h]
  _BYTE *v80; // [rsp+B0h] [rbp-70h]
  __int64 v81; // [rsp+B8h] [rbp-68h]
  int v82; // [rsp+C0h] [rbp-60h]
  _BYTE v83[88]; // [rsp+C8h] [rbp-58h] BYREF

  v9 = *(_QWORD **)(a1 + 112);
  v69 = 0;
  v70 = 0;
  v71 = 0;
  if ( (_QWORD *)(a1 + 112) == v9 )
    return 0;
  v11 = 0;
  v12 = 0;
  while ( 1 )
  {
    v78 = v9 + 2;
    if ( v12 == v11 )
    {
      sub_3963390((__int64)&v69, v12, &v78);
      v13 = v70;
    }
    else
    {
      if ( v12 )
      {
        *(_QWORD *)v12 = v9 + 2;
        v12 = v70;
      }
      v13 = v12 + 8;
      v70 = v13;
    }
    sub_39612D0((__int64)v69, ((v13 - (_BYTE *)v69) >> 3) - 1, 0, *((_QWORD *)v13 - 1));
    v9 = (_QWORD *)*v9;
    if ( (_QWORD *)(a1 + 112) == v9 )
      break;
    v12 = v70;
    v11 = v71;
  }
  v14 = (unsigned __int64)v70;
  v15 = v69;
  if ( v70 == (_BYTE *)v69 )
  {
    v66 = 0;
    goto LABEL_37;
  }
  v66 = 0;
  v16 = *v69;
  v17 = &v78;
  if ( v70 - (_BYTE *)v69 > 8 )
    goto LABEL_47;
  while ( 1 )
  {
    v18 = *(_BYTE *)(v16 + 41) == 0;
    v14 = (unsigned __int64)(v70 - 8);
    v70 -= 8;
    if ( !v18 )
      goto LABEL_45;
    v19 = sub_3962650(*(_QWORD *)(a1 + 56), 1.0);
    if ( (_BYTE)v19 )
    {
      if ( *(int *)(v16 + 8) > 0 )
        break;
    }
    if ( HIBYTE(v19) && *(int *)(v16 + 12) > 0 )
      break;
LABEL_44:
    v14 = (unsigned __int64)v70;
LABEL_45:
    v15 = v69;
    if ( v69 == (_QWORD *)v14 )
      goto LABEL_37;
    v16 = *v69;
    if ( (__int64)(v14 - (_QWORD)v69) > 8 )
    {
LABEL_47:
      v39 = *(_QWORD *)(v14 - 8);
      *(_QWORD *)(v14 - 8) = v16;
      sub_39613C0((__int64)v15, 0, (__int64)(v14 - 8 - (_QWORD)v15) >> 3, v39);
    }
  }
  v72 = 0;
  v73 = v77;
  v74 = v77;
  v75 = 4;
  v76 = 0;
  v78 = 0;
  v79 = v83;
  v80 = v83;
  v81 = 4;
  v82 = 0;
  v67 = sub_3960EF0(*(_BYTE **)v16);
  if ( !v67 )
  {
    sub_39673A0(a1, (__int64 *)v16);
    v20 = v16;
    sub_39692A0((__int64 *)a1, v16);
    v40 = *(__int64 ***)(v16 + 48);
    v41 = 8LL * *(unsigned int *)(v16 + 56);
    v65 = &v40[(unsigned __int64)v41 / 8];
    v42 = v41 >> 3;
    v22 = v41 >> 5;
    if ( v22 )
    {
      v43 = *(unsigned int ***)(v16 + 48);
      v22 = (__int64)&v40[4 * v22];
      while ( 1 )
      {
        v44 = *v43;
        v21 = (*v43)[4];
        if ( (int)v21 < 0 )
          break;
        v20 = v44[5];
        if ( (int)v20 < 0 )
          break;
        v21 = (unsigned int)v20 | (unsigned int)v21;
        if ( !(_DWORD)v21 )
          break;
        v20 = *(unsigned int *)(a1 + 72);
        if ( (int)v44[6] > (int)v20 || v44[7] > dword_5055B20 || v44[8] > dword_5055A40 )
          break;
        v45 = v43[1];
        v46 = v43 + 1;
        v21 = v45[4];
        if ( (int)v21 < 0 )
          goto LABEL_93;
        v47 = v45[5];
        if ( v47 < 0 )
          goto LABEL_93;
        v21 = v47 | (unsigned int)v21;
        if ( !(_DWORD)v21 )
          goto LABEL_93;
        if ( (int)v20 < (int)v45[6] )
          goto LABEL_93;
        if ( dword_5055B20 < v45[7] )
          goto LABEL_93;
        if ( dword_5055A40 < v45[8] )
          goto LABEL_93;
        v48 = v43[2];
        v46 = v43 + 2;
        v21 = v48[4];
        if ( (int)v21 < 0
          || (v49 = v48[5], v49 < 0)
          || (v21 = v49 | (unsigned int)v21, !(_DWORD)v21)
          || (int)v20 < (int)v48[6]
          || dword_5055B20 < v48[7]
          || dword_5055A40 < v48[8]
          || (v50 = v43[3], v46 = v43 + 3, v21 = v50[4], (int)v21 < 0)
          || (v51 = v50[5], v51 < 0)
          || (v21 = v51 | (unsigned int)v21, !(_DWORD)v21)
          || (int)v20 < (int)v50[6]
          || dword_5055B20 < v50[7]
          || dword_5055A40 < v50[8] )
        {
LABEL_93:
          v43 = v46;
          break;
        }
        v43 += 4;
        if ( (unsigned int **)v22 == v43 )
        {
          v42 = ((char *)v65 - (char *)v43) >> 3;
          goto LABEL_82;
        }
      }
LABEL_57:
      if ( v65 != (__int64 **)v43 )
      {
        v37 = (unsigned __int64)v80;
        if ( v80 == v79 )
        {
LABEL_42:
          if ( v74 != v73 )
            _libc_free((unsigned __int64)v74);
          goto LABEL_44;
        }
LABEL_41:
        _libc_free(v37);
        goto LABEL_42;
      }
LABEL_85:
      if ( v65 == v40 )
        goto LABEL_26;
LABEL_86:
      v63 = (__int64)v17;
      v52 = v40;
      do
      {
        v53 = *v52;
        v20 = v16;
        v68 = 0;
        v54 = sub_3965950(a1, (__int64 *)v16, v53, &v68, (__int64)&v72);
        if ( v54 )
        {
          v20 = v68;
          v67 = v54;
          if ( v68 )
            sub_1412190(v63, v68);
        }
        ++v52;
      }
      while ( v65 != v52 );
      v17 = (_QWORD *)v63;
      if ( !v67 )
        goto LABEL_26;
      goto LABEL_19;
    }
    v43 = *(unsigned int ***)(v16 + 48);
LABEL_82:
    switch ( v42 )
    {
      case 2LL:
        v20 = a1 + 64;
        break;
      case 3LL:
        v20 = a1 + 64;
        v61 = *(__int64 ***)(v16 + 48);
        v59 = sub_3961900((int *)*v43 + 4, a1 + 64);
        v40 = v61;
        if ( !v59 )
          goto LABEL_57;
        v20 = a1 + 64;
        ++v43;
        break;
      case 1LL:
        v20 = a1 + 64;
        goto LABEL_118;
      default:
        goto LABEL_85;
    }
    v62 = v40;
    v60 = sub_3961900((int *)*v43 + 4, v20);
    v40 = v62;
    if ( !v60 )
      goto LABEL_57;
    ++v43;
LABEL_118:
    v64 = v40;
    v58 = sub_3961900((int *)*v43 + 4, v20);
    v40 = v64;
    if ( v58 )
    {
      if ( v65 == v64 )
        goto LABEL_26;
      goto LABEL_86;
    }
    goto LABEL_57;
  }
  v20 = *(_QWORD *)v16;
  if ( *(_BYTE *)(*(_QWORD *)v16 + 16LL) != 17 )
    v20 = 0;
  if ( !(unsigned __int8)sub_39627B0(a1, v20, (__int64)&v72) )
    goto LABEL_26;
LABEL_19:
  v66 = byte_5055220;
  if ( byte_5055220 )
  {
    v27 = *(__int64 **)(v16 + 144);
    if ( v27 == *(__int64 **)(v16 + 136) )
      v28 = *(unsigned int *)(v16 + 156);
    else
      v28 = *(unsigned int *)(v16 + 152);
    v29 = &v27[v28];
    if ( v27 != v29 )
    {
      while ( 1 )
      {
        v30 = *v27;
        v31 = v27;
        if ( (unsigned __int64)*v27 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v29 == ++v27 )
          goto LABEL_25;
      }
      if ( v29 != v27 )
      {
        v55 = (unsigned __int64)v80;
        v24 = (__int64)v79;
        if ( v80 != v79 )
        {
LABEL_97:
          sub_16CCBA0((__int64)v17, v30);
          v55 = (unsigned __int64)v80;
          v24 = (__int64)v79;
          goto LABEL_98;
        }
        while ( 1 )
        {
          v23 = (__int64 *)(v55 + 8LL * HIDWORD(v81));
          if ( (__int64 *)v55 == v23 )
          {
LABEL_112:
            if ( HIDWORD(v81) >= (unsigned int)v81 )
              goto LABEL_97;
            ++HIDWORD(v81);
            *v23 = v30;
            v24 = (__int64)v79;
            v78 = (_QWORD *)((char *)v78 + 1);
            v55 = (unsigned __int64)v80;
          }
          else
          {
            v57 = (_QWORD *)v55;
            v22 = 0;
            while ( v30 != *v57 )
            {
              if ( *v57 == -2 )
                v22 = (__int64)v57;
              if ( v23 == ++v57 )
              {
                if ( !v22 )
                  goto LABEL_112;
                *(_QWORD *)v22 = v30;
                v55 = (unsigned __int64)v80;
                --v82;
                v24 = (__int64)v79;
                v78 = (_QWORD *)((char *)v78 + 1);
                break;
              }
            }
          }
LABEL_98:
          v56 = v31 + 1;
          if ( v31 + 1 == v29 )
            goto LABEL_25;
          v30 = *v56;
          ++v31;
          if ( (unsigned __int64)*v56 >= 0xFFFFFFFFFFFFFFFELL )
            break;
LABEL_102:
          if ( v29 == v31 )
            goto LABEL_25;
          if ( v55 != v24 )
            goto LABEL_97;
        }
        while ( v29 != ++v56 )
        {
          v30 = *v56;
          v31 = v56;
          if ( (unsigned __int64)*v56 < 0xFFFFFFFFFFFFFFFELL )
            goto LABEL_102;
        }
      }
    }
LABEL_25:
    v20 = (__int64)&v72;
    sub_39566E0(*(_QWORD *)(a1 + 56), (__int64)&v72, (__int64)v17, v22, (__int64)v23, v24);
    if ( byte_5055140 )
    {
      v20 = (unsigned __int8)byte_5055060;
      if ( !(unsigned __int8)sub_3968230(*(_QWORD *)(a1 + 56), byte_5055060) )
        sub_16BD130("Incorrect RP info from incremental RPA update in Rematerialization.\n", 1u);
    }
  }
  else
  {
    sub_3967F20(*(_QWORD *)(a1 + 56), v20);
    v66 = 1;
  }
LABEL_26:
  v32 = *(__int64 **)(a1 + 296);
  v33 = *(__int64 **)(a1 + 304);
  if ( v33 != v32 )
  {
    do
    {
      v34 = *v32++;
      sub_164BEC0(v34, v20, v21, v22, (__m128)0x3F800000u, a3, a4, a5, v25, v26, a8, a9);
    }
    while ( v33 != v32 );
    v35 = *(_QWORD *)(a1 + 296);
    if ( v35 != *(_QWORD *)(a1 + 304) )
      *(_QWORD *)(a1 + 304) = v35;
  }
  v36 = sub_3962650(*(_QWORD *)(a1 + 56), 1.0);
  v37 = (unsigned __int64)v80;
  if ( (_BYTE)v36 || HIBYTE(v36) )
  {
    if ( v80 == v79 )
      goto LABEL_42;
    goto LABEL_41;
  }
  if ( v80 != v79 )
    _libc_free((unsigned __int64)v80);
  if ( v74 != v73 )
    _libc_free((unsigned __int64)v74);
  v14 = (unsigned __int64)v69;
LABEL_37:
  if ( v14 )
    j_j___libc_free_0(v14);
  return v66;
}
