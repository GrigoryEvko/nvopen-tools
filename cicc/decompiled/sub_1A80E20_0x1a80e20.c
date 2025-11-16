// Function: sub_1A80E20
// Address: 0x1a80e20
//
__int64 __fastcall sub_1A80E20(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 v10; // r14
  int v12; // r8d
  int v13; // r9d
  __int64 v14; // r13
  __int64 v15; // rbx
  signed __int64 v16; // rbx
  _QWORD *v17; // r13
  _QWORD *v18; // rax
  int v19; // ebx
  __int64 *v20; // rax
  __int64 v21; // rdx
  int v22; // r8d
  int v23; // r9d
  double v24; // xmm4_8
  double v25; // xmm5_8
  __int64 v26; // rcx
  unsigned __int64 v27; // rdi
  int v28; // eax
  __int64 v29; // rcx
  __int64 *v30; // rdx
  __int64 v31; // rax
  __int64 v32; // r14
  __int64 v33; // rax
  __int64 v34; // r15
  __int64 v35; // rax
  __int64 v36; // r15
  __int64 *v37; // rax
  __int64 *v38; // rsi
  __int64 *v39; // rcx
  _BYTE *v40; // rbx
  unsigned int v41; // r13d
  __int64 v42; // rcx
  char *v43; // rax
  __int64 v44; // rdx
  char *v45; // rsi
  __int64 v46; // rdx
  unsigned __int64 v47; // r12
  unsigned __int64 v48; // rdi
  unsigned __int64 v51; // [rsp+28h] [rbp-188h]
  _QWORD *v52; // [rsp+30h] [rbp-180h]
  __int64 v53; // [rsp+38h] [rbp-178h]
  _QWORD *v54; // [rsp+50h] [rbp-160h]
  __int64 v55; // [rsp+58h] [rbp-158h]
  _QWORD *v56; // [rsp+60h] [rbp-150h] BYREF
  __int64 v57; // [rsp+68h] [rbp-148h]
  _QWORD v58[2]; // [rsp+70h] [rbp-140h] BYREF
  _BYTE *v59; // [rsp+80h] [rbp-130h] BYREF
  __int64 v60; // [rsp+88h] [rbp-128h]
  _BYTE v61[32]; // [rsp+90h] [rbp-120h] BYREF
  __int64 v62; // [rsp+B0h] [rbp-100h] BYREF
  __int64 *v63; // [rsp+B8h] [rbp-F8h] BYREF
  __int64 v64; // [rsp+C0h] [rbp-F0h]
  __int64 v65; // [rsp+C8h] [rbp-E8h] BYREF
  int v66; // [rsp+D0h] [rbp-E0h]
  _BYTE v67[40]; // [rsp+D8h] [rbp-D8h] BYREF
  _BYTE *v68; // [rsp+100h] [rbp-B0h] BYREF
  __int64 v69; // [rsp+108h] [rbp-A8h]
  _BYTE v70[160]; // [rsp+110h] [rbp-A0h] BYREF

  v51 = a1 & 0xFFFFFFFFFFFFFFF8LL;
  v10 = *(_QWORD *)(*(_QWORD *)((a1 & 0xFFFFFFFFFFFFFFF8LL) + 40) + 8LL);
  if ( !v10 )
  {
LABEL_65:
    v56 = v58;
    v57 = 0x200000000LL;
    if ( v58[1] != v58[0] )
    {
      v40 = v70;
      v69 = 0x200000000LL;
      v41 = 0;
      v42 = 0;
      v68 = v70;
      goto LABEL_78;
    }
    return 0;
  }
  while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v10) + 16) - 25) > 9u )
  {
    v10 = *(_QWORD *)(v10 + 8);
    if ( !v10 )
      goto LABEL_65;
  }
  v14 = v10;
  v15 = 0;
  v56 = v58;
  v57 = 0x200000000LL;
  while ( 1 )
  {
    v14 = *(_QWORD *)(v14 + 8);
    if ( !v14 )
      break;
    while ( (unsigned __int8)(*((_BYTE *)sub_1648700(v14) + 16) - 25) <= 9u )
    {
      v14 = *(_QWORD *)(v14 + 8);
      ++v15;
      if ( !v14 )
        goto LABEL_7;
    }
  }
LABEL_7:
  v16 = v15 + 1;
  if ( v16 > 2 )
  {
    sub_16CD150((__int64)&v56, v58, v16, 8, v12, v13);
    v17 = &v56[(unsigned int)v57];
  }
  else
  {
    v17 = v58;
  }
  v18 = sub_1648700(v10);
LABEL_12:
  if ( v17 )
    *v17 = v18[5];
  while ( 1 )
  {
    v10 = *(_QWORD *)(v10 + 8);
    if ( !v10 )
      break;
    v18 = sub_1648700(v10);
    if ( (unsigned __int8)(*((_BYTE *)v18 + 16) - 25) <= 9u )
    {
      ++v17;
      goto LABEL_12;
    }
  }
  v52 = v56;
  v19 = v57 + v16;
  LODWORD(v57) = v19;
  if ( *v56 == v56[1] )
  {
    v41 = 0;
    goto LABEL_86;
  }
  v68 = v70;
  v69 = 0x200000000LL;
  v54 = &v56[v19];
  if ( v54 == v56 )
  {
    v40 = v70;
    v42 = (unsigned int)v69;
    v41 = 0;
    goto LABEL_78;
  }
  while ( 2 )
  {
    v31 = *(v54 - 1);
    v59 = v61;
    v60 = 0x200000000LL;
    v53 = v31;
    v32 = v31;
    sub_1A7F010(a1, v31, *(_QWORD *)(v51 + 40), (__int64)&v59);
    v62 = 0;
    v65 = 4;
    v63 = (__int64 *)v67;
    v64 = (__int64)v67;
    v66 = 0;
    while ( 1 )
    {
      v33 = sub_157F0B0(v32);
      v27 = v64;
      v34 = v33;
      v20 = v63;
      if ( (__int64 *)v64 == v63 )
      {
        v26 = v64 + 8LL * HIDWORD(v65);
        if ( v64 == v26 )
        {
          v21 = v64;
        }
        else
        {
          do
          {
            if ( v34 == *v20 )
              break;
            ++v20;
          }
          while ( (__int64 *)v26 != v20 );
          v21 = v64 + 8LL * HIDWORD(v65);
        }
LABEL_44:
        while ( (__int64 *)v21 != v20 )
        {
          if ( (unsigned __int64)*v20 < 0xFFFFFFFFFFFFFFFELL )
            goto LABEL_21;
          ++v20;
        }
        if ( (__int64 *)v26 != v20 )
          goto LABEL_22;
      }
      else
      {
        v55 = v64 + 8LL * (unsigned int)v65;
        v20 = sub_16CC9F0((__int64)&v62, v34);
        v26 = v55;
        if ( v34 == *v20 )
        {
          v27 = v64;
          if ( (__int64 *)v64 == v63 )
            v21 = v64 + 8LL * HIDWORD(v65);
          else
            v21 = v64 + 8LL * (unsigned int)v65;
          goto LABEL_44;
        }
        v27 = v64;
        if ( (__int64 *)v64 == v63 )
        {
          v20 = (__int64 *)(v64 + 8LL * HIDWORD(v65));
          v21 = (__int64)v20;
          goto LABEL_44;
        }
        v20 = (__int64 *)(v64 + 8LL * (unsigned int)v65);
LABEL_21:
        if ( (__int64 *)v26 != v20 )
          goto LABEL_22;
      }
      v35 = sub_157F0B0(v32);
      v36 = v35;
      if ( !v35 )
        break;
      sub_1A7F010(a1, v35, v32, (__int64)&v59);
      v37 = v63;
      if ( (__int64 *)v64 != v63 )
        goto LABEL_48;
      v38 = &v63[HIDWORD(v65)];
      if ( v63 == v38 )
      {
LABEL_61:
        if ( HIDWORD(v65) >= (unsigned int)v65 )
        {
LABEL_48:
          sub_16CCBA0((__int64)&v62, v36);
          goto LABEL_49;
        }
        ++HIDWORD(v65);
        *v38 = v36;
        ++v62;
      }
      else
      {
        v39 = 0;
        while ( v36 != *v37 )
        {
          if ( *v37 == -2 )
            v39 = v37;
          if ( v38 == ++v37 )
          {
            if ( !v39 )
              goto LABEL_61;
            *v39 = v36;
            --v66;
            ++v62;
            break;
          }
        }
      }
LABEL_49:
      v32 = v36;
    }
    v27 = v64;
LABEL_22:
    if ( (__int64 *)v27 != v63 )
      _libc_free(v27);
    v62 = v53;
    v63 = &v65;
    v64 = 0x200000000LL;
    if ( (_DWORD)v60 )
    {
      sub_1A7EC10((__int64)&v63, (__int64)&v59, v21, v26, v22, v23);
      v28 = v69;
      if ( (unsigned int)v69 >= HIDWORD(v69) )
        goto LABEL_68;
    }
    else
    {
      v28 = v69;
      if ( (unsigned int)v69 < HIDWORD(v69) )
        goto LABEL_26;
LABEL_68:
      sub_1A7F1E0((__int64)&v68, 0);
      v28 = v69;
    }
LABEL_26:
    v29 = (__int64)v68;
    v30 = (__int64 *)&v68[56 * v28];
    if ( v30 )
    {
      *v30 = v62;
      v30[1] = (__int64)(v30 + 3);
      v30[2] = 0x200000000LL;
      if ( (_DWORD)v64 )
        sub_1A7EAB0((__int64)(v30 + 1), (__int64)&v63, (__int64)v30, v29, v22, v23);
      v28 = v69;
    }
    LODWORD(v69) = v28 + 1;
    if ( v63 != &v65 )
      _libc_free((unsigned __int64)v63);
    if ( v59 != v61 )
      _libc_free((unsigned __int64)v59);
    if ( --v54 != v52 )
      continue;
    break;
  }
  v42 = (unsigned int)v69;
  v40 = v68;
  v43 = v68;
  v44 = 56LL * (unsigned int)v69;
  v45 = &v68[v44];
  v46 = 0x6DB6DB6DB6DB6DB7LL * (v44 >> 3);
  if ( !(v46 >> 2) )
  {
LABEL_98:
    if ( v46 != 2 )
    {
      if ( v46 != 3 )
      {
        if ( v46 != 1 )
          goto LABEL_92;
        goto LABEL_91;
      }
      if ( *((_DWORD *)v43 + 4) )
        goto LABEL_76;
      v43 += 56;
    }
    if ( *((_DWORD *)v43 + 4) )
      goto LABEL_76;
    v43 += 56;
LABEL_91:
    if ( *((_DWORD *)v43 + 4) )
      goto LABEL_76;
    goto LABEL_92;
  }
  while ( !*((_DWORD *)v43 + 4) )
  {
    if ( *((_DWORD *)v43 + 18) )
    {
      v43 += 56;
      break;
    }
    if ( *((_DWORD *)v43 + 32) )
    {
      v43 += 112;
      break;
    }
    if ( *((_DWORD *)v43 + 46) )
    {
      v43 += 168;
      break;
    }
    v43 += 224;
    if ( v43 == &v68[224 * (v46 >> 2)] )
    {
      v46 = 0x6DB6DB6DB6DB6DB7LL * ((v45 - v43) >> 3);
      goto LABEL_98;
    }
  }
LABEL_76:
  if ( v45 != v43 )
  {
    v41 = 1;
    sub_1A7F3A0(a1, (__int64)&v68, a2, a3, a4, a5, a6, v24, v25, a9, a10);
    v40 = v68;
    v42 = (unsigned int)v69;
    goto LABEL_78;
  }
LABEL_92:
  v41 = 0;
LABEL_78:
  v47 = (unsigned __int64)&v40[56 * v42];
  if ( v40 != (_BYTE *)v47 )
  {
    do
    {
      v47 -= 56LL;
      v48 = *(_QWORD *)(v47 + 8);
      if ( v48 != v47 + 24 )
        _libc_free(v48);
    }
    while ( v40 != (_BYTE *)v47 );
    v47 = (unsigned __int64)v68;
  }
  if ( (_BYTE *)v47 != v70 )
    _libc_free(v47);
  v52 = v56;
LABEL_86:
  if ( v52 != v58 )
    _libc_free((unsigned __int64)v52);
  return v41;
}
