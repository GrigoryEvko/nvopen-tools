// Function: sub_185FD30
// Address: 0x185fd30
//
__int64 __fastcall sub_185FD30(
        __int64 a1,
        __int64 a2,
        _BYTE *a3,
        __int64 a4,
        __m128 a5,
        double a6,
        double a7,
        double a8,
        double a9,
        double a10,
        double a11,
        __m128 a12)
{
  __int64 v12; // r13
  __int64 v15; // rax
  signed __int64 v16; // r14
  _QWORD *v17; // rbx
  _QWORD *v18; // rax
  _QWORD *v19; // rdi
  int v20; // ebx
  __int64 v21; // rax
  int v22; // eax
  _QWORD *v23; // rdx
  _QWORD *v24; // rdi
  __int64 v25; // rax
  __int64 v26; // r14
  unsigned __int8 v27; // al
  __int64 v28; // rax
  __int64 v29; // rdx
  bool v30; // dl
  __int64 v31; // rsi
  __int64 v32; // rsi
  _QWORD *v33; // rbx
  __int64 v34; // rax
  __int64 v36; // rax
  __int64 v37; // rdx
  __int64 v38; // rcx
  bool v39; // [rsp+0h] [rbp-140h]
  unsigned __int8 v40; // [rsp+0h] [rbp-140h]
  unsigned __int64 v43[2]; // [rsp+20h] [rbp-120h] BYREF
  __int64 v44; // [rsp+30h] [rbp-110h]
  _BYTE *v45; // [rsp+40h] [rbp-100h] BYREF
  __int64 v46; // [rsp+48h] [rbp-F8h]
  _BYTE v47[240]; // [rsp+50h] [rbp-F0h] BYREF

  v12 = *(_QWORD *)(a1 + 8);
  v45 = v47;
  v46 = 0x800000000LL;
  if ( !v12 )
    return 0;
  v15 = v12;
  v16 = 0;
  do
  {
    v15 = *(_QWORD *)(v15 + 8);
    ++v16;
  }
  while ( v15 );
  v17 = v47;
  if ( v16 > 8 )
  {
    sub_170B450((__int64)&v45, v16);
    v17 = &v45[24 * (unsigned int)v46];
  }
  do
  {
    if ( v17 )
    {
      v18 = sub_1648700(v12);
      *v17 = 6;
      v17[1] = 0;
      v17[2] = v18;
      if ( v18 + 1 != 0 && v18 != 0 && v18 != (_QWORD *)-16LL )
        sub_164C220((__int64)v17);
    }
    v12 = *(_QWORD *)(v12 + 8);
    v17 += 3;
  }
  while ( v12 );
  v19 = v45;
  v20 = 0;
  LODWORD(v46) = v46 + v16;
  v21 = (unsigned int)v46;
  if ( !(_DWORD)v46 )
  {
LABEL_74:
    v40 = v20;
    goto LABEL_60;
  }
  while ( 1 )
  {
    v43[0] = 6;
    v43[1] = 0;
    v23 = &v19[3 * v21 - 3];
    v44 = v23[2];
    if ( v44 != 0 && v44 != -8 && v44 != -16 )
    {
      sub_1649AC0(v43, *v23 & 0xFFFFFFFFFFFFFFF8LL);
      v19 = v45;
    }
    LODWORD(v46) = v46 - 1;
    v24 = &v19[3 * (unsigned int)v46];
    v25 = v24[2];
    if ( v25 != 0 && v25 != -8 && v25 != -16 )
      sub_1649B30(v24);
    v26 = v44;
    if ( !v44 )
      goto LABEL_18;
    if ( v44 != -8 && v44 != -16 )
      sub_1649B30(v43);
    v27 = *(_BYTE *)(v26 + 16);
    if ( v27 <= 0x17u )
      break;
    if ( v27 == 54 )
    {
      if ( a2 )
      {
        v20 = 1;
        sub_164D160(v26, a2, a5, a6, a7, a8, a9, a10, a11, a12);
        sub_15F20C0((_QWORD *)v26);
      }
      goto LABEL_18;
    }
    if ( v27 == 55 )
    {
LABEL_45:
      v20 = 1;
      sub_15F20C0((_QWORD *)v26);
      goto LABEL_18;
    }
    if ( v27 != 56 )
    {
      if ( v27 == 78 )
      {
        v28 = *(_QWORD *)(v26 - 24);
        if ( !*(_BYTE *)(v28 + 16)
          && (*(_BYTE *)(v28 + 33) & 0x20) != 0
          && (unsigned int)(*(_DWORD *)(v28 + 36) - 133) <= 4
          && ((1LL << (*(_BYTE *)(v28 + 36) + 123)) & 0x15) != 0 )
        {
          v29 = *(_QWORD *)(v26 - 24LL * (*(_DWORD *)(v26 + 20) & 0xFFFFFFF));
          v30 = a1 == v29 && v29 != 0;
          if ( v30 )
          {
            v39 = v30;
            sub_15F20C0((_QWORD *)v26);
            v20 = v39;
          }
        }
      }
      goto LABEL_18;
    }
    if ( *(_BYTE *)(*(_QWORD *)(v26 - 24LL * (*(_DWORD *)(v26 + 20) & 0xFFFFFFF)) + 16LL) == 5 )
    {
LABEL_43:
      v31 = 0;
      goto LABEL_44;
    }
    v36 = sub_14DD210((__int64 *)v26, a3, a4);
    if ( v36 && *(_BYTE *)(v36 + 16) == 5 )
    {
      if ( !a2 )
        goto LABEL_43;
      v31 = 0;
      if ( *(_WORD *)(v36 + 18) == 32 )
        v31 = sub_14D81F0(a2, v36);
    }
    else
    {
      if ( !a2 )
        goto LABEL_43;
      v31 = 0;
    }
    if ( *(_BYTE *)(a2 + 16) == 10 && sub_15FA300(v26) )
      v31 = sub_15A06D0(*(__int64 ***)(v26 + 64), v31, v37, v38);
LABEL_44:
    v20 |= sub_185FD30(v26, v31, a3, a4);
    if ( !*(_QWORD *)(v26 + 8) )
      goto LABEL_45;
LABEL_18:
    v21 = (unsigned int)v46;
    v19 = v45;
    if ( !(_DWORD)v46 )
      goto LABEL_74;
  }
  if ( v27 == 5 )
  {
    v22 = *(unsigned __int16 *)(v26 + 18);
    switch ( v22 )
    {
      case ' ':
        v32 = 0;
        if ( a2 )
          v32 = sub_14D81F0(a2, v26);
        v20 |= sub_185FD30(v26, v32, a3, a4);
        if ( *(_QWORD *)(v26 + 8) )
          goto LABEL_18;
LABEL_49:
        v20 = 1;
        sub_159D850(v26);
        goto LABEL_18;
      case '/':
        if ( *(_BYTE *)(*(_QWORD *)v26 + 8LL) == 15 )
          goto LABEL_51;
        break;
      case '0':
LABEL_51:
        v20 |= sub_185FD30(v26, 0, a3, a4);
        break;
      default:
        break;
    }
    if ( !*(_QWORD *)(v26 + 8) )
      goto LABEL_49;
    goto LABEL_18;
  }
  if ( v27 > 0x10u )
    goto LABEL_18;
  v40 = sub_1ACF050(v26);
  if ( !v40 )
    goto LABEL_18;
  sub_159D850(v26);
  sub_185FD30(a1, a2, a3, a4);
  v33 = v45;
  v19 = &v45[24 * (unsigned int)v46];
  if ( v45 != (_BYTE *)v19 )
  {
    do
    {
      v34 = *(v19 - 1);
      v19 -= 3;
      if ( v34 != -8 && v34 != 0 && v34 != -16 )
        sub_1649B30(v19);
    }
    while ( v33 != v19 );
    v19 = v45;
  }
LABEL_60:
  if ( v19 != (_QWORD *)v47 )
    _libc_free((unsigned __int64)v19);
  return v40;
}
