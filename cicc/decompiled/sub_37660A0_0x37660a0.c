// Function: sub_37660A0
// Address: 0x37660a0
//
unsigned __int8 *__fastcall sub_37660A0(
        __int64 *a1,
        __int64 a2,
        __m128i a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7)
{
  unsigned int v7; // ebx
  unsigned __int16 *v10; // rdx
  __int64 v11; // r12
  int v12; // eax
  __int64 v13; // rdx
  unsigned __int8 *result; // rax
  __int64 v15; // rdx
  __int64 v16; // rdx
  unsigned __int16 v17; // r12
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  _BYTE *v21; // rdi
  __int64 v22; // rcx
  unsigned int v23; // r8d
  char v24; // dl
  char v25; // dl
  char v26; // si
  __int64 v27; // rdx
  __int64 v28; // rdx
  char v29; // dl
  char v30; // dl
  unsigned __int8 v31; // al
  __int64 v32; // r12
  unsigned int v33; // r8d
  int v34; // r9d
  __int64 v35; // r11
  __int64 (*v36)(); // rax
  __int64 v37; // rax
  unsigned int v38; // r8d
  unsigned __int16 v39; // r9
  __int64 v40; // r11
  unsigned int v41; // r8d
  unsigned __int16 v42; // r9
  __int64 v43; // r11
  unsigned int v44; // r8d
  unsigned __int16 v45; // r9
  __int64 v46; // r11
  __int64 v47; // rsi
  __int64 v48; // rdi
  unsigned __int8 *v49; // rax
  __int64 v50; // r14
  __int64 v51; // rdx
  _QWORD *v52; // rdi
  _QWORD *v53; // rax
  __int64 v54; // rdx
  const void *v55; // r10
  __int64 v56; // r11
  int v57; // r9d
  int v58; // r9d
  __int64 v59; // rdx
  char v60; // al
  const void *v61; // [rsp+0h] [rbp-130h]
  __int64 v62; // [rsp+8h] [rbp-128h]
  __int64 *v63; // [rsp+10h] [rbp-120h]
  unsigned __int64 v64; // [rsp+10h] [rbp-120h]
  _QWORD *v65; // [rsp+10h] [rbp-120h]
  __int64 v66; // [rsp+18h] [rbp-118h]
  __int64 v67; // [rsp+18h] [rbp-118h]
  unsigned int v68; // [rsp+30h] [rbp-100h]
  __int64 v69; // [rsp+30h] [rbp-100h]
  unsigned __int8 *v70; // [rsp+30h] [rbp-100h]
  unsigned __int8 *v71; // [rsp+30h] [rbp-100h]
  int v72; // [rsp+30h] [rbp-100h]
  __int64 v73; // [rsp+38h] [rbp-F8h]
  unsigned int v74; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v75; // [rsp+78h] [rbp-B8h]
  __int64 v76; // [rsp+80h] [rbp-B0h] BYREF
  int v77; // [rsp+88h] [rbp-A8h]
  __int64 v78; // [rsp+90h] [rbp-A0h]
  __int64 v79; // [rsp+98h] [rbp-98h]
  __int64 v80; // [rsp+A0h] [rbp-90h] BYREF
  int v81; // [rsp+A8h] [rbp-88h]
  _BYTE *v82; // [rsp+B0h] [rbp-80h] BYREF
  __int64 v83; // [rsp+B8h] [rbp-78h]
  _BYTE v84[112]; // [rsp+C0h] [rbp-70h] BYREF

  v10 = *(unsigned __int16 **)(a2 + 48);
  v11 = a1[1];
  v12 = *v10;
  v13 = *((_QWORD *)v10 + 1);
  LOWORD(v74) = v12;
  v75 = v13;
  if ( (_WORD)v12 )
  {
    if ( (unsigned __int16)(v12 - 176) <= 0x34u )
      return sub_345E690(v11, a2, (_QWORD *)*a1);
    if ( (unsigned __int16)(v12 - 17) > 0xD3u )
    {
      v15 = 1;
      if ( (_WORD)v12 == 1 )
        goto LABEL_7;
LABEL_33:
      v15 = (unsigned __int16)v12;
      if ( !*(_QWORD *)(v11 + 8LL * (unsigned __int16)v12 + 112) )
        goto LABEL_11;
      goto LABEL_7;
    }
    LOWORD(v12) = word_4456580[v12 - 1];
  }
  else
  {
    if ( sub_3007100((__int64)&v74) )
      return sub_345E690(v11, a2, (_QWORD *)*a1);
    if ( !sub_30070B0((__int64)&v74) )
      goto LABEL_11;
    LOWORD(v12) = sub_3009970((__int64)&v74, a2, v16, a5, a6);
  }
  v15 = 1;
  if ( (_WORD)v12 != 1 )
  {
    if ( !(_WORD)v12 )
      goto LABEL_11;
    goto LABEL_33;
  }
LABEL_7:
  if ( (*(_BYTE *)(v11 + 500 * v15 + 6615) & 0xFB) == 0 )
    return 0;
LABEL_11:
  v17 = v74;
  if ( (_WORD)v74 )
  {
    if ( (unsigned __int16)(v74 - 17) > 0xD3u )
    {
LABEL_13:
      v18 = v75;
      goto LABEL_14;
    }
    v17 = word_4456580[(unsigned __int16)v74 - 1];
    v18 = 0;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v74) )
      goto LABEL_13;
    v17 = sub_3009970((__int64)&v74, a2, v28, a5, a6);
  }
LABEL_14:
  LOWORD(v82) = v17;
  v83 = v18;
  if ( v17 )
  {
    if ( v17 == 1 || (unsigned __int16)(v17 - 504) <= 7u )
      BUG();
    v19 = *(_QWORD *)&byte_444C4A0[16 * v17 - 16];
  }
  else
  {
    v19 = sub_3007260((__int64)&v82);
    v78 = v19;
    v79 = v20;
  }
  if ( (unsigned int)v19 <= 8 || (v19 & 7) != 0 )
  {
LABEL_18:
    v21 = (_BYTE *)a1[1];
    if ( (_WORD)v74 == 1 )
    {
      v29 = v21[7104];
      if ( v29 && v29 != 4 )
        return 0;
      v30 = v21[7106];
      if ( v30 )
      {
        if ( v30 != 4 )
          return 0;
      }
      v23 = 1;
      v27 = 1;
      v26 = v21[7100];
      if ( !v26 || v26 == 4 )
        goto LABEL_42;
    }
    else
    {
      if ( !(_WORD)v74 )
        return 0;
      v22 = (unsigned __int16)v74;
      if ( !*(_QWORD *)&v21[8 * (unsigned __int16)v74 + 112] )
        return 0;
      v23 = (unsigned __int16)v74;
      v24 = v21[500 * (unsigned __int16)v74 + 6604];
      if ( v24 )
      {
        if ( v24 != 4 || !*(_QWORD *)&v21[8 * (unsigned __int16)v74 + 112] )
          return 0;
      }
      v25 = v21[500 * (unsigned __int16)v74 + 6606];
      if ( v25 )
      {
        if ( v25 != 4 || !*(_QWORD *)&v21[8 * (unsigned __int16)v74 + 112] )
          return 0;
      }
      v26 = v21[500 * (unsigned __int16)v74 + 6600];
      if ( !v26 )
        goto LABEL_28;
      if ( v26 == 4 )
      {
LABEL_26:
        if ( (_WORD)v74 == 1 )
        {
LABEL_29:
          v27 = v23;
LABEL_42:
          v31 = v21[500 * v27 + 6601];
          if ( v31 <= 1u || v31 == 4 )
            return sub_345E690((__int64)v21, a2, (_QWORD *)*a1);
          return 0;
        }
        v22 = (unsigned __int16)v74;
LABEL_28:
        if ( !*(_QWORD *)&v21[8 * v22 + 112] )
          return 0;
        goto LABEL_29;
      }
    }
    if ( v26 != 1 )
      return 0;
    goto LABEL_26;
  }
  v32 = 0;
  v82 = v84;
  v83 = 0x1000000000LL;
  sub_3763720(v74, v75, (__int64)&v82, a5, a6, a7);
  v68 = v83;
  v63 = *(__int64 **)(*a1 + 64);
  v34 = sub_2D43050(5, v83);
  if ( !(_WORD)v34 )
  {
    v7 = sub_3009400(v63, 5, 0, v68, 0);
    v34 = v7;
    v32 = v59;
  }
  v35 = a1[1];
  LOWORD(v7) = v34;
  v36 = *(__int64 (**)())(*(_QWORD *)v35 + 624LL);
  if ( v36 != sub_2FE3180 )
  {
    v72 = v34;
    v60 = ((__int64 (__fastcall *)(__int64, _BYTE *, _QWORD, _QWORD, __int64))v36)(v35, v82, (unsigned int)v83, v7, v32);
    v34 = v72;
    if ( !v60 )
      goto LABEL_67;
    v35 = a1[1];
  }
  v37 = 1;
  if ( (_WORD)v34 != 1
    && (!(_WORD)v34 || (v37 = (unsigned __int16)v34, !*(_QWORD *)(v35 + 8LL * (unsigned __int16)v34 + 112)))
    || (*(_BYTE *)(v35 + 500 * v37 + 6615) & 0xFB) != 0 )
  {
    if ( !(unsigned __int8)sub_3763340(v35, 0xBEu, v34, 0, v33)
      || !(unsigned __int8)sub_3763340(v40, 0xC0u, v39, 0, v38)
      || !(unsigned __int8)sub_3763270(v43, 0xBAu, v42, 0, v41)
      || !(unsigned __int8)sub_3763270(v46, 0xBBu, v45, 0, v44) )
    {
LABEL_67:
      if ( v82 != v84 )
        _libc_free((unsigned __int64)v82);
      goto LABEL_18;
    }
  }
  v47 = *(_QWORD *)(a2 + 80);
  v76 = v47;
  if ( v47 )
    sub_B96E90((__int64)&v76, v47, 1);
  v48 = *a1;
  v77 = *(_DWORD *)(a2 + 72);
  v49 = sub_33FAF80(v48, 234, (__int64)&v76, v7, v32, v34, a3);
  v50 = *a1;
  v73 = v51;
  v52 = (_QWORD *)*a1;
  v64 = (unsigned __int64)v82;
  v66 = (unsigned int)v83;
  v69 = (__int64)v49;
  v80 = 0;
  v81 = 0;
  v53 = sub_33F17F0(v52, 51, (__int64)&v80, v7, v32);
  v55 = (const void *)v64;
  v56 = v66;
  if ( v80 )
  {
    v61 = (const void *)v64;
    v62 = v66;
    v65 = v53;
    v67 = v54;
    sub_B91220((__int64)&v80, v80);
    v55 = v61;
    v56 = v62;
    v53 = v65;
    v54 = v67;
  }
  sub_33FCE10(v50, v7, v32, (__int64)&v76, v69, v73, a3, (__int64)v53, v54, v55, v56);
  sub_33FAF80(*a1, 201, (__int64)&v76, v7, v32, v57, a3);
  result = sub_33FAF80(*a1, 234, (__int64)&v76, v74, v75, v58, a3);
  if ( v76 )
  {
    v70 = result;
    sub_B91220((__int64)&v76, v76);
    result = v70;
  }
  if ( v82 != v84 )
  {
    v71 = result;
    _libc_free((unsigned __int64)v82);
    return v71;
  }
  return result;
}
