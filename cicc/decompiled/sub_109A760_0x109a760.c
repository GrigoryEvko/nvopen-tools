// Function: sub_109A760
// Address: 0x109a760
//
__int64 __fastcall sub_109A760(__int64 a1, signed __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  char v7; // al
  __int64 v8; // r12
  __int64 v9; // rbx
  bool v10; // zf
  __int64 v11; // rax
  char *v12; // rdi
  const char *v13; // rbx
  const char *v14; // r12
  const char *v15; // rdi
  size_t v17; // r13
  unsigned __int64 v18; // r14
  __int64 v19; // r15
  char v20; // al
  size_t v21; // r12
  size_t v22; // r15
  char *v23; // rax
  __int64 v24; // r9
  char *v25; // r15
  size_t v26; // rsi
  char *v27; // rdx
  char v28; // al
  __int64 v29; // r9
  char *v30; // rdx
  char *v31; // rax
  __int64 v32; // rdx
  __int64 v33; // rdx
  __int64 v34; // rbx
  char v35; // al
  char v36; // dl
  char *v37; // rax
  unsigned __int64 *v38; // rbx
  unsigned __int64 *v39; // rbx
  unsigned __int64 v40; // rax
  __int64 v41; // [rsp+0h] [rbp-130h]
  signed __int64 v42; // [rsp+8h] [rbp-128h]
  __int64 v43; // [rsp+10h] [rbp-120h]
  char *v44; // [rsp+18h] [rbp-118h]
  char **v45; // [rsp+20h] [rbp-110h]
  __int64 *v46; // [rsp+28h] [rbp-108h]
  const char *v47; // [rsp+30h] [rbp-100h] BYREF
  __int64 v48; // [rsp+38h] [rbp-F8h]
  char *v49; // [rsp+40h] [rbp-F0h] BYREF
  __int64 v50; // [rsp+48h] [rbp-E8h]
  __int64 v51; // [rsp+50h] [rbp-E0h]
  char v52[8]; // [rsp+58h] [rbp-D8h] BYREF
  unsigned __int64 v53; // [rsp+60h] [rbp-D0h] BYREF
  unsigned int v54; // [rsp+68h] [rbp-C8h]
  char v55; // [rsp+70h] [rbp-C0h] BYREF
  int v56; // [rsp+A0h] [rbp-90h]
  char v57; // [rsp+A8h] [rbp-88h]
  __int64 v58; // [rsp+B0h] [rbp-80h] BYREF
  _QWORD v59[2]; // [rsp+B8h] [rbp-78h] BYREF
  _BYTE v60[48]; // [rsp+C8h] [rbp-68h] BYREF
  int v61; // [rsp+F8h] [rbp-38h]

  v45 = &v49;
  v47 = (const char *)&v49;
  v43 = a1;
  v48 = 0;
  v44 = v52;
  v49 = v52;
  v50 = 0;
  v51 = 0;
  if ( !a3 )
  {
    v6 = v43;
    v7 = *(_BYTE *)(v43 + 40);
    v8 = v43 + 16;
    *(_QWORD *)(v43 + 8) = 0;
    *(_QWORD *)v6 = v8;
    LOBYTE(v46) = v7;
    *(_BYTE *)(v6 + 40) = v7 & 0xFC | 2;
LABEL_3:
    v9 = v43;
    v10 = v50 == 0;
    v11 = v43 + 40;
    *(_QWORD *)(v43 + 24) = 0;
    *(_QWORD *)(v9 + 16) = v11;
    *(_QWORD *)(v9 + 32) = 0;
    if ( !v10 )
    {
      a2 = (signed __int64)v45;
      sub_1099150(v8, v45, a3, a4, a5, a6);
    }
LABEL_5:
    v12 = v49;
    if ( v49 != v44 )
LABEL_6:
      _libc_free(v12, a2);
    goto LABEL_7;
  }
  v17 = a3;
  v18 = a2;
  v19 = 0;
  sub_C8D290((__int64)v45, v52, a3, 1u, a5, a6);
  memcpy(&v49[v50], (const void *)a2, v17);
  v50 += v17;
  v46 = &v58;
  while ( 1 )
  {
    v20 = *(_BYTE *)(v18 + v19);
    v21 = v19 + 1;
    if ( v20 != 91 )
      break;
    v22 = v19 + 2;
    if ( v22 >= v17
      || (v23 = (char *)memchr((const void *)(v18 + v22), 93, v17 - v22)) == 0
      || (v25 = &v23[-v18], &v23[-v18] == (char *)-1LL) )
    {
      a2 = (signed __int64)"invalid glob pattern, unmatched '['";
      sub_10995A0(&v58, "invalid glob pattern, unmatched '['", 22, a4, a5);
      goto LABEL_64;
    }
    v26 = v21;
    if ( v17 <= v21 )
      v26 = v17;
    v27 = &v25[-v21];
    if ( (unsigned __int64)&v25[-v21] > v17 - v26 )
      v27 = (char *)(v17 - v26);
    v28 = *(_BYTE *)(v18 + v21);
    a2 = v18 + v26;
    if ( v28 == 33 || v28 == 94 )
    {
      if ( v27 )
      {
        --v27;
        ++a2;
      }
      sub_1099610((__int64)&v53, a2, v27, v18, v17, v24);
      v36 = v57 & 1;
      v57 = (2 * (v57 & 1)) | v57 & 0xFD;
      if ( v36 )
      {
LABEL_66:
        v39 = (unsigned __int64 *)v43;
        v40 = v53 & 0xFFFFFFFFFFFFFFFELL;
        *(_BYTE *)(v43 + 40) |= 3u;
        *v39 = v40;
        goto LABEL_5;
      }
      v37 = (char *)v53;
      v30 = (char *)(v53 + 8LL * v54);
      if ( (char *)v53 != v30 )
      {
        do
        {
          *(_QWORD *)v37 = ~*(_QWORD *)v37;
          v37 += 8;
        }
        while ( v37 != v30 );
      }
      if ( (v56 & 0x3F) != 0 )
      {
        a2 = v54;
        v30 = (char *)v53;
        *(_QWORD *)(v53 + 8LL * v54 - 8) &= ~(-1LL << (v56 & 0x3F));
      }
      v19 = (__int64)(v25 + 1);
      v58 = v19;
      if ( (v57 & 2) != 0 )
LABEL_56:
        sub_1099A00(&v53, a2);
    }
    else
    {
      sub_1099610((__int64)&v53, a2, v27, v18, v17, v24);
      v30 = (char *)(v57 & 1);
      v57 = (2 * (_BYTE)v30) | v57 & 0xFD;
      if ( (_BYTE)v30 )
        goto LABEL_66;
      v19 = (__int64)(v25 + 1);
      v58 = v19;
    }
    v59[0] = v60;
    v59[1] = 0x600000000LL;
    if ( v54 )
      sub_10992B0((__int64)v59, (char **)&v53, (__int64)v30, v54, a5, v29);
    a4 = (unsigned int)v48;
    a2 = (signed __int64)v47;
    a6 = (__int64)v46;
    v61 = v56;
    a3 = (unsigned int)v48;
    if ( (unsigned __int64)(unsigned int)v48 + 1 > HIDWORD(v48) )
    {
      if ( v47 > (const char *)v46 || v46 >= (__int64 *)&v47[80 * (unsigned int)v48] )
      {
        sub_F30F50((__int64)&v47, (unsigned int)v48 + 1LL, (unsigned int)v48, (unsigned int)v48, a5, (__int64)v46);
        a4 = (unsigned int)v48;
        a2 = (signed __int64)v47;
        a6 = (__int64)v46;
        a3 = (unsigned int)v48;
      }
      else
      {
        v42 = (char *)v46 - v47;
        sub_F30F50((__int64)&v47, (unsigned int)v48 + 1LL, (unsigned int)v48, (unsigned int)v48, a5, (__int64)v46);
        a2 = (signed __int64)v47;
        a4 = (unsigned int)v48;
        a6 = (__int64)&v47[v42];
        a3 = (unsigned int)v48;
      }
    }
    v31 = (char *)(a2 + 80 * a4);
    if ( v31 )
    {
      a4 = 0x600000000LL;
      v32 = *(_QWORD *)a6;
      *((_QWORD *)v31 + 2) = 0x600000000LL;
      *(_QWORD *)v31 = v32;
      *((_QWORD *)v31 + 1) = v31 + 24;
      v33 = *(unsigned int *)(a6 + 16);
      if ( (_DWORD)v33 )
      {
        a2 = a6 + 8;
        v41 = a6;
        v42 = (signed __int64)v31;
        sub_10992B0((__int64)(v31 + 8), (char **)(a6 + 8), v33, 0x600000000LL, a5, a6);
        a6 = v41;
        v31 = (char *)v42;
      }
      *((_DWORD *)v31 + 18) = *(_DWORD *)(a6 + 72);
      a3 = (unsigned int)v48;
    }
    LODWORD(v48) = a3 + 1;
    if ( (_BYTE *)v59[0] != v60 )
      _libc_free(v59[0], a2);
    if ( (v57 & 2) != 0 )
      goto LABEL_56;
    if ( (v57 & 1) != 0 )
    {
      if ( v53 )
        (*(void (__fastcall **)(unsigned __int64, signed __int64, __int64, __int64, __int64, __int64))(*(_QWORD *)v53 + 8LL))(
          v53,
          a2,
          a3,
          a4,
          a5,
          a6);
      goto LABEL_18;
    }
    if ( (char *)v53 == &v55 )
    {
LABEL_18:
      if ( v17 == v19 )
        goto LABEL_44;
    }
    else
    {
      _libc_free(v53, a2);
      if ( v17 == v19 )
      {
LABEL_44:
        v34 = v43;
        v35 = *(_BYTE *)(v43 + 40);
        v8 = v43 + 16;
        *(_QWORD *)(v43 + 8) = 0;
        *(_QWORD *)v34 = v8;
        LOBYTE(v46) = v35;
        *(_BYTE *)(v34 + 40) = v35 & 0xFC | 2;
        if ( (_DWORD)v48 )
        {
          a2 = (signed __int64)&v47;
          sub_109A3D0(v34, (__int64)&v47, a3, a4, a5, a6);
        }
        goto LABEL_3;
      }
    }
  }
  if ( v20 != 92 )
  {
    ++v19;
    goto LABEL_18;
  }
  if ( v17 != v21 )
  {
    v19 += 2;
    goto LABEL_18;
  }
  a2 = (signed __int64)"invalid glob pattern, stray '\\'";
  sub_1099530(&v58, "invalid glob pattern, stray '\\'", 22, a4, a5);
LABEL_64:
  v38 = (unsigned __int64 *)v43;
  *(_BYTE *)(v43 + 40) |= 3u;
  *v38 = v58 & 0xFFFFFFFFFFFFFFFELL;
  v12 = v49;
  if ( v49 != v44 )
    goto LABEL_6;
LABEL_7:
  v13 = v47;
  v14 = &v47[80 * (unsigned int)v48];
  if ( v47 != v14 )
  {
    do
    {
      v14 -= 80;
      v15 = (const char *)*((_QWORD *)v14 + 1);
      if ( v15 != v14 + 24 )
        _libc_free(v15, a2);
    }
    while ( v13 != v14 );
    v14 = v47;
  }
  if ( v14 != (const char *)v45 )
    _libc_free(v14, a2);
  return v43;
}
