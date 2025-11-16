// Function: sub_25E0FF0
// Address: 0x25e0ff0
//
__int64 __fastcall sub_25E0FF0(__int64 a1, _BYTE *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int8 *v6; // rax
  __int64 v7; // r14
  __int64 v8; // rax
  signed __int64 v9; // rbx
  _BYTE *v10; // rax
  __int64 v11; // rax
  unsigned __int8 *v12; // r15
  __int64 v13; // rcx
  unsigned __int8 *v14; // r8
  char v15; // dl
  __int64 v16; // rdx
  __int64 v17; // rdx
  __int16 v18; // ax
  char v19; // al
  _QWORD *v20; // rbx
  unsigned int v21; // r14d
  _QWORD *v22; // r12
  __int64 v23; // rax
  __int64 v25; // rsi
  unsigned int v26; // eax
  unsigned __int8 *v27; // r11
  unsigned __int8 *v28; // rax
  __int64 v29; // rdx
  __int64 v30; // rcx
  __int64 v31; // r8
  __int64 v32; // rsi
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // rdx
  __int64 v37; // [rsp+0h] [rbp-1A0h]
  __int64 v38; // [rsp+8h] [rbp-198h]
  unsigned __int8 *v39; // [rsp+20h] [rbp-180h]
  unsigned __int8 v41; // [rsp+4Fh] [rbp-151h] BYREF
  __int64 v42[2]; // [rsp+50h] [rbp-150h] BYREF
  unsigned __int64 v43; // [rsp+60h] [rbp-140h] BYREF
  unsigned int v44; // [rsp+68h] [rbp-138h]
  void (__fastcall *v45)(unsigned __int64 *, unsigned __int64 *, __int64); // [rsp+70h] [rbp-130h]
  _BYTE *v46; // [rsp+80h] [rbp-120h] BYREF
  __int64 v47; // [rsp+88h] [rbp-118h]
  _BYTE v48[48]; // [rsp+90h] [rbp-110h] BYREF
  _BYTE *v49; // [rsp+C0h] [rbp-E0h] BYREF
  __int64 v50; // [rsp+C8h] [rbp-D8h]
  _BYTE v51[64]; // [rsp+D0h] [rbp-D0h] BYREF
  __int64 v52; // [rsp+110h] [rbp-90h] BYREF
  char *v53; // [rsp+118h] [rbp-88h]
  __int64 v54; // [rsp+120h] [rbp-80h]
  int v55; // [rsp+128h] [rbp-78h]
  char v56; // [rsp+12Ch] [rbp-74h]
  char v57; // [rsp+130h] [rbp-70h] BYREF

  v6 = *(unsigned __int8 **)(a1 - 32);
  v7 = *(_QWORD *)(a1 + 16);
  v49 = v51;
  v39 = v6;
  v50 = 0x800000000LL;
  if ( v7 )
  {
    v8 = v7;
    v9 = 0;
    do
    {
      v8 = *(_QWORD *)(v8 + 8);
      ++v9;
    }
    while ( v8 );
    v10 = v51;
    if ( v9 > 8 )
    {
      sub_C8D5F0((__int64)&v49, v51, v9, 8u, a5, a6);
      v10 = &v49[8 * (unsigned int)v50];
    }
    do
    {
      v10 += 8;
      *((_QWORD *)v10 - 1) = *(_QWORD *)(v7 + 24);
      v7 = *(_QWORD *)(v7 + 8);
    }
    while ( v7 );
    LODWORD(v11) = v9 + v50;
  }
  else
  {
    LODWORD(v11) = 0;
  }
  LODWORD(v50) = v11;
  v53 = &v57;
  v54 = 8;
  v52 = 0;
  v55 = 0;
  v56 = 1;
  v41 = 0;
  v47 = 0x200000000LL;
  v42[0] = (__int64)&v46;
  v42[1] = (__int64)&v41;
  v46 = v48;
LABEL_9:
  while ( (_DWORD)v11 )
  {
    while ( 1 )
    {
      v12 = *(unsigned __int8 **)&v49[8 * (unsigned int)v11 - 8];
      LODWORD(v50) = v11 - 1;
      sub_AE6EC0((__int64)&v52, (__int64)v12);
      if ( !v15 )
        goto LABEL_35;
      v16 = *v12;
      if ( (unsigned __int8)v16 <= 0x1Cu )
      {
        if ( (_BYTE)v16 != 5 )
          goto LABEL_35;
        v18 = *((_WORD *)v12 + 1);
        if ( v18 != 49 )
          goto LABEL_32;
      }
      else if ( (_BYTE)v16 != 78 )
      {
        goto LABEL_13;
      }
      sub_25DC650((__int64)&v49, &v49[8 * (unsigned int)v50], *((_QWORD *)v12 + 2), 0);
      v16 = *v12;
      if ( (unsigned __int8)v16 <= 0x1Cu )
      {
        LODWORD(v11) = v50;
        if ( (_BYTE)v16 != 5 )
          goto LABEL_9;
        v18 = *((_WORD *)v12 + 1);
LABEL_32:
        if ( v18 != 50 && v18 != 34 )
          goto LABEL_35;
LABEL_34:
        sub_25DC650((__int64)&v49, &v49[8 * (unsigned int)v50], *((_QWORD *)v12 + 2), 0);
        goto LABEL_35;
      }
LABEL_13:
      if ( (_BYTE)v16 == 79 || (_BYTE)v16 == 63 )
        goto LABEL_34;
      if ( (_BYTE)v16 != 61 )
        break;
      v38 = *((_QWORD *)v12 + 1);
      v25 = sub_96E500(v39, v38, (__int64)a2);
      if ( v25 )
      {
        sub_BD84D0((__int64)v12, v25);
        sub_25E0E10(v42, (__int64)v12, v29, v30, v31);
        LODWORD(v11) = v50;
        goto LABEL_9;
      }
      v37 = *((_QWORD *)v12 - 4);
      v26 = sub_AE43F0((__int64)a2, *(_QWORD *)(v37 + 8));
      v27 = (unsigned __int8 *)v37;
      v44 = v26;
      if ( v26 > 0x40 )
      {
        sub_C43690((__int64)&v43, 0, 0);
        v27 = (unsigned __int8 *)v37;
      }
      else
      {
        v43 = 0;
      }
      v28 = sub_BD45C0(v27, (__int64)a2, (__int64)&v43, 1, 0, 0, 0, 0);
      if ( (*v28 != 85
         || (v36 = *((_QWORD *)v28 - 4)) == 0
         || *(_BYTE *)v36
         || *(_QWORD *)(v36 + 24) != *((_QWORD *)v28 + 10)
         || (*(_BYTE *)(v36 + 33) & 0x20) == 0
         || *(_DWORD *)(v36 + 36) != 353
         || (v28 = *(unsigned __int8 **)&v28[-32 * (*((_DWORD *)v28 + 1) & 0x7FFFFFF)]) != 0)
        && v28 == (unsigned __int8 *)a1 )
      {
        v32 = sub_9714E0((__int64)v39, v38, (__int64)&v43, a2);
        if ( v32 )
        {
          sub_BD84D0((__int64)v12, v32);
          sub_25E0E10(v42, (__int64)v12, v33, v34, v35);
        }
      }
      if ( v44 > 0x40 && v43 )
      {
        j_j___libc_free_0_0(v43);
        LODWORD(v11) = v50;
        goto LABEL_9;
      }
LABEL_35:
      LODWORD(v11) = v50;
      if ( !(_DWORD)v50 )
        goto LABEL_36;
    }
    if ( (_BYTE)v16 == 62 )
    {
LABEL_55:
      sub_25E0E10(v42, (__int64)v12, v16, v13, (__int64)v14);
      LODWORD(v11) = v50;
      continue;
    }
    LODWORD(v11) = v50;
    if ( (_BYTE)v16 != 85 )
      continue;
    v17 = *((_QWORD *)v12 - 4);
    if ( v17 && !*(_BYTE *)v17 && *(_QWORD *)(v17 + 24) == *((_QWORD *)v12 + 10) && (*(_BYTE *)(v17 + 33) & 0x20) != 0 )
    {
      if ( (unsigned int)(*(_DWORD *)(v17 + 36) - 238) <= 7 && ((1LL << (*(_BYTE *)(v17 + 36) + 18)) & 0xAD) != 0 )
      {
        v14 = sub_98ACB0(*(unsigned __int8 **)&v12[-32 * (*((_DWORD *)v12 + 1) & 0x7FFFFFF)], 6u);
        LODWORD(v11) = v50;
        if ( (unsigned __int8 *)a1 != v14 )
          continue;
        goto LABEL_55;
      }
      v11 = (unsigned int)v50;
    }
    else
    {
      v11 = (unsigned int)v50;
      if ( !v17 )
        continue;
    }
    if ( !*(_BYTE *)v17
      && *(_QWORD *)(v17 + 24) == *((_QWORD *)v12 + 10)
      && (*(_BYTE *)(v17 + 33) & 0x20) != 0
      && *(_DWORD *)(v17 + 36) == 353 )
    {
      sub_25DC650((__int64)&v49, &v49[8 * v11], *((_QWORD *)v12 + 2), 0);
      LODWORD(v11) = v50;
    }
  }
LABEL_36:
  v45 = 0;
  v19 = sub_F5C6D0((__int64)&v46, 0, 0, (__int64)&v43);
  v41 |= v19;
  if ( v45 )
    v45(&v43, &v43, 3);
  sub_AD0030(a1);
  v20 = v46;
  v21 = v41;
  v22 = &v46[24 * (unsigned int)v47];
  if ( v46 != (_BYTE *)v22 )
  {
    do
    {
      v23 = *(v22 - 1);
      v22 -= 3;
      if ( v23 != 0 && v23 != -4096 && v23 != -8192 )
        sub_BD60C0(v22);
    }
    while ( v20 != v22 );
    v22 = v46;
  }
  if ( v22 != (_QWORD *)v48 )
    _libc_free((unsigned __int64)v22);
  if ( !v56 )
    _libc_free((unsigned __int64)v53);
  if ( v49 != v51 )
    _libc_free((unsigned __int64)v49);
  return v21;
}
