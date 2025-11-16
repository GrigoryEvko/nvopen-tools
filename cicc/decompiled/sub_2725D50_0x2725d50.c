// Function: sub_2725D50
// Address: 0x2725d50
//
__int64 __fastcall sub_2725D50(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rdx
  __int64 v8; // rcx
  int v9; // r11d
  unsigned int i; // eax
  __int64 v11; // r8
  unsigned int v12; // eax
  unsigned __int64 v13; // r13
  unsigned __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 *v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // r9
  int v25; // eax
  char v26; // r13
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  char v36; // [rsp+8h] [rbp-748h]
  unsigned __int64 v37; // [rsp+10h] [rbp-740h]
  void *v38; // [rsp+18h] [rbp-738h]
  __int64 v39; // [rsp+30h] [rbp-720h] BYREF
  __int64 *v40; // [rsp+38h] [rbp-718h]
  __int64 v41; // [rsp+40h] [rbp-710h]
  __int64 v42; // [rsp+48h] [rbp-708h]
  __int64 v43[2]; // [rsp+50h] [rbp-700h] BYREF
  __int64 v44; // [rsp+60h] [rbp-6F0h] BYREF
  __int64 *v45; // [rsp+68h] [rbp-6E8h]
  __int64 v46; // [rsp+70h] [rbp-6E0h]
  __int64 v47; // [rsp+78h] [rbp-6D8h] BYREF
  __int64 v48[2]; // [rsp+80h] [rbp-6D0h] BYREF
  unsigned int v49; // [rsp+90h] [rbp-6C0h]
  _BYTE *v50; // [rsp+98h] [rbp-6B8h]
  __int64 v51; // [rsp+A0h] [rbp-6B0h]
  _BYTE v52[1024]; // [rsp+A8h] [rbp-6A8h] BYREF
  __int64 v53; // [rsp+4A8h] [rbp-2A8h]
  char *v54; // [rsp+4B0h] [rbp-2A0h]
  __int64 v55; // [rsp+4B8h] [rbp-298h]
  int v56; // [rsp+4C0h] [rbp-290h]
  char v57; // [rsp+4C4h] [rbp-28Ch]
  char v58; // [rsp+4C8h] [rbp-288h] BYREF
  __int64 v59; // [rsp+5C8h] [rbp-188h]
  __int64 v60; // [rsp+5D0h] [rbp-180h]
  __int64 v61; // [rsp+5D8h] [rbp-178h]
  __int64 v62; // [rsp+5E0h] [rbp-170h]
  _BYTE *v63; // [rsp+5E8h] [rbp-168h]
  __int64 v64; // [rsp+5F0h] [rbp-160h]
  _BYTE v65[128]; // [rsp+5F8h] [rbp-158h] BYREF
  __int64 v66; // [rsp+678h] [rbp-D8h]
  char *v67; // [rsp+680h] [rbp-D0h]
  __int64 v68; // [rsp+688h] [rbp-C8h]
  int v69; // [rsp+690h] [rbp-C0h]
  char v70; // [rsp+694h] [rbp-BCh]
  char v71; // [rsp+698h] [rbp-B8h] BYREF

  v7 = *(unsigned int *)(a4 + 88);
  v8 = *(_QWORD *)(a4 + 72);
  if ( !(_DWORD)v7 )
    goto LABEL_31;
  v9 = 1;
  for ( i = (v7 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F81450 >> 9) ^ ((unsigned int)&unk_4F81450 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v7 - 1) & v12 )
  {
    v11 = v8 + 24LL * i;
    if ( *(_UNKNOWN **)v11 == &unk_4F81450 && a3 == *(_QWORD *)(v11 + 8) )
      break;
    if ( *(_QWORD *)v11 == -4096 && *(_QWORD *)(v11 + 8) == -4096 )
      goto LABEL_31;
    v12 = v9 + i;
    ++v9;
  }
  if ( v11 == v8 + 24 * v7 )
  {
LABEL_31:
    v13 = 0;
  }
  else
  {
    v13 = *(_QWORD *)(*(_QWORD *)(v11 + 16) + 24LL);
    if ( v13 )
      v13 += 8LL;
  }
  v39 = a3;
  v63 = v65;
  v41 = sub_BC1CD0(a4, &unk_4F8FBC8, a3) + 8;
  v51 = 0x8000000000LL;
  v54 = &v58;
  v64 = 0x1000000000LL;
  v67 = &v71;
  v40 = (__int64 *)v13;
  v42 = 0;
  v43[0] = 0;
  v43[1] = 0;
  LODWORD(v44) = 0;
  v45 = &v47;
  v46 = 0;
  v47 = 0;
  v48[0] = 0;
  v48[1] = 0;
  v49 = 0;
  v50 = v52;
  v53 = 0;
  v55 = 32;
  v56 = 0;
  v57 = 1;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v66 = 0;
  v68 = 16;
  v69 = 0;
  v70 = 1;
  sub_2723690((__int64)&v39, (__int64)&unk_4F8FBC8, (__int64)v65, v14, v15, v16);
  sub_2721090((__int64)&v39, (__int64)&unk_4F8FBC8, v17, v18, v19, v20);
  v25 = sub_2722E50((__int64)&v39, (__int64)&unk_4F8FBC8, v21, v22, v23, v24);
  v26 = v25;
  v36 = BYTE1(v25);
  v37 = (unsigned __int64)v25 >> 16;
  if ( !v70 )
    _libc_free((unsigned __int64)v67);
  if ( v63 != v65 )
    _libc_free((unsigned __int64)v63);
  sub_C7D6A0(v60, 8LL * (unsigned int)v62, 8);
  if ( !v57 )
    _libc_free((unsigned __int64)v54);
  if ( v50 != v52 )
    _libc_free((unsigned __int64)v50);
  sub_C7D6A0(v48[0], 24LL * v49, 8);
  if ( v45 != &v47 )
    _libc_free((unsigned __int64)v45);
  sub_C7D6A0(v43[0], 16LL * (unsigned int)v44, 8);
  v30 = a1 + 32;
  if ( v26 )
  {
    v39 = 0;
    v40 = v43;
    v41 = 2;
    LODWORD(v42) = 0;
    BYTE4(v42) = 1;
    v44 = 0;
    v45 = v48;
    v46 = 2;
    LODWORD(v47) = 0;
    BYTE4(v47) = 1;
    if ( !(_BYTE)v37 )
    {
      HIDWORD(v41) = 1;
      v39 = 1;
      v43[0] = (__int64)&unk_4F82408;
      if ( !v36 )
      {
        sub_271E7A0((__int64)&v39, (__int64)&unk_4F8F810, v27, v28, v29, v30);
        v30 = a1 + 32;
      }
    }
    v38 = (void *)v30;
    sub_271E7A0((__int64)&v39, (__int64)&unk_4F81450, v27, v28, v29, v30);
    sub_271E7A0((__int64)&v39, (__int64)&unk_4F8FBC8, v31, v32, v33, v34);
    sub_C8CF70(a1, v38, 2, (__int64)v43, (__int64)&v39);
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v48, (__int64)&v44);
    if ( !BYTE4(v47) )
      _libc_free((unsigned __int64)v45);
    if ( !BYTE4(v42) )
      _libc_free((unsigned __int64)v40);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v30;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  return a1;
}
