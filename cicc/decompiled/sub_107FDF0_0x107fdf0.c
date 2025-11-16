// Function: sub_107FDF0
// Address: 0x107fdf0
//
_DWORD *__fastcall sub_107FDF0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // r13
  __int64 v9; // r15
  __int64 v10; // rdx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // r13
  __int64 v15; // r9
  char **v16; // rsi
  char v17; // al
  __int64 v18; // r8
  __int64 v19; // r9
  __int64 v20; // r13
  char v21; // r15
  int v22; // r13d
  _DWORD *result; // rax
  __int64 v24; // r15
  unsigned __int64 v25; // rcx
  char *v26; // rdx
  unsigned __int64 v27; // rsi
  int v28; // eax
  _QWORD *v29; // r15
  __int64 v30; // rcx
  unsigned int v31; // esi
  int v32; // eax
  int v33; // eax
  __int64 v34; // r8
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v41; // rdi
  char *v42; // r15
  char *v43; // [rsp+38h] [rbp-128h]
  char *v44; // [rsp+38h] [rbp-128h]
  __int64 v45; // [rsp+40h] [rbp-120h] BYREF
  __int64 v46; // [rsp+48h] [rbp-118h] BYREF
  char *v47; // [rsp+50h] [rbp-110h] BYREF
  __int64 v48; // [rsp+58h] [rbp-108h]
  char v49; // [rsp+60h] [rbp-100h] BYREF
  _BYTE *v50; // [rsp+68h] [rbp-F8h] BYREF
  __int64 v51; // [rsp+70h] [rbp-F0h]
  _BYTE v52[16]; // [rsp+78h] [rbp-E8h] BYREF
  __int64 v53; // [rsp+88h] [rbp-D8h]
  char *v54; // [rsp+90h] [rbp-D0h] BYREF
  __int64 v55; // [rsp+98h] [rbp-C8h]
  char v56; // [rsp+A0h] [rbp-C0h] BYREF
  char *v57; // [rsp+A8h] [rbp-B8h] BYREF
  __int64 v58; // [rsp+B0h] [rbp-B0h]
  _BYTE v59[16]; // [rsp+B8h] [rbp-A8h] BYREF
  __int64 v60; // [rsp+C8h] [rbp-98h]
  __int64 v61; // [rsp+D0h] [rbp-90h]
  char *v62[2]; // [rsp+E0h] [rbp-80h] BYREF
  char v63; // [rsp+F0h] [rbp-70h] BYREF
  char *v64[2]; // [rsp+F8h] [rbp-68h] BYREF
  _BYTE v65[16]; // [rsp+108h] [rbp-58h] BYREF
  __int64 v66; // [rsp+118h] [rbp-48h]
  int v67; // [rsp+120h] [rbp-40h]

  v8 = *(_QWORD *)(a2 + 120);
  v47 = &v49;
  v50 = v52;
  v48 = 0x100000000LL;
  v51 = 0x400000000LL;
  v53 = 0;
  if ( v8 )
  {
    v9 = a1 + 432;
    sub_10774E0((__int64)&v47, v8, a3, a4, a5, a6);
    sub_10774E0((__int64)&v50, v8 + 24, v10, v11, v12, v13);
    v55 = 0x100000000LL;
    v14 = *(unsigned int *)(a1 + 472);
    v54 = &v56;
    if ( (_DWORD)v48 )
      sub_10774E0((__int64)&v54, (__int64)&v47, a3, a4, a5, a6);
  }
  else
  {
    v55 = 0x100000000LL;
    v14 = *(unsigned int *)(a1 + 472);
    v9 = a1 + 432;
    v54 = &v56;
  }
  v57 = v59;
  v58 = 0x400000000LL;
  if ( (_DWORD)v51 )
    sub_10774E0((__int64)&v57, (__int64)&v50, a3, a4, a5, a6);
  v15 = (unsigned int)v55;
  v61 = v14;
  v60 = v53;
  v62[0] = &v63;
  v62[1] = (char *)0x100000000LL;
  if ( (_DWORD)v55 )
    sub_1077380((__int64)v62, &v54, a3, a4, a5, (unsigned int)v55);
  v64[0] = v65;
  v64[1] = (char *)0x400000000LL;
  if ( (_DWORD)v58 )
    sub_1077380((__int64)v64, &v57, a3, a4, (unsigned int)v58, v15);
  v16 = v62;
  v66 = v60;
  v67 = v61;
  v17 = sub_107D670(v9, (__int64)v62, &v45);
  v20 = v45;
  if ( v17 )
  {
    v21 = 0;
    goto LABEL_12;
  }
  v31 = *(_DWORD *)(a1 + 456);
  v32 = *(_DWORD *)(a1 + 448);
  v46 = v45;
  ++*(_QWORD *)(a1 + 432);
  v33 = v32 + 1;
  v34 = 2 * v31;
  if ( 4 * v33 >= 3 * v31 )
  {
    v31 *= 2;
  }
  else
  {
    v35 = v31 - *(_DWORD *)(a1 + 452) - v33;
    v36 = v31 >> 3;
    if ( (unsigned int)v35 > (unsigned int)v36 )
      goto LABEL_37;
  }
  sub_107F9A0(v9, v31);
  sub_107D670(v9, (__int64)v62, &v46);
  v20 = v46;
  v33 = *(_DWORD *)(a1 + 448) + 1;
LABEL_37:
  *(_DWORD *)(a1 + 448) = v33;
  if ( *(_DWORD *)(v20 + 60) != 1 || *(_DWORD *)(v20 + 8) || *(_DWORD *)(v20 + 32) )
    --*(_DWORD *)(a1 + 452);
  v21 = 1;
  sub_1077380(v20, v62, v35, v36, v34, v19);
  v16 = v64;
  sub_1077380(v20 + 24, v64, v37, v38, v39, v40);
  *(_QWORD *)(v20 + 56) = v66;
  *(_DWORD *)(v20 + 64) = v67;
LABEL_12:
  if ( v64[0] != v65 )
    _libc_free(v64[0], v16);
  if ( v62[0] != &v63 )
    _libc_free(v62[0], v16);
  if ( v57 != v59 )
    _libc_free(v57, v16);
  if ( v54 != &v56 )
    _libc_free(v54, v16);
  if ( v21 )
  {
    v24 = *(unsigned int *)(a1 + 472);
    v25 = *(_QWORD *)(a1 + 464);
    v26 = (char *)&v47;
    v27 = v24 + 1;
    v28 = v24;
    if ( v24 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 476) )
    {
      v41 = a1 + 464;
      if ( v25 > (unsigned __int64)&v47 || (unsigned __int64)&v47 >= v25 + (v24 << 6) )
      {
        sub_107D930(v41, v27, (__int64)&v47, v25, v18, v19);
        v24 = *(unsigned int *)(a1 + 472);
        v25 = *(_QWORD *)(a1 + 464);
        v26 = (char *)&v47;
        v28 = *(_DWORD *)(a1 + 472);
      }
      else
      {
        v42 = (char *)&v47 - v25;
        sub_107D930(v41, v27, (__int64)&v47 - v25, v25, v18, v19);
        v25 = *(_QWORD *)(a1 + 464);
        v26 = &v42[v25];
        v24 = *(unsigned int *)(a1 + 472);
        v28 = *(_DWORD *)(a1 + 472);
      }
    }
    v29 = (_QWORD *)(v25 + (v24 << 6));
    if ( v29 )
    {
      *v29 = v29 + 2;
      v29[1] = 0x100000000LL;
      v30 = *((unsigned int *)v26 + 2);
      if ( (_DWORD)v30 )
      {
        v44 = v26;
        sub_10774E0((__int64)v29, (__int64)v26, (__int64)v26, v30, v18, v19);
        v26 = v44;
      }
      v29[3] = v29 + 5;
      v29[4] = 0x400000000LL;
      if ( *((_DWORD *)v26 + 8) )
      {
        v43 = v26;
        sub_10774E0((__int64)(v29 + 3), (__int64)(v26 + 24), (__int64)v26, v30, v18, v19);
        v26 = v43;
      }
      v29[7] = *((_QWORD *)v26 + 7);
      v28 = *(_DWORD *)(a1 + 472);
    }
    *(_DWORD *)(a1 + 472) = v28 + 1;
  }
  v22 = *(_DWORD *)(v20 + 64);
  v62[0] = (char *)a2;
  result = sub_107DC60(a1 + 168, (__int64 *)v62);
  *result = v22;
  if ( v50 != v52 )
    result = (_DWORD *)_libc_free(v50, v62);
  if ( v47 != &v49 )
    return (_DWORD *)_libc_free(v47, v62);
  return result;
}
