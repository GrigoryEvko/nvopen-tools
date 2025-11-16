// Function: sub_34B9520
// Address: 0x34b9520
//
__int64 __fastcall sub_34B9520(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5)
{
  int v8; // eax
  __int64 v9; // rax
  __int64 v10; // rdi
  char v11; // r12
  char v12; // al
  __int64 v13; // r9
  char *v14; // rax
  int v15; // edx
  __int64 v16; // r8
  __int64 v17; // r10
  __int64 v18; // rbx
  int v19; // r9d
  char *v20; // rax
  int v21; // edx
  __int64 v22; // r9
  __int64 v23; // r8
  __int64 v24; // r10
  __int64 v25; // rbx
  __int64 v26; // rbx
  _BYTE *v27; // rdi
  size_t v29; // rdx
  char v30; // bl
  __int64 **v31; // rax
  __int64 v32; // [rsp+30h] [rbp-190h]
  __int64 v33; // [rsp+30h] [rbp-190h]
  unsigned __int8 v35; // [rsp+47h] [rbp-179h]
  char v36; // [rsp+48h] [rbp-178h]
  __int64 v37; // [rsp+48h] [rbp-178h]
  __int64 v38; // [rsp+48h] [rbp-178h]
  unsigned __int8 *v39; // [rsp+70h] [rbp-150h]
  unsigned __int8 *v40; // [rsp+78h] [rbp-148h]
  unsigned __int8 *v42; // [rsp+90h] [rbp-130h]
  __int64 v43; // [rsp+90h] [rbp-130h]
  __int64 v44; // [rsp+90h] [rbp-130h]
  char v45; // [rsp+A7h] [rbp-119h] BYREF
  unsigned int v46; // [rsp+A8h] [rbp-118h] BYREF
  unsigned int v47; // [rsp+ACh] [rbp-114h] BYREF
  _BYTE *v48; // [rsp+B0h] [rbp-110h] BYREF
  __int64 v49; // [rsp+B8h] [rbp-108h]
  _BYTE v50[16]; // [rsp+C0h] [rbp-100h] BYREF
  _BYTE *v51; // [rsp+D0h] [rbp-F0h] BYREF
  __int64 v52; // [rsp+D8h] [rbp-E8h]
  _BYTE v53[16]; // [rsp+E0h] [rbp-E0h] BYREF
  void *s2; // [rsp+F0h] [rbp-D0h] BYREF
  __int64 v55; // [rsp+F8h] [rbp-C8h]
  _BYTE v56[16]; // [rsp+100h] [rbp-C0h] BYREF
  void *s1; // [rsp+110h] [rbp-B0h] BYREF
  __int64 v58; // [rsp+118h] [rbp-A8h]
  _BYTE v59[16]; // [rsp+120h] [rbp-A0h] BYREF
  _BYTE *v60; // [rsp+130h] [rbp-90h] BYREF
  __int64 v61; // [rsp+138h] [rbp-88h]
  _BYTE v62[32]; // [rsp+140h] [rbp-80h] BYREF
  unsigned __int64 v63[2]; // [rsp+160h] [rbp-60h] BYREF
  _BYTE v64[80]; // [rsp+170h] [rbp-50h] BYREF

  v39 = (unsigned __int8 *)a2;
  v35 = sub_34B9300(a1, a2, a3, a4, &v45);
  if ( !v35 )
    return v35;
  if ( a5 )
    return a5;
  v8 = *(_DWORD *)(a3 + 4);
  v60 = v62;
  v48 = v50;
  v9 = *(_QWORD *)(a3 - 32LL * (v8 & 0x7FFFFFF));
  v49 = 0x400000000LL;
  v52 = 0x400000000LL;
  v61 = 0x400000000LL;
  v63[1] = 0x400000000LL;
  v10 = *(_QWORD *)(v9 + 8);
  v51 = v53;
  v63[0] = (unsigned __int64)v64;
  v40 = (unsigned __int8 *)v9;
  v11 = sub_34B8A40(v10, (__int64)&v60, (__int64)&v48);
  v12 = sub_34B8A40(*(_QWORD *)(a2 + 8), (__int64)v63, (__int64)&v51);
  if ( !v11 )
    goto LABEL_23;
  if ( v12 != 1 )
    goto LABEL_43;
  while ( 1 )
  {
    v13 = (unsigned int)v49;
    s2 = v56;
    v14 = v56;
    v15 = 0;
    v55 = 0x400000000LL;
    v16 = (__int64)v48;
    v17 = 4LL * (unsigned int)v49;
    v18 = (unsigned int)v49;
    if ( (unsigned int)v49 > 4uLL )
    {
      v33 = 4LL * (unsigned int)v49;
      v38 = (__int64)v48;
      v44 = (unsigned int)v49;
      sub_C8D5F0((__int64)&s2, v56, (unsigned int)v49, 4u, (__int64)v48, (unsigned int)v49);
      v15 = v55;
      v17 = v33;
      v16 = v38;
      v13 = v44;
      v14 = (char *)s2 + 4 * (unsigned int)v55;
    }
    if ( v17 )
    {
      do
      {
        v14 += 4;
        *((_DWORD *)v14 - 1) = *(_DWORD *)(v16 + v17 - 4 * v13 + 4 * v18-- - 4);
      }
      while ( v18 );
      v15 = v55;
    }
    v19 = v15 + v13;
    s1 = v59;
    v20 = v59;
    v21 = 0;
    LODWORD(v55) = v19;
    v22 = (unsigned int)v52;
    v58 = 0x400000000LL;
    v23 = (__int64)v51;
    v24 = 4LL * (unsigned int)v52;
    v25 = (unsigned int)v52;
    if ( (unsigned int)v52 > 4uLL )
    {
      v32 = 4LL * (unsigned int)v52;
      v37 = (__int64)v51;
      v43 = (unsigned int)v52;
      sub_C8D5F0((__int64)&s1, v59, (unsigned int)v52, 4u, (__int64)v51, (unsigned int)v52);
      v21 = v58;
      v24 = v32;
      v23 = v37;
      v22 = v43;
      v20 = (char *)s1 + 4 * (unsigned int)v58;
    }
    if ( v24 )
    {
      do
      {
        v20 += 4;
        *((_DWORD *)v20 - 1) = *(_DWORD *)(v23 + v24 - 4 * v22 + 4 * v25-- - 4);
      }
      while ( v25 );
      v21 = v58;
    }
    LODWORD(v58) = v21 + v22;
    v26 = sub_B2BEC0(a1);
    v46 = -1;
    v36 = v45;
    v42 = sub_34B8690(v40, (__int64)&s2, &v46, a4, v26);
    if ( (unsigned __int8)(*v42 - 12) <= 1u )
      goto LABEL_37;
    v47 = -1;
    if ( v42 != sub_34B8690(v39, (__int64)&s1, &v47, a4, v26) )
      break;
    v27 = s1;
    if ( (unsigned int)v58 != (unsigned __int64)(unsigned int)v55 )
      goto LABEL_18;
    v29 = 4LL * (unsigned int)v58;
    if ( v29 )
    {
      v27 = s1;
      if ( memcmp(s1, s2, v29) )
        goto LABEL_18;
    }
    if ( v47 < v46 || !v36 && v47 != v46 )
      goto LABEL_18;
LABEL_37:
    v30 = sub_34B8590((__int64)v63, (__int64)&v51) ^ 1;
    if ( s1 != v59 )
      _libc_free((unsigned __int64)s1);
    if ( s2 != v56 )
      _libc_free((unsigned __int64)s2);
    if ( !(unsigned __int8)sub_34B8590((__int64)&v60, (__int64)&v48) )
      goto LABEL_23;
    if ( v30 )
    {
LABEL_43:
      v31 = (__int64 **)sub_B501B0(
                          *(_QWORD *)&v60[8 * (unsigned int)v61 - 8],
                          (unsigned int *)&v48[4 * (unsigned int)v49 - 4],
                          1);
      v39 = (unsigned __int8 *)sub_ACA8A0(v31);
    }
  }
  v27 = s1;
LABEL_18:
  if ( v27 != v59 )
    _libc_free((unsigned __int64)v27);
  if ( s2 != v56 )
    _libc_free((unsigned __int64)s2);
  v35 = a5;
LABEL_23:
  if ( (_BYTE *)v63[0] != v64 )
    _libc_free(v63[0]);
  if ( v60 != v62 )
    _libc_free((unsigned __int64)v60);
  if ( v51 != v53 )
    _libc_free((unsigned __int64)v51);
  if ( v48 != v50 )
    _libc_free((unsigned __int64)v48);
  return v35;
}
