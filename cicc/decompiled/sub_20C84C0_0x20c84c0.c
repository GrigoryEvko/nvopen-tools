// Function: sub_20C84C0
// Address: 0x20c84c0
//
__int64 __fastcall sub_20C84C0(__int64 a1, __int64 a2, __int64 a3, const char **a4)
{
  __int64 v6; // rcx
  int v7; // r8d
  int v8; // r9d
  __int64 v9; // rax
  int v10; // eax
  const char *v11; // rbx
  char v12; // r12
  __int64 v13; // rcx
  int v14; // r8d
  int v15; // r9d
  char v16; // al
  __int64 v17; // r9
  char *v18; // rax
  int v19; // edx
  unsigned __int64 v20; // r8
  __int64 v21; // r10
  __int64 v22; // rbx
  int v23; // r9d
  char *v24; // rax
  int v25; // edx
  __int64 v26; // r9
  unsigned __int64 v27; // r8
  __int64 v28; // r10
  __int64 v29; // rbx
  __int64 v30; // rax
  __int64 *v31; // rbx
  _BYTE *v32; // rdi
  size_t v34; // rdx
  char v35; // bl
  __int64 **v36; // rax
  const char *v37; // rbx
  const char *v38; // rbx
  unsigned __int8 v39; // [rsp+2Fh] [rbp-191h]
  __int64 v40; // [rsp+30h] [rbp-190h]
  __int64 v41; // [rsp+30h] [rbp-190h]
  char v42; // [rsp+70h] [rbp-150h]
  unsigned __int64 v43; // [rsp+70h] [rbp-150h]
  unsigned __int64 v44; // [rsp+70h] [rbp-150h]
  __int64 *v46; // [rsp+80h] [rbp-140h]
  __int64 v47; // [rsp+88h] [rbp-138h]
  __int64 v48; // [rsp+98h] [rbp-128h]
  __int64 v49; // [rsp+98h] [rbp-128h]
  __int64 v50; // [rsp+98h] [rbp-128h]
  char v51; // [rsp+A7h] [rbp-119h] BYREF
  unsigned int v52; // [rsp+A8h] [rbp-118h] BYREF
  unsigned int v53; // [rsp+ACh] [rbp-114h] BYREF
  _BYTE *v54; // [rsp+B0h] [rbp-110h] BYREF
  __int64 v55; // [rsp+B8h] [rbp-108h]
  _BYTE v56[16]; // [rsp+C0h] [rbp-100h] BYREF
  _BYTE *v57; // [rsp+D0h] [rbp-F0h] BYREF
  __int64 v58; // [rsp+D8h] [rbp-E8h]
  _BYTE v59[16]; // [rsp+E0h] [rbp-E0h] BYREF
  void *s2; // [rsp+F0h] [rbp-D0h] BYREF
  __int64 v61; // [rsp+F8h] [rbp-C8h]
  _BYTE v62[16]; // [rsp+100h] [rbp-C0h] BYREF
  void *s1; // [rsp+110h] [rbp-B0h] BYREF
  __int64 v64; // [rsp+118h] [rbp-A8h]
  _BYTE v65[16]; // [rsp+120h] [rbp-A0h] BYREF
  _BYTE *v66; // [rsp+130h] [rbp-90h] BYREF
  __int64 v67; // [rsp+138h] [rbp-88h]
  _BYTE v68[32]; // [rsp+140h] [rbp-80h] BYREF
  unsigned __int64 v69[2]; // [rsp+160h] [rbp-60h] BYREF
  _BYTE v70[80]; // [rsp+170h] [rbp-50h] BYREF

  v47 = a2;
  v39 = sub_20C83A0(a1, a2, a3, (__int64)a4, &v51);
  if ( !v39 )
    return v39;
  v46 = *(__int64 **)(a3 - 24LL * (*(_DWORD *)(a3 + 20) & 0xFFFFFFF));
  v9 = *(_QWORD *)(a2 - 24);
  if ( *(_BYTE *)(v9 + 16) )
    goto LABEL_9;
  v10 = *(_DWORD *)(v9 + 36);
  switch ( v10 )
  {
    case 133:
      v38 = a4[9588];
      if ( !v38 || strlen(a4[9588]) != 6 || *(_DWORD *)v38 != 1668113773 || *((_WORD *)v38 + 2) != 31088 )
        break;
      goto LABEL_61;
    case 135:
      v37 = a4[9589];
      if ( !v37
        || strlen(a4[9589]) != 7
        || *(_DWORD *)v37 != 1835885933
        || *((_WORD *)v37 + 2) != 30319
        || v37[6] != 101 )
      {
        break;
      }
LABEL_61:
      if ( v46 == *(__int64 **)(a2 - 24LL * (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)) )
        return v39;
      break;
    case 137:
      v11 = a4[9590];
      if ( v11 )
      {
        if ( strlen(a4[9590]) == 6 && *(_DWORD *)v11 == 1936549229 && *((_WORD *)v11 + 2) == 29797 )
          goto LABEL_61;
      }
      break;
  }
LABEL_9:
  v54 = v56;
  v66 = v68;
  v57 = v59;
  v55 = 0x400000000LL;
  v58 = 0x400000000LL;
  v67 = 0x400000000LL;
  v69[0] = (unsigned __int64)v70;
  v69[1] = 0x400000000LL;
  v12 = sub_20C7AC0(*v46, (__int64)&v66, (__int64)&v54, v6, v7, v8);
  v16 = sub_20C7AC0(*(_QWORD *)a2, (__int64)v69, (__int64)&v57, v13, v14, v15);
  if ( !v12 )
    goto LABEL_29;
  if ( v16 != 1 )
    goto LABEL_50;
  while ( 1 )
  {
    v17 = (unsigned int)v55;
    s2 = v62;
    v18 = v62;
    v19 = 0;
    v61 = 0x400000000LL;
    v20 = (unsigned __int64)v54;
    v21 = 4LL * (unsigned int)v55;
    v22 = (unsigned int)v55;
    if ( (unsigned int)v55 > 4uLL )
    {
      v41 = 4LL * (unsigned int)v55;
      v44 = (unsigned __int64)v54;
      v50 = (unsigned int)v55;
      sub_16CD150((__int64)&s2, v62, (unsigned int)v55, 4, (int)v54, v55);
      v19 = v61;
      v21 = v41;
      v20 = v44;
      v17 = v50;
      v18 = (char *)s2 + 4 * (unsigned int)v61;
    }
    if ( v21 )
    {
      do
      {
        v18 += 4;
        *((_DWORD *)v18 - 1) = *(_DWORD *)(v20 + v21 - 4 * v17 + 4 * v22-- - 4);
      }
      while ( v22 );
      v19 = v61;
    }
    v23 = v19 + v17;
    s1 = v65;
    v24 = v65;
    v25 = 0;
    LODWORD(v61) = v23;
    v26 = (unsigned int)v58;
    v64 = 0x400000000LL;
    v27 = (unsigned __int64)v57;
    v28 = 4LL * (unsigned int)v58;
    v29 = (unsigned int)v58;
    if ( (unsigned int)v58 > 4uLL )
    {
      v40 = 4LL * (unsigned int)v58;
      v43 = (unsigned __int64)v57;
      v49 = (unsigned int)v58;
      sub_16CD150((__int64)&s1, v65, (unsigned int)v58, 4, (int)v57, v58);
      v25 = v64;
      v28 = v40;
      v27 = v43;
      v26 = v49;
      v24 = (char *)s1 + 4 * (unsigned int)v64;
    }
    if ( v28 )
    {
      do
      {
        v24 += 4;
        *((_DWORD *)v24 - 1) = *(_DWORD *)(v27 + v28 - 4 * v26 + 4 * v29-- - 4);
      }
      while ( v29 );
      v25 = v64;
    }
    LODWORD(v64) = v25 + v26;
    v30 = sub_1632FA0(*(_QWORD *)(a1 + 40));
    v52 = -1;
    v48 = v30;
    v42 = v51;
    v31 = sub_20C7600((__int64)v46, (__int64)&s2, &v52, (__int64)a4, v30);
    if ( *((_BYTE *)v31 + 16) == 9 )
      goto LABEL_44;
    v53 = -1;
    if ( v31 != sub_20C7600(v47, (__int64)&s1, &v53, (__int64)a4, v48) )
      break;
    v32 = s1;
    if ( (unsigned int)v64 != (unsigned __int64)(unsigned int)v61 )
      goto LABEL_24;
    v34 = 4LL * (unsigned int)v64;
    if ( v34 )
    {
      v32 = s1;
      if ( memcmp(s1, s2, v34) )
        goto LABEL_24;
    }
    if ( v53 < v52 || !v42 && v53 != v52 )
      goto LABEL_24;
LABEL_44:
    v35 = sub_20C7510((__int64)v69, (__int64)&v57) ^ 1;
    if ( s1 != v65 )
      _libc_free((unsigned __int64)s1);
    if ( s2 != v62 )
      _libc_free((unsigned __int64)s2);
    if ( !(unsigned __int8)sub_20C7510((__int64)&v66, (__int64)&v54) )
      goto LABEL_29;
    if ( v35 )
    {
LABEL_50:
      v36 = (__int64 **)sub_1643D80(
                          *(_QWORD *)&v66[8 * (unsigned int)v67 - 8],
                          *(_DWORD *)&v54[4 * (unsigned int)v55 - 4]);
      v47 = sub_1599EF0(v36);
    }
  }
  v32 = s1;
LABEL_24:
  if ( v32 != v65 )
    _libc_free((unsigned __int64)v32);
  if ( s2 != v62 )
    _libc_free((unsigned __int64)s2);
  v39 = 0;
LABEL_29:
  if ( (_BYTE *)v69[0] != v70 )
    _libc_free(v69[0]);
  if ( v66 != v68 )
    _libc_free((unsigned __int64)v66);
  if ( v57 != v59 )
    _libc_free((unsigned __int64)v57);
  if ( v54 != v56 )
    _libc_free((unsigned __int64)v54);
  return v39;
}
