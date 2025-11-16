// Function: sub_B05AE0
// Address: 0xb05ae0
//
__int64 __fastcall sub_B05AE0(
        _QWORD *a1,
        int a2,
        __int64 a3,
        __int64 a4,
        int a5,
        _BYTE *a6,
        __int64 a7,
        __int64 a8,
        int a9,
        __int64 a10,
        __int64 a11,
        __int64 a12,
        int a13,
        __int128 a14,
        unsigned int a15,
        char a16)
{
  _QWORD *v17; // r12
  _DWORD *v19; // r10
  int v20; // r11d
  int v21; // eax
  _QWORD *v22; // r10
  int v23; // r11d
  __int64 *v24; // r12
  __int64 v25; // r8
  __int64 v26; // rax
  __int16 v27; // ax
  __int64 v28; // rdi
  __int64 v29; // rax
  _BYTE *v30; // rax
  __int64 *v31; // rcx
  __int64 result; // rax
  __int64 v33; // r13
  __int64 v34; // rbx
  __int64 v35; // rax
  __int8 *v36; // rax
  __int8 *v37; // rax
  _BYTE *v38; // [rsp+8h] [rbp-198h]
  __int64 v39; // [rsp+10h] [rbp-190h]
  __int64 v40; // [rsp+10h] [rbp-190h]
  __int64 v41; // [rsp+18h] [rbp-188h]
  __int64 v42; // [rsp+18h] [rbp-188h]
  int v43; // [rsp+20h] [rbp-180h]
  int i; // [rsp+28h] [rbp-178h]
  __int64 v45; // [rsp+28h] [rbp-178h]
  _QWORD *v46; // [rsp+30h] [rbp-170h]
  __int8 *v47; // [rsp+38h] [rbp-168h]
  _QWORD *srcb; // [rsp+40h] [rbp-160h]
  _QWORD *srca; // [rsp+40h] [rbp-160h]
  int v51; // [rsp+48h] [rbp-158h]
  unsigned int v52; // [rsp+48h] [rbp-158h]
  int v53; // [rsp+48h] [rbp-158h]
  __int64 v54; // [rsp+50h] [rbp-150h]
  __int64 v57; // [rsp+70h] [rbp-130h] BYREF
  __int64 v58; // [rsp+78h] [rbp-128h] BYREF
  __int64 v59; // [rsp+80h] [rbp-120h] BYREF
  __int64 v60; // [rsp+88h] [rbp-118h] BYREF
  __int64 v61; // [rsp+90h] [rbp-110h] BYREF
  int v62; // [rsp+98h] [rbp-108h] BYREF
  _BYTE *v63; // [rsp+A0h] [rbp-100h] BYREF
  __int64 v64[3]; // [rsp+A8h] [rbp-F8h] BYREF
  int v65; // [rsp+C0h] [rbp-E0h]
  __int64 v66; // [rsp+C4h] [rbp-DCh]
  __int64 v67; // [rsp+CCh] [rbp-D4h]
  int v68; // [rsp+D4h] [rbp-CCh] BYREF
  __int128 v69; // [rsp+D8h] [rbp-C8h]
  __m128i dest; // [rsp+F0h] [rbp-B0h] BYREF
  __int128 v71; // [rsp+100h] [rbp-A0h]
  __int128 v72; // [rsp+110h] [rbp-90h]
  __int128 v73; // [rsp+120h] [rbp-80h]
  unsigned __int64 v74[7]; // [rsp+130h] [rbp-70h] BYREF
  __int64 (__fastcall *v75)(); // [rsp+168h] [rbp-38h]

  v17 = a1;
  if ( a15 )
  {
LABEL_23:
    *(_QWORD *)&v71 = a3;
    dest.m128i_i64[1] = (__int64)a6;
    dest.m128i_i64[0] = a4;
    *((_QWORD *)&v71 + 1) = a7;
    v72 = a14;
    v33 = *v17 + 920LL;
    v58 = a11;
    v34 = sub_B97910(56, 6, a15);
    if ( v34 )
    {
      v59 = v58;
      sub_B971C0(v34, (_DWORD)v17, 13, a15, (unsigned int)&dest, 6, 0, 0);
      *(_WORD *)(v34 + 2) = a2;
      *(_DWORD *)(v34 + 40) = 0;
      *(_DWORD *)(v34 + 16) = a5;
      *(_DWORD *)(v34 + 20) = a13;
      *(_QWORD *)(v34 + 24) = a8;
      *(_DWORD *)(v34 + 4) = a9;
      *(_QWORD *)(v34 + 32) = a10;
      *(_QWORD *)(v34 + 44) = v59;
      if ( BYTE4(a12) )
        *(_DWORD *)(v34 + 4) = a12;
    }
    return sub_B05A00(v34, a15, v33);
  }
  v19 = (_DWORD *)*a1;
  LODWORD(v59) = a2;
  v60 = a3;
  dest.m128i_i64[0] = a12;
  v67 = a12;
  v61 = a4;
  v68 = a13;
  v62 = a5;
  v64[0] = a7;
  v69 = a14;
  v58 = a11;
  v63 = a6;
  v64[1] = a8;
  v64[2] = a10;
  v65 = a9;
  v66 = a11;
  v20 = v19[236];
  v54 = *((_QWORD *)v19 + 116);
  if ( !v20 )
    goto LABEL_22;
  if ( a3 != 0
    && a2 == 13
    && a6
    && *a6 == 14
    && (srca = v19, v53 = v19[236], v35 = sub_AF5140((__int64)a6, 7u), v20 = v53, v19 = srca, v35) )
  {
    v75 = sub_C64CA0;
    dest = 0;
    v71 = 0;
    v72 = 0;
    v73 = 0;
    memset(v74, 0, sizeof(v74));
    v57 = 0;
    v36 = sub_AF8740(&dest, &v57, dest.m128i_i8, (unsigned __int64)v74, a3);
    v58 = v57;
    v37 = sub_AF70F0(&dest, &v58, v36, (unsigned __int64)v74, (__int64)v63);
    if ( v58 )
    {
      v45 = v58;
      v47 = v37;
      sub_AF1140(dest.m128i_i8, v37, (char *)v74);
      sub_AC2A10(v74, &dest);
      v21 = sub_AF1490(v74, v47 - (__int8 *)&dest + v45);
      v22 = srca;
      v23 = v53;
    }
    else
    {
      v21 = sub_AC25F0(&dest, v37 - (__int8 *)&dest, (__int64)v75);
      v23 = v53;
      v22 = srca;
    }
  }
  else
  {
    srcb = v19;
    v51 = v20;
    v21 = sub_AF95C0((int *)&v59, &v60, &v61, &v62, (__int64 *)&v63, v64, &v68);
    v22 = srcb;
    v23 = v51;
  }
  v46 = v22;
  v43 = v23 - 1;
  v52 = (v23 - 1) & v21;
  for ( i = 1; ; ++i )
  {
    v24 = (__int64 *)(v54 + 8LL * v52);
    v25 = *v24;
    if ( *v24 == -4096 )
    {
LABEL_32:
      v17 = a1;
      goto LABEL_22;
    }
    if ( v25 != -8192 )
      break;
LABEL_18:
    if ( v25 == -4096 )
      goto LABEL_32;
    v52 = v43 & (i + v52);
  }
  if ( v60 == 0 || (_DWORD)v59 != 13 )
    goto LABEL_36;
  if ( !v63 )
    goto LABEL_36;
  v41 = v60;
  if ( *v63 != 14 )
    goto LABEL_36;
  v39 = *v24;
  v38 = v63;
  v26 = sub_AF5140((__int64)v63, 7u);
  v25 = v39;
  if ( !v26 )
    goto LABEL_36;
  v27 = sub_AF18C0(v39);
  v25 = v39;
  if ( v27 != 13
    || (v28 = v39, v40 = v41, v42 = v25, v29 = sub_AF5140(v28, 2u), v25 = v42, v40 != v29)
    || (v30 = sub_A17150((_BYTE *)(v42 - 16)), v25 = v42, v38 != *((_BYTE **)v30 + 1)) )
  {
LABEL_36:
    if ( !sub_AF5170((int *)&v59, v25) )
    {
      v25 = *v24;
      goto LABEL_18;
    }
  }
  v31 = (__int64 *)(v54 + 8LL * v52);
  v17 = a1;
  if ( v31 == (__int64 *)(v46[116] + 8LL * *((unsigned int *)v46 + 236)) || (result = *v31) == 0 )
  {
LABEL_22:
    result = 0;
    if ( a16 )
      goto LABEL_23;
  }
  return result;
}
