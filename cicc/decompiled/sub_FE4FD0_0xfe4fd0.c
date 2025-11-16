// Function: sub_FE4FD0
// Address: 0xfe4fd0
//
__int64 __fastcall sub_FE4FD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __int64 v15; // rdx
  __int64 v16; // rcx
  __int64 v17; // r8
  __int64 v18; // r9
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  unsigned int v27; // edx
  __int64 v28; // rcx
  char *v29; // rsi
  char *v30; // rax
  _BYTE *v31; // rsi
  __int64 v32; // rax
  _BYTE v34[8]; // [rsp+0h] [rbp-DB0h] BYREF
  __int64 v35; // [rsp+8h] [rbp-DA8h]
  char v36; // [rsp+1Ch] [rbp-D94h]
  char *v37; // [rsp+60h] [rbp-D50h]
  char v38; // [rsp+70h] [rbp-D40h] BYREF
  _BYTE v39[8]; // [rsp+1B0h] [rbp-C00h] BYREF
  __int64 v40; // [rsp+1B8h] [rbp-BF8h]
  char v41; // [rsp+1CCh] [rbp-BE4h]
  char *v42; // [rsp+210h] [rbp-BA0h]
  char v43; // [rsp+220h] [rbp-B90h] BYREF
  _BYTE v44[8]; // [rsp+360h] [rbp-A50h] BYREF
  __int64 v45; // [rsp+368h] [rbp-A48h]
  char v46; // [rsp+37Ch] [rbp-A34h]
  char *v47; // [rsp+3C0h] [rbp-9F0h]
  char v48; // [rsp+3D0h] [rbp-9E0h] BYREF
  _BYTE v49[8]; // [rsp+510h] [rbp-8A0h] BYREF
  __int64 v50; // [rsp+518h] [rbp-898h]
  char v51; // [rsp+52Ch] [rbp-884h]
  char *v52; // [rsp+570h] [rbp-840h]
  char v53; // [rsp+580h] [rbp-830h] BYREF
  _BYTE v54[8]; // [rsp+6C0h] [rbp-6F0h] BYREF
  __int64 v55; // [rsp+6C8h] [rbp-6E8h]
  char v56; // [rsp+6DCh] [rbp-6D4h]
  char *v57; // [rsp+720h] [rbp-690h]
  char v58; // [rsp+730h] [rbp-680h] BYREF
  _BYTE v59[8]; // [rsp+870h] [rbp-540h] BYREF
  __int64 v60; // [rsp+878h] [rbp-538h]
  char v61; // [rsp+88Ch] [rbp-524h]
  char *v62; // [rsp+8D0h] [rbp-4E0h]
  char v63; // [rsp+8E0h] [rbp-4D0h] BYREF
  _BYTE v64[8]; // [rsp+A20h] [rbp-390h] BYREF
  __int64 v65; // [rsp+A28h] [rbp-388h]
  char v66; // [rsp+A3Ch] [rbp-374h]
  char *v67; // [rsp+A80h] [rbp-330h]
  int v68; // [rsp+A88h] [rbp-328h]
  char v69; // [rsp+A90h] [rbp-320h] BYREF
  _BYTE v70[8]; // [rsp+BD0h] [rbp-1E0h] BYREF
  __int64 v71; // [rsp+BD8h] [rbp-1D8h]
  char v72; // [rsp+BECh] [rbp-1C4h]
  char *v73; // [rsp+C30h] [rbp-180h]
  unsigned int v74; // [rsp+C38h] [rbp-178h]
  char v75; // [rsp+C40h] [rbp-170h] BYREF

  sub_FDEF40((__int64)v44, a2, a3, a4, a5, a6);
  sub_FDEE20((__int64)v49, (__int64)v44);
  sub_FDEF40((__int64)v34, a1, v7, v8, v9, v10);
  sub_FDEE20((__int64)v39, (__int64)v34);
  sub_FDEF40((__int64)v59, (__int64)v49, v11, v12, v13, v14);
  sub_FDEF40((__int64)v54, (__int64)v39, v15, v16, v17, v18);
  sub_FDEF40((__int64)v70, (__int64)v59, v19, v20, v21, v22);
  sub_FDEF40((__int64)v64, (__int64)v54, v23, v24, v25, v26);
LABEL_2:
  v27 = v68;
  while ( 1 )
  {
    v28 = 40LL * v27;
    if ( v27 == (unsigned __int64)v74 )
      break;
LABEL_7:
    v31 = *(_BYTE **)(a3 + 8);
    v32 = (__int64)&v67[v28 - 40];
    if ( v31 == *(_BYTE **)(a3 + 16) )
    {
      sub_F46430(a3, v31, (_QWORD *)(v32 + 32));
      v27 = v68;
    }
    else
    {
      if ( v31 )
      {
        *(_QWORD *)v31 = *(_QWORD *)(v32 + 32);
        v31 = *(_BYTE **)(a3 + 8);
        v27 = v68;
      }
      *(_QWORD *)(a3 + 8) = v31 + 8;
    }
    v68 = --v27;
    if ( v27 )
    {
      sub_FDEBC0((__int64)v64);
      goto LABEL_2;
    }
  }
  v29 = v73;
  if ( v67 != &v67[v28] )
  {
    v30 = v67;
    while ( *((_QWORD *)v30 + 4) == *((_QWORD *)v29 + 4)
         && *((_DWORD *)v30 + 6) == *((_DWORD *)v29 + 6)
         && *((_DWORD *)v30 + 2) == *((_DWORD *)v29 + 2) )
    {
      v30 += 40;
      v29 += 40;
      if ( &v67[v28] == v30 )
        goto LABEL_16;
    }
    goto LABEL_7;
  }
LABEL_16:
  if ( v67 != &v69 )
    _libc_free(v67, v29);
  if ( !v66 )
    _libc_free(v65, v29);
  if ( v73 != &v75 )
    _libc_free(v73, v29);
  if ( !v72 )
    _libc_free(v71, v29);
  if ( v57 != &v58 )
    _libc_free(v57, v29);
  if ( !v56 )
    _libc_free(v55, v29);
  if ( v62 != &v63 )
    _libc_free(v62, v29);
  if ( !v61 )
    _libc_free(v60, v29);
  if ( v42 != &v43 )
    _libc_free(v42, v29);
  if ( !v41 )
    _libc_free(v40, v29);
  if ( v37 != &v38 )
    _libc_free(v37, v29);
  if ( !v36 )
    _libc_free(v35, v29);
  if ( v52 != &v53 )
    _libc_free(v52, v29);
  if ( !v51 )
    _libc_free(v50, v29);
  if ( v47 != &v48 )
    _libc_free(v47, v29);
  if ( !v46 )
    _libc_free(v45, v29);
  return a3;
}
