// Function: sub_F1FD70
// Address: 0xf1fd70
//
__int64 __fastcall sub_F1FD70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
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
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // rax
  __int64 v30; // rdx
  char *v31; // rsi
  char *v32; // rcx
  char *v33; // rax
  __int64 v34; // rax
  __int64 v35; // r14
  unsigned int v36; // eax
  _BYTE v38[8]; // [rsp+0h] [rbp-DB0h] BYREF
  __int64 v39; // [rsp+8h] [rbp-DA8h]
  char v40; // [rsp+1Ch] [rbp-D94h]
  char *v41; // [rsp+60h] [rbp-D50h]
  char v42; // [rsp+70h] [rbp-D40h] BYREF
  _BYTE v43[8]; // [rsp+1B0h] [rbp-C00h] BYREF
  __int64 v44; // [rsp+1B8h] [rbp-BF8h]
  char v45; // [rsp+1CCh] [rbp-BE4h]
  char *v46; // [rsp+210h] [rbp-BA0h]
  char v47; // [rsp+220h] [rbp-B90h] BYREF
  _BYTE v48[8]; // [rsp+360h] [rbp-A50h] BYREF
  __int64 v49; // [rsp+368h] [rbp-A48h]
  char v50; // [rsp+37Ch] [rbp-A34h]
  char *v51; // [rsp+3C0h] [rbp-9F0h]
  char v52; // [rsp+3D0h] [rbp-9E0h] BYREF
  _BYTE v53[8]; // [rsp+510h] [rbp-8A0h] BYREF
  __int64 v54; // [rsp+518h] [rbp-898h]
  char v55; // [rsp+52Ch] [rbp-884h]
  char *v56; // [rsp+570h] [rbp-840h]
  char v57; // [rsp+580h] [rbp-830h] BYREF
  _BYTE v58[8]; // [rsp+6C0h] [rbp-6F0h] BYREF
  __int64 v59; // [rsp+6C8h] [rbp-6E8h]
  char v60; // [rsp+6DCh] [rbp-6D4h]
  char *v61; // [rsp+720h] [rbp-690h]
  char v62; // [rsp+730h] [rbp-680h] BYREF
  _BYTE v63[8]; // [rsp+870h] [rbp-540h] BYREF
  __int64 v64; // [rsp+878h] [rbp-538h]
  char v65; // [rsp+88Ch] [rbp-524h]
  char *v66; // [rsp+8D0h] [rbp-4E0h]
  char v67; // [rsp+8E0h] [rbp-4D0h] BYREF
  _BYTE v68[8]; // [rsp+A20h] [rbp-390h] BYREF
  __int64 v69; // [rsp+A28h] [rbp-388h]
  char v70; // [rsp+A3Ch] [rbp-374h]
  char *v71; // [rsp+A80h] [rbp-330h]
  unsigned int v72; // [rsp+A88h] [rbp-328h]
  char v73; // [rsp+A90h] [rbp-320h] BYREF
  _BYTE v74[8]; // [rsp+BD0h] [rbp-1E0h] BYREF
  __int64 v75; // [rsp+BD8h] [rbp-1D8h]
  char v76; // [rsp+BECh] [rbp-1C4h]
  char *v77; // [rsp+C30h] [rbp-180h]
  int v78; // [rsp+C38h] [rbp-178h]
  char v79; // [rsp+C40h] [rbp-170h] BYREF

  sub_F1FCA0((__int64)v48, a2, a3, a4, a5, a6);
  sub_F1FB80((__int64)v53, (__int64)v48);
  sub_F1FCA0((__int64)v38, a1, v7, v8, v9, v10);
  sub_F1FB80((__int64)v43, (__int64)v38);
  sub_F1FCA0((__int64)v63, (__int64)v53, v11, v12, v13, v14);
  sub_F1FCA0((__int64)v58, (__int64)v43, v15, v16, v17, v18);
  sub_F1FCA0((__int64)v74, (__int64)v63, v19, v20, v21, v22);
  sub_F1FCA0((__int64)v68, (__int64)v58, v23, v24, v25, v26);
LABEL_2:
  v29 = v72;
  while ( 1 )
  {
    v30 = 40 * v29;
    if ( v29 == v78 )
      break;
LABEL_7:
    v34 = *(unsigned int *)(a3 + 8);
    v35 = *(_QWORD *)&v71[v30 - 8];
    if ( v34 + 1 > (unsigned __int64)*(unsigned int *)(a3 + 12) )
    {
      sub_C8D5F0(a3, (const void *)(a3 + 16), v34 + 1, 8u, v27, v28);
      v34 = *(unsigned int *)(a3 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a3 + 8 * v34) = v35;
    v36 = v72;
    ++*(_DWORD *)(a3 + 8);
    v29 = v36 - 1;
    v72 = v29;
    if ( (_DWORD)v29 )
    {
      sub_D4D230((__int64)v68);
      goto LABEL_2;
    }
  }
  v31 = &v71[v30];
  v32 = v77;
  if ( v71 != &v71[v30] )
  {
    v33 = v71;
    do
    {
      v27 = *((_QWORD *)v32 + 4);
      if ( *((_QWORD *)v33 + 4) != v27 )
        goto LABEL_7;
      v28 = *((unsigned int *)v32 + 6);
      if ( *((_DWORD *)v33 + 6) != (_DWORD)v28 || *((_DWORD *)v33 + 2) != *((_DWORD *)v32 + 2) )
        goto LABEL_7;
      v33 += 40;
      v32 += 40;
    }
    while ( v31 != v33 );
  }
  if ( v71 != &v73 )
    _libc_free(v71, v31);
  if ( !v70 )
    _libc_free(v69, v31);
  if ( v77 != &v79 )
    _libc_free(v77, v31);
  if ( !v76 )
    _libc_free(v75, v31);
  if ( v61 != &v62 )
    _libc_free(v61, v31);
  if ( !v60 )
    _libc_free(v59, v31);
  if ( v66 != &v67 )
    _libc_free(v66, v31);
  if ( !v65 )
    _libc_free(v64, v31);
  if ( v46 != &v47 )
    _libc_free(v46, v31);
  if ( !v45 )
    _libc_free(v44, v31);
  if ( v41 != &v42 )
    _libc_free(v41, v31);
  if ( !v40 )
    _libc_free(v39, v31);
  if ( v56 != &v57 )
    _libc_free(v56, v31);
  if ( !v55 )
    _libc_free(v54, v31);
  if ( v51 != &v52 )
    _libc_free(v51, v31);
  if ( !v50 )
    _libc_free(v49, v31);
  return a3;
}
