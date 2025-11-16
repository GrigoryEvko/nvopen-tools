// Function: sub_2D06F50
// Address: 0x2d06f50
//
void __fastcall sub_2D06F50(__int64 a1, __int64 a2)
{
  __int64 v2; // rsi
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r9
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
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // r8
  __int64 v34; // r9
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rax
  __int64 v38; // rdx
  char *v39; // rcx
  char *v40; // rax
  __int64 v41; // rax
  __int64 v42; // r15
  unsigned int v43; // eax
  _BYTE v44[8]; // [rsp+0h] [rbp-17D0h] BYREF
  unsigned __int64 v45; // [rsp+8h] [rbp-17C8h]
  char v46; // [rsp+1Ch] [rbp-17B4h]
  char *v47; // [rsp+60h] [rbp-1770h]
  char v48; // [rsp+70h] [rbp-1760h] BYREF
  unsigned __int64 v49[54]; // [rsp+1B0h] [rbp-1620h] BYREF
  _BYTE v50[8]; // [rsp+360h] [rbp-1470h] BYREF
  unsigned __int64 v51; // [rsp+368h] [rbp-1468h]
  char v52; // [rsp+37Ch] [rbp-1454h]
  char *v53; // [rsp+3C0h] [rbp-1410h]
  char v54; // [rsp+3D0h] [rbp-1400h] BYREF
  _BYTE v55[8]; // [rsp+510h] [rbp-12C0h] BYREF
  unsigned __int64 v56; // [rsp+518h] [rbp-12B8h]
  char v57; // [rsp+52Ch] [rbp-12A4h]
  char *v58; // [rsp+570h] [rbp-1260h]
  char v59; // [rsp+580h] [rbp-1250h] BYREF
  _BYTE v60[8]; // [rsp+6C0h] [rbp-1110h] BYREF
  unsigned __int64 v61; // [rsp+6C8h] [rbp-1108h]
  char v62; // [rsp+6DCh] [rbp-10F4h]
  char *v63; // [rsp+720h] [rbp-10B0h]
  char v64; // [rsp+730h] [rbp-10A0h] BYREF
  _BYTE v65[8]; // [rsp+870h] [rbp-F60h] BYREF
  unsigned __int64 v66; // [rsp+878h] [rbp-F58h]
  char v67; // [rsp+88Ch] [rbp-F44h]
  char *v68; // [rsp+8D0h] [rbp-F00h]
  char v69; // [rsp+8E0h] [rbp-EF0h] BYREF
  _BYTE v70[8]; // [rsp+A20h] [rbp-DB0h] BYREF
  unsigned __int64 v71; // [rsp+A28h] [rbp-DA8h]
  char v72; // [rsp+A3Ch] [rbp-D94h]
  char *v73; // [rsp+A80h] [rbp-D50h]
  char v74; // [rsp+A90h] [rbp-D40h] BYREF
  _BYTE v75[8]; // [rsp+BD0h] [rbp-C00h] BYREF
  unsigned __int64 v76; // [rsp+BD8h] [rbp-BF8h]
  char v77; // [rsp+BECh] [rbp-BE4h]
  char *v78; // [rsp+C30h] [rbp-BA0h]
  char v79; // [rsp+C40h] [rbp-B90h] BYREF
  _BYTE v80[8]; // [rsp+D80h] [rbp-A50h] BYREF
  unsigned __int64 v81; // [rsp+D88h] [rbp-A48h]
  char v82; // [rsp+D9Ch] [rbp-A34h]
  char *v83; // [rsp+DE0h] [rbp-9F0h]
  char v84; // [rsp+DF0h] [rbp-9E0h] BYREF
  _BYTE v85[8]; // [rsp+F30h] [rbp-8A0h] BYREF
  unsigned __int64 v86; // [rsp+F38h] [rbp-898h]
  char v87; // [rsp+F4Ch] [rbp-884h]
  char *v88; // [rsp+F90h] [rbp-840h]
  char v89; // [rsp+FA0h] [rbp-830h] BYREF
  _BYTE v90[8]; // [rsp+10E0h] [rbp-6F0h] BYREF
  unsigned __int64 v91; // [rsp+10E8h] [rbp-6E8h]
  char v92; // [rsp+10FCh] [rbp-6D4h]
  char *v93; // [rsp+1140h] [rbp-690h]
  char v94; // [rsp+1150h] [rbp-680h] BYREF
  _BYTE v95[8]; // [rsp+1290h] [rbp-540h] BYREF
  unsigned __int64 v96; // [rsp+1298h] [rbp-538h]
  char v97; // [rsp+12ACh] [rbp-524h]
  char *v98; // [rsp+12F0h] [rbp-4E0h]
  char v99; // [rsp+1300h] [rbp-4D0h] BYREF
  _BYTE v100[8]; // [rsp+1440h] [rbp-390h] BYREF
  unsigned __int64 v101; // [rsp+1448h] [rbp-388h]
  char v102; // [rsp+145Ch] [rbp-374h]
  char *v103; // [rsp+14A0h] [rbp-330h]
  unsigned int v104; // [rsp+14A8h] [rbp-328h]
  char v105; // [rsp+14B0h] [rbp-320h] BYREF
  _BYTE v106[8]; // [rsp+15F0h] [rbp-1E0h] BYREF
  unsigned __int64 v107; // [rsp+15F8h] [rbp-1D8h]
  char v108; // [rsp+160Ch] [rbp-1C4h]
  char *v109; // [rsp+1650h] [rbp-180h]
  int v110; // [rsp+1658h] [rbp-178h]
  char v111; // [rsp+1660h] [rbp-170h] BYREF

  v2 = *(_QWORD *)(a2 + 80);
  memset(v49, 0, sizeof(v49));
  LODWORD(v49[2]) = 8;
  v49[1] = (unsigned __int64)&v49[4];
  if ( v2 )
    v2 -= 24;
  BYTE4(v49[3]) = 1;
  v49[12] = (unsigned __int64)&v49[14];
  HIDWORD(v49[13]) = 8;
  sub_CE3280((__int64)v44, v2);
  sub_CE3710((__int64)v60, (__int64)v49, v3, v4, v5, v6);
  sub_CE35F0((__int64)v65, (__int64)v60);
  sub_CE3710((__int64)v50, (__int64)v44, v7, v8, v9, v10);
  sub_CE35F0((__int64)v55, (__int64)v50);
  sub_CE3710((__int64)v80, (__int64)v65, v11, v12, v13, v14);
  sub_CE35F0((__int64)v85, (__int64)v80);
  sub_CE3710((__int64)v70, (__int64)v55, v15, v16, v17, v18);
  sub_CE35F0((__int64)v75, (__int64)v70);
  sub_CE3710((__int64)v95, (__int64)v85, v19, v20, v21, v22);
  sub_CE3710((__int64)v90, (__int64)v75, v23, v24, v25, v26);
  sub_CE3710((__int64)v106, (__int64)v95, v27, v28, v29, v30);
  sub_CE3710((__int64)v100, (__int64)v90, v31, v32, v33, v34);
LABEL_4:
  v37 = v104;
  while ( 1 )
  {
    v38 = 40 * v37;
    if ( v37 == v110 )
      break;
LABEL_9:
    v41 = *(unsigned int *)(a1 + 8);
    v42 = *(_QWORD *)&v103[v38 - 8];
    if ( v41 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
    {
      sub_C8D5F0(a1, (const void *)(a1 + 16), v41 + 1, 8u, v35, v36);
      v41 = *(unsigned int *)(a1 + 8);
    }
    *(_QWORD *)(*(_QWORD *)a1 + 8 * v41) = v42;
    v43 = v104;
    ++*(_DWORD *)(a1 + 8);
    v37 = v43 - 1;
    v104 = v37;
    if ( (_DWORD)v37 )
    {
      sub_CE27D0((__int64)v100);
      goto LABEL_4;
    }
  }
  v39 = v109;
  if ( v103 != &v103[v38] )
  {
    v40 = v103;
    do
    {
      v35 = *((_QWORD *)v39 + 4);
      if ( *((_QWORD *)v40 + 4) != v35 )
        goto LABEL_9;
      v36 = *((unsigned int *)v39 + 6);
      if ( *((_DWORD *)v40 + 6) != (_DWORD)v36 || *((_DWORD *)v40 + 2) != *((_DWORD *)v39 + 2) )
        goto LABEL_9;
      v40 += 40;
      v39 += 40;
    }
    while ( &v103[v38] != v40 );
  }
  if ( v103 != &v105 )
    _libc_free((unsigned __int64)v103);
  if ( !v102 )
    _libc_free(v101);
  if ( v109 != &v111 )
    _libc_free((unsigned __int64)v109);
  if ( !v108 )
    _libc_free(v107);
  if ( v93 != &v94 )
    _libc_free((unsigned __int64)v93);
  if ( !v92 )
    _libc_free(v91);
  if ( v98 != &v99 )
    _libc_free((unsigned __int64)v98);
  if ( !v97 )
    _libc_free(v96);
  if ( v78 != &v79 )
    _libc_free((unsigned __int64)v78);
  if ( !v77 )
    _libc_free(v76);
  if ( v73 != &v74 )
    _libc_free((unsigned __int64)v73);
  if ( !v72 )
    _libc_free(v71);
  if ( v88 != &v89 )
    _libc_free((unsigned __int64)v88);
  if ( !v87 )
    _libc_free(v86);
  if ( v83 != &v84 )
    _libc_free((unsigned __int64)v83);
  if ( !v82 )
    _libc_free(v81);
  if ( v58 != &v59 )
    _libc_free((unsigned __int64)v58);
  if ( !v57 )
    _libc_free(v56);
  if ( v53 != &v54 )
    _libc_free((unsigned __int64)v53);
  if ( !v52 )
    _libc_free(v51);
  if ( v68 != &v69 )
    _libc_free((unsigned __int64)v68);
  if ( !v67 )
    _libc_free(v66);
  if ( v63 != &v64 )
    _libc_free((unsigned __int64)v63);
  if ( !v62 )
    _libc_free(v61);
  if ( v47 != &v48 )
    _libc_free((unsigned __int64)v47);
  if ( !v46 )
    _libc_free(v45);
  if ( (unsigned __int64 *)v49[12] != &v49[14] )
    _libc_free(v49[12]);
  if ( !BYTE4(v49[3]) )
    _libc_free(v49[1]);
}
