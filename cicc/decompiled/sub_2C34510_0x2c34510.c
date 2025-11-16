// Function: sub_2C34510
// Address: 0x2c34510
//
__int64 __fastcall sub_2C34510(__int64 *a1)
{
  __int64 *v1; // r12
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
  __int64 v23; // r8
  __int64 v24; // r9
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rcx
  _BYTE *v28; // rsi
  __int64 v29; // r15
  __int64 v30; // rax
  __int64 v31; // rax
  unsigned __int64 v32; // rdx
  __int64 v33; // rcx
  char v34; // di
  __int64 *v35; // rdi
  int v36; // eax
  __int64 v37; // r15
  __int64 v38; // rbx
  __int64 v39; // rax
  __int64 v40; // rbx
  __int64 v41; // r12
  __int64 v42; // rdi
  _QWORD *v43; // rdi
  __int64 v44; // rsi
  _QWORD *v45; // rdi
  int v46; // r8d
  _QWORD *v47; // rdi
  __int64 v48; // rsi
  _QWORD *v49; // rax
  int v50; // r8d
  __int64 v51; // r9
  __int64 v52; // rax
  unsigned __int64 v53; // r14
  const void *v54; // r8
  __int64 v55; // r12
  __int64 *v56; // rdi
  __int64 v57; // r14
  _QWORD *v58; // rdi
  __int64 v59; // rsi
  _QWORD *v60; // rax
  int v61; // r10d
  _QWORD *v62; // rdi
  __int64 v63; // rsi
  _QWORD *v64; // rax
  __int64 v65; // r8
  __int64 v66; // r9
  int v67; // r10d
  __int64 v68; // rax
  __int64 v69; // rax
  __int64 *v71; // [rsp+8h] [rbp-4E8h]
  __int64 *v72; // [rsp+30h] [rbp-4C0h]
  __int64 v73; // [rsp+38h] [rbp-4B8h]
  __int64 *v74; // [rsp+38h] [rbp-4B8h]
  const void *v75; // [rsp+38h] [rbp-4B8h]
  __int64 *v76; // [rsp+40h] [rbp-4B0h] BYREF
  __int64 v77; // [rsp+48h] [rbp-4A8h]
  _BYTE v78[48]; // [rsp+50h] [rbp-4A0h] BYREF
  _QWORD v79[12]; // [rsp+80h] [rbp-470h] BYREF
  __int64 v80; // [rsp+E0h] [rbp-410h]
  __int64 v81; // [rsp+E8h] [rbp-408h]
  __int16 v82; // [rsp+F8h] [rbp-3F8h]
  _QWORD v83[12]; // [rsp+100h] [rbp-3F0h] BYREF
  _BYTE *v84; // [rsp+160h] [rbp-390h]
  __int64 v85; // [rsp+168h] [rbp-388h]
  __int16 v86; // [rsp+178h] [rbp-378h]
  __int16 v87; // [rsp+188h] [rbp-368h]
  __int64 v88[12]; // [rsp+190h] [rbp-360h] BYREF
  _BYTE *v89; // [rsp+1F0h] [rbp-300h]
  __int64 v90; // [rsp+1F8h] [rbp-2F8h]
  __int16 v91; // [rsp+208h] [rbp-2E8h] BYREF
  _QWORD v92[15]; // [rsp+210h] [rbp-2E0h] BYREF
  __int16 v93; // [rsp+288h] [rbp-268h]
  __int16 v94; // [rsp+298h] [rbp-258h]
  __int64 *v95; // [rsp+2A0h] [rbp-250h] BYREF
  __int64 v96; // [rsp+2A8h] [rbp-248h]
  _BYTE v97[104]; // [rsp+2B0h] [rbp-240h] BYREF
  __int16 v98; // [rsp+318h] [rbp-1D8h]
  _BYTE v99[120]; // [rsp+320h] [rbp-1D0h] BYREF
  __int16 v100; // [rsp+398h] [rbp-158h]
  __int16 v101; // [rsp+3A8h] [rbp-148h]
  _BYTE v102[120]; // [rsp+3B0h] [rbp-140h] BYREF
  __int16 v103; // [rsp+428h] [rbp-C8h]
  _BYTE v104[120]; // [rsp+430h] [rbp-C0h] BYREF
  __int16 v105; // [rsp+4A8h] [rbp-48h]
  __int16 v106; // [rsp+4B8h] [rbp-38h]

  v1 = (__int64 *)v104;
  v2 = *a1;
  v76 = (__int64 *)v78;
  v77 = 0x600000000LL;
  sub_2C2F4B0(v88, v2);
  sub_2C31060((__int64)&v95, (__int64)v88, v3, v4, v5, v6);
  sub_2AB1B50((__int64)&v91);
  sub_2AB1B50((__int64)v88);
  sub_2ABCC20(v79, (__int64)&v95, v7, v8, v9, v10);
  v82 = v98;
  sub_2ABCC20(v83, (__int64)v99, v11, v12, v13, v14);
  v86 = v100;
  v87 = v101;
  sub_2ABCC20(v88, (__int64)v102, v15, v16, v17, v18);
  v91 = v103;
  sub_2ABCC20(v92, (__int64)v104, v19, v20, v21, v22);
  v25 = v80;
  v93 = v105;
  v94 = v106;
  v26 = v81;
LABEL_3:
  v28 = v89;
  v27 = v90 - (_QWORD)v89;
  if ( v26 - v25 != v90 - (_QWORD)v89 )
  {
LABEL_4:
    v29 = *(_QWORD *)(v26 - 32);
    if ( *(_QWORD *)(v29 + 48) )
    {
      if ( *(_DWORD *)(v29 + 64) == 1 )
      {
        v30 = **(_QWORD **)(v29 + 56);
        if ( v30 )
        {
          v27 = *(unsigned __int8 *)(v30 + 8);
          v25 = (unsigned int)(v27 - 1);
          if ( (unsigned int)v25 <= 1 && *(_DWORD *)(v30 + 88) == 1 && (_BYTE)v27 != 2 )
          {
            v31 = (unsigned int)v77;
            v27 = HIDWORD(v77);
            v32 = (unsigned int)v77 + 1LL;
            if ( v32 > HIDWORD(v77) )
            {
              v28 = v78;
              sub_C8D5F0((__int64)&v76, v78, v32, 8u, v23, v24);
              v31 = (unsigned int)v77;
            }
            v25 = (__int64)v76;
            v76[v31] = v29;
            LODWORD(v77) = v77 + 1;
          }
        }
      }
    }
    while ( 1 )
    {
      sub_2AD7320((__int64)v79, (__int64)v28, v25, v27, v23, v24);
      v26 = v81;
      v25 = v80;
      v28 = v84;
      if ( v81 - v80 == v85 - (_QWORD)v84 )
      {
        if ( v80 == v81 )
          goto LABEL_3;
        v33 = v80;
        while ( *(_QWORD *)v33 == *(_QWORD *)v28 )
        {
          v34 = *(_BYTE *)(v33 + 24);
          if ( v34 != v28[24]
            || v34 && (*(_QWORD *)(v33 + 8) != *((_QWORD *)v28 + 1) || *(_QWORD *)(v33 + 16) != *((_QWORD *)v28 + 2)) )
          {
            break;
          }
          v33 += 32;
          v28 += 32;
          if ( v81 == v33 )
            goto LABEL_3;
        }
      }
      v27 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v81 - 32) + 8LL) - 1;
      if ( (unsigned int)v27 <= 1 )
        goto LABEL_3;
    }
  }
  while ( v25 != v26 )
  {
    v27 = *(_QWORD *)v28;
    if ( *(_QWORD *)v25 != *(_QWORD *)v28 )
      goto LABEL_4;
    v27 = *(unsigned __int8 *)(v25 + 24);
    if ( (_BYTE)v27 != v28[24] )
      goto LABEL_4;
    if ( (_BYTE)v27 )
    {
      v27 = *((_QWORD *)v28 + 1);
      if ( *(_QWORD *)(v25 + 8) != v27 )
        goto LABEL_4;
      v27 = *((_QWORD *)v28 + 2);
      if ( *(_QWORD *)(v25 + 16) != v27 )
        goto LABEL_4;
    }
    v25 += 32;
    v28 += 32;
  }
  sub_2AB1B50((__int64)v92);
  sub_2AB1B50((__int64)v88);
  sub_2AB1B50((__int64)v83);
  sub_2AB1B50((__int64)v79);
  sub_2AB1B50((__int64)v104);
  sub_2AB1B50((__int64)v102);
  sub_2AB1B50((__int64)v99);
  sub_2AB1B50((__int64)&v95);
  v35 = v76;
  v36 = v77;
  v71 = &v76[(unsigned int)v77];
  if ( v71 == v76 )
    goto LABEL_65;
  v72 = v76;
  do
  {
    v37 = 0;
    v38 = *v72;
    if ( *(_DWORD *)(*v72 + 64) == 1 )
      v37 = **(_QWORD **)(v38 + 56);
    v39 = v38 + 112;
    if ( v38 + 112 != *(_QWORD *)(v38 + 120) )
    {
      v73 = *v72;
      v40 = *(_QWORD *)(v38 + 120);
      v41 = v39;
      do
      {
        v42 = v40;
        v40 = *(_QWORD *)(v40 + 8);
        sub_2C19EE0((_QWORD *)(v42 - 24), v37, (unsigned __int64 *)(v37 + 112));
      }
      while ( v41 != v40 );
      v38 = v73;
    }
    v95 = (__int64 *)v38;
    v43 = *(_QWORD **)(v37 + 80);
    v44 = (__int64)&v43[*(unsigned int *)(v37 + 88)];
    v45 = sub_2C25750(v43, v44, (__int64 *)&v95);
    if ( v45 + 1 != (_QWORD *)v44 )
    {
      memmove(v45, v45 + 1, v44 - (_QWORD)(v45 + 1));
      v46 = *(_DWORD *)(v37 + 88);
    }
    *(_DWORD *)(v37 + 88) = v46 - 1;
    v95 = (__int64 *)v37;
    v47 = *(_QWORD **)(v38 + 56);
    v48 = (__int64)&v47[*(unsigned int *)(v38 + 64)];
    v49 = sub_2C25750(v47, v48, (__int64 *)&v95);
    if ( v49 + 1 != (_QWORD *)v48 )
    {
      memmove(v49, v49 + 1, v48 - (_QWORD)(v49 + 1));
      v50 = *(_DWORD *)(v38 + 64);
    }
    v52 = *(_QWORD *)(v38 + 48);
    *(_DWORD *)(v38 + 64) = v50 - 1;
    if ( v52 && v38 == *(_QWORD *)(v52 + 120) )
    {
      *(_QWORD *)(v52 + 120) = v37;
      *(_QWORD *)(v37 + 48) = v52;
    }
    v53 = *(unsigned int *)(v38 + 88);
    v54 = *(const void **)(v38 + 80);
    v95 = (__int64 *)v97;
    v55 = 8 * v53;
    v96 = 0x600000000LL;
    if ( v53 > 6 )
    {
      v75 = v54;
      sub_C8D5F0((__int64)&v95, v97, v53, 8u, (__int64)v54, v51);
      v54 = v75;
      v56 = &v95[(unsigned int)v96];
    }
    else
    {
      v56 = (__int64 *)v97;
      if ( !v55 )
        goto LABEL_49;
    }
    memcpy(v56, v54, 8 * v53);
    v56 = v95;
    LODWORD(v55) = v96;
LABEL_49:
    LODWORD(v96) = v53 + v55;
    v1 = (__int64 *)(unsigned int)(v53 + v55);
    v74 = &v56[(_QWORD)v1];
    if ( v74 != v56 )
    {
      v1 = v56;
      do
      {
        v57 = *v1;
        v88[0] = *v1;
        v58 = *(_QWORD **)(v38 + 80);
        v59 = (__int64)&v58[*(unsigned int *)(v38 + 88)];
        v60 = sub_2C25750(v58, v59, v88);
        if ( v60 + 1 != (_QWORD *)v59 )
        {
          memmove(v60, v60 + 1, v59 - (_QWORD)(v60 + 1));
          v61 = *(_DWORD *)(v38 + 88);
        }
        *(_DWORD *)(v38 + 88) = v61 - 1;
        v88[0] = v38;
        v62 = *(_QWORD **)(v57 + 56);
        v63 = (__int64)&v62[*(unsigned int *)(v57 + 64)];
        v64 = sub_2C25750(v62, v63, v88);
        if ( v64 + 1 != (_QWORD *)v63 )
        {
          memmove(v64, v64 + 1, v63 - (_QWORD)(v64 + 1));
          v67 = *(_DWORD *)(v57 + 64);
        }
        *(_DWORD *)(v57 + 64) = v67 - 1;
        v68 = *(unsigned int *)(v37 + 88);
        if ( v68 + 1 > (unsigned __int64)*(unsigned int *)(v37 + 92) )
        {
          sub_C8D5F0(v37 + 80, (const void *)(v37 + 96), v68 + 1, 8u, v65, v66);
          v68 = *(unsigned int *)(v37 + 88);
        }
        *(_QWORD *)(*(_QWORD *)(v37 + 80) + 8 * v68) = v57;
        ++*(_DWORD *)(v37 + 88);
        v69 = *(unsigned int *)(v57 + 64);
        if ( v69 + 1 > (unsigned __int64)*(unsigned int *)(v57 + 68) )
        {
          sub_C8D5F0(v57 + 56, (const void *)(v57 + 72), v69 + 1, 8u, v65, v66);
          v69 = *(unsigned int *)(v57 + 64);
        }
        ++v1;
        *(_QWORD *)(*(_QWORD *)(v57 + 56) + 8 * v69) = v37;
        ++*(_DWORD *)(v57 + 64);
      }
      while ( v74 != v1 );
      v56 = v95;
    }
    if ( v56 != (__int64 *)v97 )
      _libc_free((unsigned __int64)v56);
    ++v72;
  }
  while ( v71 != v72 );
  v35 = v76;
  v36 = v77;
LABEL_65:
  LOBYTE(v1) = v36 != 0;
  if ( v35 != (__int64 *)v78 )
    _libc_free((unsigned __int64)v35);
  return (unsigned int)v1;
}
