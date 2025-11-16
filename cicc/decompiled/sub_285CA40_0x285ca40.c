// Function: sub_285CA40
// Address: 0x285ca40
//
void __fastcall sub_285CA40(__int64 *a1, _BYTE *a2, __int64 a3)
{
  __int64 v3; // rax
  unsigned __int64 *v4; // r15
  __int64 v5; // rcx
  unsigned __int64 v6; // r14
  unsigned __int64 *v7; // rbx
  int v8; // r13d
  __int64 v9; // r13
  __int64 *v10; // r15
  _QWORD *v11; // rax
  __int64 v12; // r8
  __int64 v13; // rdx
  unsigned __int64 v14; // r9
  unsigned int v15; // edx
  __int64 *v16; // r13
  __int64 v17; // r12
  __int64 *v18; // rax
  _BYTE *v19; // r12
  __int64 *v20; // rax
  __int64 v21; // rdx
  unsigned __int64 v22; // rax
  __int64 v23; // rcx
  __int64 v24; // rcx
  __int64 *v25; // r13
  __int64 v26; // r12
  __int64 *v27; // rax
  __int64 v28; // rsi
  __int64 v29; // rax
  __int64 *v30; // rdi
  __int64 v31; // rdx
  unsigned __int64 v32; // rax
  __int64 v33; // rcx
  __int64 v34; // rcx
  unsigned __int64 *v35; // rdi
  _QWORD *v36; // r12
  __int64 v37; // rsi
  __int64 v38; // rax
  __int64 *v39; // rdi
  __int64 v40; // rdx
  unsigned __int64 v41; // r14
  __int64 v42; // rax
  __int64 v43; // rax
  int v44; // r13d
  __int64 v45; // r15
  __int64 *v46; // r13
  _QWORD *v47; // rax
  __int64 v48; // r8
  __int64 v49; // rdx
  unsigned __int64 v50; // r9
  unsigned int v51; // edx
  __int64 *v52; // r13
  __int64 v53; // r12
  __int64 *v54; // rax
  __int64 v55; // r12
  __int64 *v56; // r13
  __int64 v57; // r12
  __int64 *v58; // rax
  __int64 v59; // rax
  __int64 v60; // r13
  __int64 v61; // rsi
  unsigned __int8 *v62; // rsi
  _QWORD *v63; // r15
  __int64 v64; // rax
  __int64 v65; // rsi
  unsigned __int8 *v66; // rsi
  __int64 v67; // rax
  unsigned __int64 *v68; // rdi
  const void *v69; // r13
  const void *v70; // rdx
  const void *v71; // r15
  __int64 v72; // r9
  int v73; // eax
  __int64 v74; // r8
  __int64 v75; // rax
  unsigned __int64 *v76; // rdi
  const void *v77; // r9
  const void *v78; // rdx
  const void *v79; // r13
  signed __int64 v80; // r15
  int v81; // eax
  __int64 v82; // r8
  const void *v83; // [rsp+0h] [rbp-A0h]
  __int64 v84; // [rsp+0h] [rbp-A0h]
  _QWORD *v85; // [rsp+8h] [rbp-98h]
  _QWORD *v86; // [rsp+8h] [rbp-98h]
  int v89; // [rsp+18h] [rbp-88h]
  int v90; // [rsp+18h] [rbp-88h]
  __int64 v91; // [rsp+18h] [rbp-88h]
  unsigned __int8 *v92; // [rsp+28h] [rbp-78h] BYREF
  unsigned __int64 v93; // [rsp+30h] [rbp-70h] BYREF
  __int64 v94; // [rsp+38h] [rbp-68h]
  _BYTE dest[96]; // [rsp+40h] [rbp-60h] BYREF

  v3 = *a1;
  v4 = *(unsigned __int64 **)a3;
  v5 = *(unsigned int *)(a3 + 8);
  v93 = (unsigned __int64)v4;
  v6 = v3 & 0xFFFFFFFFFFFFFFF8LL;
  v7 = &v4[v5];
  if ( (v3 & 4) == 0 )
  {
    if ( v4 == v7 )
      goto LABEL_63;
    v8 = 0;
    do
    {
      v8 += *v4 == 4101;
      v4 += (unsigned int)sub_AF4160((unsigned __int64 **)&v93);
      v93 = (unsigned __int64)v4;
    }
    while ( v4 != v7 );
    if ( !v8 )
    {
LABEL_63:
      sub_2854010(v6, **(_QWORD **)a2, a3);
    }
    else
    {
      if ( v8 == 1 && **(_QWORD **)a3 == 4101 )
      {
        v75 = sub_285CA30((_QWORD *)a3, 2);
        v93 = (unsigned __int64)dest;
        v76 = (unsigned __int64 *)dest;
        v77 = (const void *)v75;
        v79 = v78;
        v80 = (signed __int64)v78 - v75;
        v94 = 0x600000000LL;
        v81 = 0;
        v82 = v80 >> 3;
        if ( (unsigned __int64)v80 > 0x30 )
        {
          v83 = v77;
          sub_C8D5F0((__int64)&v93, dest, v80 >> 3, 8u, v82, (__int64)v77);
          v81 = v94;
          v77 = v83;
          v82 = v80 >> 3;
          v76 = (unsigned __int64 *)(v93 + 8LL * (unsigned int)v94);
        }
        if ( v79 != v77 )
        {
          v90 = v82;
          memcpy(v76, v77, v80);
          v81 = v94;
          LODWORD(v82) = v90;
        }
        LODWORD(v94) = v82 + v81;
        sub_2854010(v6, **(_QWORD **)a2, (__int64)&v93);
        v35 = (unsigned __int64 *)v93;
        if ( (_BYTE *)v93 == dest )
          goto LABEL_32;
      }
      else
      {
        v94 = 0x300000000LL;
        v93 = (unsigned __int64)dest;
        v9 = *(_QWORD *)a2 + 8LL * *((unsigned int *)a2 + 2);
        if ( *(_QWORD *)a2 == v9 )
        {
          v17 = 0;
          v16 = (__int64 *)dest;
        }
        else
        {
          v10 = *(__int64 **)a2;
          do
          {
            v11 = sub_B98A20(*v10, (__int64)a2);
            v13 = (unsigned int)v94;
            v14 = (unsigned int)v94 + 1LL;
            if ( v14 > HIDWORD(v94) )
            {
              a2 = dest;
              v86 = v11;
              sub_C8D5F0((__int64)&v93, dest, (unsigned int)v94 + 1LL, 8u, v12, v14);
              v13 = (unsigned int)v94;
              v11 = v86;
            }
            ++v10;
            *(_QWORD *)(v93 + 8 * v13) = v11;
            v15 = v94 + 1;
            LODWORD(v94) = v94 + 1;
          }
          while ( (__int64 *)v9 != v10 );
          v16 = (__int64 *)v93;
          v17 = v15;
        }
        v18 = (__int64 *)sub_BD5C60(v6);
        v19 = (_BYTE *)sub_B00B60(v18, v16, v17);
        v20 = (__int64 *)sub_BD5C60(v6);
        v21 = sub_B9F6F0(v20, v19);
        v22 = v6 - 32LL * (*(_DWORD *)(v6 + 4) & 0x7FFFFFF);
        if ( *(_QWORD *)v22 )
        {
          v23 = *(_QWORD *)(v22 + 8);
          **(_QWORD **)(v22 + 16) = v23;
          if ( v23 )
            *(_QWORD *)(v23 + 16) = *(_QWORD *)(v22 + 16);
        }
        *(_QWORD *)v22 = v21;
        if ( v21 )
        {
          v24 = *(_QWORD *)(v21 + 16);
          *(_QWORD *)(v22 + 8) = v24;
          if ( v24 )
            *(_QWORD *)(v24 + 16) = v22 + 8;
          *(_QWORD *)(v22 + 16) = v21 + 16;
          *(_QWORD *)(v21 + 16) = v22;
        }
        v25 = *(__int64 **)a3;
        v26 = *(unsigned int *)(a3 + 8);
        v27 = (__int64 *)sub_BD5C60(v6);
        v28 = sub_B0D000(v27, v25, v26, 0, 1);
        v29 = *(_QWORD *)(v28 + 8);
        v30 = (__int64 *)(v29 & 0xFFFFFFFFFFFFFFF8LL);
        if ( (v29 & 4) != 0 )
          v30 = (__int64 *)*v30;
        v31 = sub_B9F6F0(v30, (_BYTE *)v28);
        v32 = v6 + 32 * (2LL - (*(_DWORD *)(v6 + 4) & 0x7FFFFFF));
        if ( *(_QWORD *)v32 )
        {
          v33 = *(_QWORD *)(v32 + 8);
          **(_QWORD **)(v32 + 16) = v33;
          if ( v33 )
            *(_QWORD *)(v33 + 16) = *(_QWORD *)(v32 + 16);
        }
        *(_QWORD *)v32 = v31;
        if ( v31 )
        {
          v34 = *(_QWORD *)(v31 + 16);
          *(_QWORD *)(v32 + 8) = v34;
          if ( v34 )
            *(_QWORD *)(v34 + 16) = v32 + 8;
          *(_QWORD *)(v32 + 16) = v31 + 16;
          *(_QWORD *)(v31 + 16) = v32;
        }
        v35 = (unsigned __int64 *)v93;
        if ( (_BYTE *)v93 == dest )
          goto LABEL_32;
      }
      _libc_free((unsigned __int64)v35);
    }
LABEL_32:
    v36 = *(_QWORD **)(*(_QWORD *)(v6 + 32 * (2LL - (*(_DWORD *)(v6 + 4) & 0x7FFFFFF))) + 24LL);
    if ( !(unsigned __int8)sub_AF4500(a1[1]) && (unsigned __int8)sub_AF4500((__int64)v36) )
    {
      v93 = 159;
      v37 = sub_B0DED0(v36, &v93, 1);
      v38 = *(_QWORD *)(v37 + 8);
      v39 = (__int64 *)(v38 & 0xFFFFFFFFFFFFFFF8LL);
      if ( (v38 & 4) != 0 )
        v39 = (__int64 *)*v39;
      v40 = sub_B9F6F0(v39, (_BYTE *)v37);
      v41 = 32 * (2LL - (*(_DWORD *)(v6 + 4) & 0x7FFFFFF)) + v6;
      if ( *(_QWORD *)v41 )
      {
        v42 = *(_QWORD *)(v41 + 8);
        **(_QWORD **)(v41 + 16) = v42;
        if ( v42 )
          *(_QWORD *)(v42 + 16) = *(_QWORD *)(v41 + 16);
      }
      *(_QWORD *)v41 = v40;
      if ( v40 )
      {
        v43 = *(_QWORD *)(v40 + 16);
        *(_QWORD *)(v41 + 8) = v43;
        if ( v43 )
          *(_QWORD *)(v43 + 16) = v41 + 8;
        *(_QWORD *)(v41 + 16) = v40 + 16;
        *(_QWORD *)(v40 + 16) = v41;
      }
    }
    return;
  }
  if ( v4 == v7 )
    goto LABEL_64;
  v44 = 0;
  do
  {
    v44 += *v4 == 4101;
    v4 += (unsigned int)sub_AF4160((unsigned __int64 **)&v93);
    v93 = (unsigned __int64)v4;
  }
  while ( v4 != v7 );
  if ( !v44 )
  {
LABEL_64:
    v60 = v6 + 80;
    sub_28514E0(v6, **(_QWORD **)a2, a3);
  }
  else if ( v44 == 1 && **(_QWORD **)a3 == 4101 )
  {
    v67 = sub_285CA30((_QWORD *)a3, 2);
    v93 = (unsigned __int64)dest;
    v68 = (unsigned __int64 *)dest;
    v69 = (const void *)v67;
    v71 = v70;
    v72 = (__int64)v70 - v67;
    v94 = 0x600000000LL;
    v73 = 0;
    v74 = v72 >> 3;
    if ( (unsigned __int64)v72 > 0x30 )
    {
      v84 = v72;
      v91 = v72 >> 3;
      sub_C8D5F0((__int64)&v93, dest, v72 >> 3, 8u, v74, v72);
      v73 = v94;
      v72 = v84;
      LODWORD(v74) = v91;
      v68 = (unsigned __int64 *)(v93 + 8LL * (unsigned int)v94);
    }
    if ( v71 != v69 )
    {
      v89 = v74;
      memcpy(v68, v69, v72);
      v73 = v94;
      LODWORD(v74) = v89;
    }
    LODWORD(v94) = v74 + v73;
    sub_28514E0(v6, **(_QWORD **)a2, (__int64)&v93);
    if ( (_BYTE *)v93 != dest )
      _libc_free(v93);
    v60 = v6 + 80;
  }
  else
  {
    v94 = 0x300000000LL;
    v93 = (unsigned __int64)dest;
    v45 = *(_QWORD *)a2 + 8LL * *((unsigned int *)a2 + 2);
    if ( *(_QWORD *)a2 == v45 )
    {
      v53 = 0;
      v52 = (__int64 *)dest;
    }
    else
    {
      v46 = *(__int64 **)a2;
      do
      {
        v47 = sub_B98A20(*v46, (__int64)a2);
        v49 = (unsigned int)v94;
        v50 = (unsigned int)v94 + 1LL;
        if ( v50 > HIDWORD(v94) )
        {
          a2 = dest;
          v85 = v47;
          sub_C8D5F0((__int64)&v93, dest, (unsigned int)v94 + 1LL, 8u, v48, v50);
          v49 = (unsigned int)v94;
          v47 = v85;
        }
        ++v46;
        *(_QWORD *)(v93 + 8 * v49) = v47;
        v51 = v94 + 1;
        LODWORD(v94) = v94 + 1;
      }
      while ( (__int64 *)v45 != v46 );
      v52 = (__int64 *)v93;
      v53 = v51;
    }
    v54 = (__int64 *)sub_B141C0(v6);
    v55 = sub_B00B60(v54, v52, v53);
    sub_B91340(v6 + 40, 0);
    *(_QWORD *)(v6 + 40) = v55;
    sub_B96F50(v6 + 40, 0);
    v56 = *(__int64 **)a3;
    v57 = *(unsigned int *)(a3 + 8);
    v58 = (__int64 *)sub_B141C0(v6);
    v59 = sub_B0D000(v58, v56, v57, 0, 1);
    v60 = v6 + 80;
    sub_B11F20(&v92, v59);
    v61 = *(_QWORD *)(v6 + 80);
    if ( v61 )
      sub_B91220(v6 + 80, v61);
    v62 = v92;
    *(_QWORD *)(v6 + 80) = v92;
    if ( v62 )
      sub_B976B0((__int64)&v92, v62, v6 + 80);
    if ( (_BYTE *)v93 != dest )
      _libc_free(v93);
  }
  v63 = (_QWORD *)sub_B11F60(v60);
  if ( !(unsigned __int8)sub_AF4500(a1[1]) && (unsigned __int8)sub_AF4500((__int64)v63) )
  {
    v93 = 159;
    v64 = sub_B0DED0(v63, &v93, 1);
    sub_B11F20(&v93, v64);
    v65 = *(_QWORD *)(v6 + 80);
    if ( v65 )
      sub_B91220(v60, v65);
    v66 = (unsigned __int8 *)v93;
    *(_QWORD *)(v6 + 80) = v93;
    if ( v66 )
      sub_B976B0((__int64)&v93, v66, v60);
  }
}
