// Function: sub_21834F0
// Address: 0x21834f0
//
__int64 __fastcall sub_21834F0(__int64 a1, __int64 a2, int a3)
{
  __int64 v5; // rbx
  __int64 *v6; // rax
  char v7; // al
  __int64 v8; // rcx
  int v9; // r8d
  int v10; // r9d
  _QWORD *v11; // rdx
  __int64 *v12; // rax
  __int64 v13; // r12
  unsigned __int16 v14; // ax
  __int64 v15; // rax
  __int64 v16; // rbx
  __int64 v17; // r13
  int v18; // r11d
  __int64 *v19; // r10
  unsigned int v20; // edx
  __int64 *v21; // rdi
  __int64 v22; // rax
  int v23; // edi
  __int64 v24; // rax
  int v25; // esi
  int v26; // ecx
  __int64 *v27; // rax
  unsigned int v28; // esi
  __int64 v29; // rdx
  unsigned int v30; // eax
  __int64 v31; // rdi
  int v32; // r11d
  _QWORD *v33; // r10
  unsigned int v34; // r13d
  __int64 v35; // rdx
  __int64 v36; // rdi
  __int64 *v37; // rbx
  unsigned __int64 v38; // r12
  __int64 v39; // rdi
  int v41; // esi
  int v42; // eax
  __int64 *v43; // [rsp+8h] [rbp-D8h]
  __int64 v44; // [rsp+28h] [rbp-B8h] BYREF
  __int64 v45; // [rsp+30h] [rbp-B0h] BYREF
  _QWORD *v46; // [rsp+38h] [rbp-A8h] BYREF
  __int64 v47; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v48; // [rsp+48h] [rbp-98h]
  __int64 v49; // [rsp+50h] [rbp-90h]
  __int64 v50; // [rsp+58h] [rbp-88h]
  __int64 v51; // [rsp+60h] [rbp-80h] BYREF
  __int64 v52; // [rsp+68h] [rbp-78h]
  __int64 *v53; // [rsp+70h] [rbp-70h]
  __int64 *v54; // [rsp+78h] [rbp-68h]
  _QWORD *v55; // [rsp+80h] [rbp-60h]
  __int64 *v56; // [rsp+88h] [rbp-58h]
  __int64 *v57; // [rsp+90h] [rbp-50h]
  __int64 *v58; // [rsp+98h] [rbp-48h]
  _QWORD *v59; // [rsp+A0h] [rbp-40h]
  __int64 v60; // [rsp+A8h] [rbp-38h]

  v44 = a2;
  v57 = 0;
  v52 = 8;
  v51 = sub_22077B0(64);
  v5 = v51 + 24;
  v6 = (__int64 *)sub_22077B0(512);
  v56 = (__int64 *)(v51 + 24);
  *(_QWORD *)(v51 + 24) = v6;
  v54 = v6;
  v55 = v6 + 64;
  v60 = v5;
  v58 = v6;
  v59 = v6 + 64;
  v53 = v6;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  if ( v6 )
    *v6 = v44;
  v57 = v6 + 1;
  v7 = sub_217F570((__int64)&v47, &v44, &v46);
  v11 = v46;
  if ( !v7 )
  {
    v41 = v50;
    ++v47;
    v42 = v49 + 1;
    v9 = 2 * v50;
    if ( 4 * ((int)v49 + 1) >= (unsigned int)(3 * v50) )
    {
      v41 = 2 * v50;
    }
    else
    {
      v8 = (unsigned int)(v50 - HIDWORD(v49) - v42);
      if ( (unsigned int)v8 > (unsigned int)v50 >> 3 )
      {
LABEL_68:
        LODWORD(v49) = v42;
        if ( *v11 != -8 )
          --HIDWORD(v49);
        *v11 = v44;
        goto LABEL_4;
      }
    }
    sub_2183380((__int64)&v47, v41);
    sub_217F570((__int64)&v47, &v44, &v46);
    v11 = v46;
    v42 = v49 + 1;
    goto LABEL_68;
  }
LABEL_4:
  v12 = v53;
  if ( v53 == v57 )
  {
LABEL_53:
    v34 = 0;
    goto LABEL_56;
  }
  while ( 1 )
  {
    v13 = *v12;
    if ( v12 == v55 - 1 )
    {
      j_j___libc_free_0(v54, 512);
      v35 = *++v56 + 512;
      v54 = (__int64 *)*v56;
      v55 = (_QWORD *)v35;
      v53 = v54;
    }
    else
    {
      v53 = v12 + 1;
    }
    if ( !v13 )
      goto LABEL_7;
    v14 = **(_WORD **)(v13 + 16);
    if ( v14 == 3358 )
      break;
    if ( v14 > 0xD1Eu )
    {
      if ( (unsigned __int16)(v14 - 3641) <= 1u )
        break;
      v12 = v53;
      if ( v53 == v57 )
        goto LABEL_53;
    }
    else
    {
      if ( v14 <= 0x8Du )
      {
        if ( v14 > 0x8Bu )
          break;
      }
      else if ( v14 == 190 )
      {
        break;
      }
LABEL_7:
      v12 = v53;
      if ( v53 == v57 )
        goto LABEL_53;
    }
  }
  v15 = *(unsigned int *)(v13 + 40);
  if ( !(_DWORD)v15 )
    goto LABEL_7;
  v16 = 0;
  v17 = 40 * v15;
  while ( 1 )
  {
    v22 = v16 + *(_QWORD *)(v13 + 32);
    if ( *(_BYTE *)v22 || (*(_BYTE *)(v22 + 3) & 0x10) != 0 )
      goto LABEL_18;
    v23 = *(_DWORD *)(v22 + 8);
    if ( a3 == v23 )
      break;
    v24 = sub_217D7E0(v23, a1, 0, v8, v9, v10);
    v25 = v50;
    v45 = v24;
    if ( !(_DWORD)v50 )
    {
      ++v47;
LABEL_24:
      v25 = 2 * v50;
LABEL_25:
      sub_2183380((__int64)&v47, v25);
      sub_217F570((__int64)&v47, &v45, &v46);
      v19 = v46;
      v24 = v45;
      v26 = v49 + 1;
      goto LABEL_35;
    }
    v10 = v50 - 1;
    v9 = v48;
    v18 = 1;
    v19 = 0;
    v20 = (v50 - 1) & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
    v21 = (__int64 *)(v48 + 8LL * v20);
    v8 = *v21;
    if ( v24 == *v21 )
      goto LABEL_18;
    while ( v8 != -8 )
    {
      if ( v19 || v8 != -16 )
        v21 = v19;
      v20 = v10 & (v18 + v20);
      v8 = *(_QWORD *)(v48 + 8LL * v20);
      if ( v24 == v8 )
        goto LABEL_18;
      ++v18;
      v19 = v21;
      v21 = (__int64 *)(v48 + 8LL * v20);
    }
    if ( !v19 )
      v19 = v21;
    ++v47;
    v26 = v49 + 1;
    if ( 4 * ((int)v49 + 1) >= (unsigned int)(3 * v50) )
      goto LABEL_24;
    if ( (int)v50 - HIDWORD(v49) - v26 <= (unsigned int)v50 >> 3 )
      goto LABEL_25;
LABEL_35:
    LODWORD(v49) = v26;
    if ( *v19 != -8 )
      --HIDWORD(v49);
    *v19 = v24;
    v27 = v57;
    if ( v57 == v59 - 1 )
    {
      sub_217EE60(&v51, &v45);
      v28 = v50;
      if ( !(_DWORD)v50 )
        goto LABEL_63;
    }
    else
    {
      if ( v57 )
      {
        *v57 = v45;
        v27 = v57;
      }
      v28 = v50;
      v57 = v27 + 1;
      if ( !(_DWORD)v50 )
      {
LABEL_63:
        ++v47;
        goto LABEL_64;
      }
    }
    v29 = v45;
    v10 = v28 - 1;
    v9 = v48;
    v30 = (v28 - 1) & (((unsigned int)v45 >> 9) ^ ((unsigned int)v45 >> 4));
    v8 = v48 + 8LL * v30;
    v31 = *(_QWORD *)v8;
    if ( *(_QWORD *)v8 != v45 )
    {
      v32 = 1;
      v33 = 0;
      while ( v31 != -8 )
      {
        if ( v31 != -16 || v33 )
          v8 = (__int64)v33;
        v30 = v10 & (v32 + v30);
        v43 = (__int64 *)(v48 + 8LL * v30);
        v31 = *v43;
        if ( v45 == *v43 )
          goto LABEL_18;
        ++v32;
        v33 = (_QWORD *)v8;
        v8 = v48 + 8LL * v30;
      }
      if ( !v33 )
        v33 = (_QWORD *)v8;
      ++v47;
      v8 = (unsigned int)(v49 + 1);
      if ( 4 * (int)v8 >= 3 * v28 )
      {
LABEL_64:
        v28 *= 2;
      }
      else if ( v28 - HIDWORD(v49) - (unsigned int)v8 > v28 >> 3 )
      {
LABEL_48:
        LODWORD(v49) = v8;
        if ( *v33 != -8 )
          --HIDWORD(v49);
        *v33 = v29;
        goto LABEL_18;
      }
      sub_2183380((__int64)&v47, v28);
      sub_217F570((__int64)&v47, &v45, &v46);
      v33 = v46;
      v29 = v45;
      v8 = (unsigned int)(v49 + 1);
      goto LABEL_48;
    }
LABEL_18:
    v16 += 40;
    if ( v17 == v16 )
      goto LABEL_7;
  }
  v34 = 1;
LABEL_56:
  j___libc_free_0(v48);
  v36 = v51;
  if ( v51 )
  {
    v37 = v56;
    v38 = v60 + 8;
    if ( v60 + 8 > (unsigned __int64)v56 )
    {
      do
      {
        v39 = *v37++;
        j_j___libc_free_0(v39, 512);
      }
      while ( v38 > (unsigned __int64)v37 );
      v36 = v51;
    }
    j_j___libc_free_0(v36, 8 * v52);
  }
  return v34;
}
