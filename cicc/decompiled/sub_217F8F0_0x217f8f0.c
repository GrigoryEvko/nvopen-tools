// Function: sub_217F8F0
// Address: 0x217f8f0
//
__int64 __fastcall sub_217F8F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // r15
  __int64 v6; // r13
  __int64 v7; // rax
  int *v8; // rbx
  __int64 v9; // rax
  __int64 v10; // r8
  unsigned int v11; // ecx
  int *v12; // rdi
  int v13; // esi
  _DWORD *v14; // rax
  int v15; // eax
  __int64 v16; // rdx
  int v17; // edi
  int v18; // r10d
  __int64 v19; // r11
  unsigned int v20; // ebx
  int v21; // r14d
  unsigned int v22; // r13d
  unsigned __int64 v23; // rax
  unsigned int v24; // esi
  unsigned int v25; // r8d
  unsigned int v26; // ecx
  __int64 v27; // rdx
  unsigned int *v28; // rax
  unsigned int v29; // edi
  unsigned int *v30; // rdx
  unsigned int v31; // eax
  unsigned int v32; // esi
  unsigned int v33; // esi
  unsigned __int64 v34; // r14
  _BYTE *v35; // rbx
  int *v36; // rsi
  int v38; // r10d
  int *v40; // [rsp+38h] [rbp-118h]
  int v41; // [rsp+48h] [rbp-108h] BYREF
  int v42; // [rsp+4Ch] [rbp-104h] BYREF
  __int64 v43; // [rsp+50h] [rbp-100h] BYREF
  __int64 v44; // [rsp+58h] [rbp-F8h]
  __int64 v45; // [rsp+60h] [rbp-F0h]
  __int64 v46; // [rsp+68h] [rbp-E8h]
  __int64 v47; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v48; // [rsp+78h] [rbp-D8h]
  __int64 v49; // [rsp+80h] [rbp-D0h]
  __int64 v50; // [rsp+88h] [rbp-C8h]
  _BYTE *v51; // [rsp+90h] [rbp-C0h] BYREF
  __int64 v52; // [rsp+98h] [rbp-B8h]
  _BYTE v53[16]; // [rsp+A0h] [rbp-B0h] BYREF
  int *v54; // [rsp+B0h] [rbp-A0h] BYREF
  __int64 v55; // [rsp+B8h] [rbp-98h]
  _BYTE v56[16]; // [rsp+C0h] [rbp-90h] BYREF
  _BYTE *v57; // [rsp+D0h] [rbp-80h] BYREF
  __int64 v58; // [rsp+D8h] [rbp-78h]
  _BYTE v59[16]; // [rsp+E0h] [rbp-70h] BYREF
  _BYTE *v60; // [rsp+F0h] [rbp-60h] BYREF
  __int64 v61; // [rsp+F8h] [rbp-58h]
  _BYTE v62[16]; // [rsp+100h] [rbp-50h] BYREF
  char v63; // [rsp+110h] [rbp-40h]

  v51 = v53;
  v54 = (int *)v56;
  v4 = *(unsigned int *)(a2 + 40);
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v52 = 0x400000000LL;
  v55 = 0x400000000LL;
  if ( !(_DWORD)v4 )
  {
    v58 = 0x400000000LL;
    v57 = v59;
    v60 = v62;
    v61 = 0x400000000LL;
    goto LABEL_53;
  }
  v5 = 0;
  v6 = 40 * v4;
  do
  {
    while ( 1 )
    {
      v7 = v5 + *(_QWORD *)(a2 + 32);
      if ( !*(_BYTE *)v7 )
      {
        LODWORD(v57) = *(_DWORD *)(v7 + 8);
        if ( (*(_BYTE *)(v7 + 3) & 0x10) == 0 )
        {
          sub_217F7B0((__int64)&v60, (__int64)&v47, (int *)&v57);
          if ( v63 )
            sub_1525B90((__int64)&v54, &v57);
          goto LABEL_5;
        }
        sub_217F7B0((__int64)&v60, (__int64)&v43, (int *)&v57);
        if ( v63 )
          break;
      }
LABEL_5:
      v5 += 40;
      if ( v5 == v6 )
        goto LABEL_10;
    }
    v5 += 40;
    sub_1525B90((__int64)&v51, &v57);
  }
  while ( v5 != v6 );
LABEL_10:
  v8 = v54;
  v60 = v62;
  v57 = v59;
  v40 = &v54[(unsigned int)v55];
  v58 = 0x400000000LL;
  v61 = 0x400000000LL;
  if ( v54 == v40 )
  {
LABEL_53:
    v22 = 0;
    goto LABEL_39;
  }
  while ( 2 )
  {
    while ( 1 )
    {
      v15 = *v8;
      v41 = v15;
      if ( v15 >= 0
        || (v16 = *(_QWORD *)(a1 + 72),
            *(_DWORD *)(*(_QWORD *)(v16 + 280)
                      + 24LL
                      * (*(unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 80) + 24LL)
                                                                     + 16LL * (v15 & 0x7FFFFFFF))
                                                         & 0xFFFFFFFFFFFFFFF8LL)
                                             + 24LL)
                       + *(_DWORD *)(v16 + 288)
                       * (unsigned int)((__int64)(*(_QWORD *)(v16 + 264) - *(_QWORD *)(v16 + 256)) >> 3))) <= 0x20u) )
      {
        v42 = 1;
      }
      else
      {
        v42 = 2;
      }
      sub_1525B90((__int64)&v60, &v42);
      v9 = *(unsigned int *)(a3 + 24);
      if ( !(_DWORD)v9 )
        break;
      v10 = *(_QWORD *)(a3 + 8);
      v11 = (v9 - 1) & (37 * v41);
      v12 = (int *)(v10 + 8LL * v11);
      v13 = *v12;
      if ( v41 != *v12 )
      {
        v17 = 1;
        while ( v13 != -1 )
        {
          v38 = v17 + 1;
          v11 = (v9 - 1) & (v17 + v11);
          v12 = (int *)(v10 + 8LL * v11);
          v13 = *v12;
          if ( v41 == *v12 )
            goto LABEL_15;
          v17 = v38;
        }
        break;
      }
LABEL_15:
      if ( v12 == (int *)(v10 + 8 * v9) )
        break;
      ++v8;
      v14 = sub_1E49390(a3, &v41);
      sub_1525B90((__int64)&v57, v14 + 1);
      if ( v8 == v40 )
        goto LABEL_23;
    }
    ++v8;
    sub_1525B90((__int64)&v57, &v42);
    if ( v8 != v40 )
      continue;
    break;
  }
LABEL_23:
  v18 = v58;
  if ( !(_DWORD)v58 )
    goto LABEL_53;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v22 = 0;
  while ( 1 )
  {
    v23 = (unsigned __int64)v57;
    ++v20;
    v24 = *(_DWORD *)&v57[v19];
    v25 = *(_DWORD *)&v60[v19];
    if ( v20 == v18 )
      break;
    v26 = v20;
    while ( 1 )
    {
      v27 = 4LL * v26;
      v28 = (unsigned int *)(v27 + v23);
      v29 = *v28;
      if ( *v28 > v24 || v29 == v24 && *(_DWORD *)&v60[4 * v26] < v25 )
        break;
      if ( ++v26 == v18 )
        goto LABEL_32;
LABEL_29:
      v23 = (unsigned __int64)v57;
    }
    *v28 = v24;
    v30 = (unsigned int *)&v60[v27];
    v24 = v29;
    v31 = *v30;
    *v30 = v25;
    v25 = v31;
    if ( ++v26 != v18 )
      goto LABEL_29;
LABEL_32:
    *(_DWORD *)&v60[v19] = v25;
    *(_DWORD *)&v57[v19] = v24;
    v32 = v21 + v24;
    if ( v22 < v32 )
      v22 = v32;
    v21 += v25;
    v19 += 4;
  }
  v33 = v21 + v24;
  if ( v33 >= v22 )
    v22 = v33;
LABEL_39:
  v34 = (unsigned __int64)v51;
  v35 = &v51[4 * (unsigned int)v52];
  if ( v35 != v51 )
  {
    do
    {
      v36 = (int *)v34;
      v34 += 4LL;
      sub_1E49390(a3, v36)[1] = v22;
    }
    while ( (_BYTE *)v34 != v35 );
  }
  if ( v60 != v62 )
    _libc_free((unsigned __int64)v60);
  if ( v57 != v59 )
    _libc_free((unsigned __int64)v57);
  if ( v54 != (int *)v56 )
    _libc_free((unsigned __int64)v54);
  if ( v51 != v53 )
    _libc_free((unsigned __int64)v51);
  j___libc_free_0(v48);
  j___libc_free_0(v44);
  return v22;
}
