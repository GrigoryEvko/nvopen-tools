// Function: sub_3085D00
// Address: 0x3085d00
//
__int64 *__fastcall sub_3085D00(__int64 a1, __int64 a2, __int64 a3, __int64 *a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r11
  __int64 v7; // r13
  __int64 v8; // r14
  unsigned __int64 v11; // rax
  char v12; // al
  __int64 v13; // r15
  char v14; // al
  char *v15; // rcx
  __int64 v16; // rdx
  int v17; // edx
  __int64 v18; // rax
  __int64 v19; // rdx
  int v20; // eax
  __int64 v21; // r9
  __int64 v22; // rdx
  int v23; // r15d
  __int64 v24; // r13
  __int64 v25; // rax
  int v26; // r8d
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 v29; // r9
  __int64 v30; // rdx
  int v31; // eax
  int v33; // edx
  __int64 v34; // rax
  __int64 v35; // r9
  unsigned int v36; // ecx
  __int64 v37; // rdx
  __int64 v38; // r13
  __int64 v39; // r14
  int v40; // eax
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // r9
  __int64 v44; // rdx
  char v45; // al
  __int64 v46; // rdx
  int v47; // eax
  const char *v48; // rdx
  int v49; // ecx
  unsigned __int64 v50; // rax
  char v51; // cl
  _QWORD *v52; // rax
  _BYTE *v53; // r13
  unsigned int *v54; // r15
  __int64 v55; // r12
  __int64 v56; // rdx
  unsigned int v57; // esi
  _BYTE *v58; // rsi
  char v59; // al
  __int64 v60; // rdi
  const void *v61; // r15
  __int64 v62; // r10
  __int64 v63; // r12
  __int64 (__fastcall *v64)(__int64, _BYTE *, _BYTE *, __int64, __int64); // rax
  __int64 v65; // rax
  __int64 v66; // r14
  _QWORD *v67; // rax
  __int64 v68; // r12
  unsigned int *v69; // rbx
  unsigned int *v70; // r12
  __int64 v71; // rdx
  unsigned int v72; // esi
  __int64 v73; // rax
  __int64 v74; // [rsp+0h] [rbp-F0h]
  __int64 v75; // [rsp+0h] [rbp-F0h]
  __int64 v76; // [rsp+18h] [rbp-D8h]
  unsigned int v77; // [rsp+24h] [rbp-CCh]
  char v78; // [rsp+24h] [rbp-CCh]
  __int64 v79; // [rsp+28h] [rbp-C8h]
  __int64 v80; // [rsp+28h] [rbp-C8h]
  int v81; // [rsp+30h] [rbp-C0h]
  char *v84; // [rsp+48h] [rbp-A8h]
  __int64 v85; // [rsp+48h] [rbp-A8h]
  __int64 v86; // [rsp+48h] [rbp-A8h]
  __int64 v87; // [rsp+48h] [rbp-A8h]
  _BYTE *v88; // [rsp+58h] [rbp-98h] BYREF
  char *v89; // [rsp+60h] [rbp-90h] BYREF
  __int64 v90; // [rsp+68h] [rbp-88h]
  char *v91; // [rsp+70h] [rbp-80h]
  __int16 v92; // [rsp+80h] [rbp-70h]
  const char *v93; // [rsp+90h] [rbp-60h] BYREF
  __int64 v94; // [rsp+98h] [rbp-58h]
  const char *v95; // [rsp+A0h] [rbp-50h]
  __int16 v96; // [rsp+B0h] [rbp-40h]

  v8 = a1;
  v11 = *(unsigned __int8 *)(a3 + 8);
  v84 = (char *)a5;
  if ( (unsigned __int8)v11 > 3u && (_BYTE)v11 != 5 )
  {
    if ( (unsigned __int8)v11 > 0x14u )
      goto LABEL_70;
    v16 = 1463376;
    if ( !_bittest64(&v16, v11) )
    {
      if ( (_BYTE)v11 == 16 )
      {
        v17 = *(_DWORD *)(a1 + 104) | *(_DWORD *)(a1 + 108);
        v77 = *(_DWORD *)(a1 + 104);
        *(_DWORD *)(a1 + 104) = v17 & -v17;
        v18 = sub_9208B0(*(_QWORD *)a1, *(_QWORD *)(a3 + 24));
        v94 = v19;
        v93 = (const char *)((unsigned __int64)(v18 + 7) >> 3);
        v20 = sub_CA1930(&v93);
        v22 = *(_QWORD *)(a3 + 32);
        v81 = v20;
        if ( (_DWORD)v22 )
        {
          v23 = 0;
          v24 = 0;
          v79 = (unsigned int)v22;
          v25 = *(unsigned int *)(a1 + 24);
          do
          {
            v26 = v24;
            if ( v25 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 28) )
            {
              sub_C8D5F0(a1 + 16, (const void *)(a1 + 32), v25 + 1, 4u, (unsigned int)v24, v21);
              v25 = *(unsigned int *)(a1 + 24);
              v26 = v24;
            }
            *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4 * v25) = v26;
            ++*(_DWORD *)(a1 + 24);
            v27 = sub_BCB2D0(*(_QWORD **)(a2 + 72));
            v28 = sub_ACD640(v27, v24, 0);
            v30 = *(unsigned int *)(a1 + 56);
            if ( v30 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 60) )
            {
              v74 = v28;
              sub_C8D5F0(a1 + 48, (const void *)(a1 + 64), v30 + 1, 8u, v30 + 1, v29);
              v30 = *(unsigned int *)(a1 + 56);
              v28 = v74;
            }
            ++v24;
            *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8 * v30) = v28;
            ++*(_DWORD *)(a1 + 56);
            *(_DWORD *)(a1 + 108) = v23;
            sub_3085D00(a1, a2, *(_QWORD *)(a3 + 24), a4, v84, a6);
            v31 = *(_DWORD *)(a1 + 24);
            --*(_DWORD *)(a1 + 56);
            v23 += v81;
            v25 = (unsigned int)(v31 - 1);
            *(_DWORD *)(a1 + 24) = v25;
          }
          while ( v79 != v24 );
        }
LABEL_20:
        *(_DWORD *)(v8 + 104) = v77;
        return (__int64 *)v77;
      }
      if ( (_BYTE)v11 == 15 )
      {
        v33 = *(_DWORD *)(a1 + 104) | *(_DWORD *)(a1 + 108);
        v77 = *(_DWORD *)(a1 + 104);
        *(_DWORD *)(a1 + 104) = v33 & -v33;
        v34 = sub_AE4AC0(*(_QWORD *)a1, a3);
        v36 = *(_DWORD *)(a3 + 12);
        if ( v36 )
        {
          v37 = *(unsigned int *)(a1 + 24);
          v38 = 0;
          v80 = v36;
          v39 = v34 + 24;
          do
          {
            v40 = v38;
            if ( v37 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 28) )
            {
              sub_C8D5F0(a1 + 16, (const void *)(a1 + 32), v37 + 1, 4u, v37 + 1, v35);
              v37 = *(unsigned int *)(a1 + 24);
              v40 = v38;
            }
            *(_DWORD *)(*(_QWORD *)(a1 + 16) + 4 * v37) = v40;
            ++*(_DWORD *)(a1 + 24);
            v41 = sub_BCB2D0(*(_QWORD **)(a2 + 72));
            v42 = sub_ACD640(v41, v38, 0);
            v44 = *(unsigned int *)(a1 + 56);
            if ( v44 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 60) )
            {
              v75 = v42;
              sub_C8D5F0(a1 + 48, (const void *)(a1 + 64), v44 + 1, 8u, v44 + 1, v43);
              v44 = *(unsigned int *)(a1 + 56);
              v42 = v75;
            }
            v39 += 16;
            *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8 * v44) = v42;
            ++*(_DWORD *)(a1 + 56);
            v45 = *(_BYTE *)(v39 - 8);
            v93 = *(const char **)(v39 - 16);
            LOBYTE(v94) = v45;
            *(_DWORD *)(a1 + 108) = sub_CA1930(&v93);
            v46 = *(_QWORD *)(*(_QWORD *)(a3 + 16) + 8 * v38++);
            sub_3085D00(a1, a2, v46, a4, v84, a6);
            v47 = *(_DWORD *)(a1 + 24);
            --*(_DWORD *)(a1 + 56);
            v37 = (unsigned int)(v47 - 1);
            *(_DWORD *)(a1 + 24) = v37;
          }
          while ( v80 != v38 );
          v8 = a1;
        }
        goto LABEL_20;
      }
LABEL_70:
      BUG();
    }
  }
  v12 = *(_BYTE *)(a5 + 32);
  if ( v12 )
  {
    if ( v12 == 1 )
    {
      v93 = ".ldgsplit";
      v96 = 259;
    }
    else
    {
      if ( *(_BYTE *)(a5 + 33) == 1 )
      {
        v6 = *(_QWORD *)(a5 + 8);
        v48 = *(const char **)a5;
      }
      else
      {
        v48 = (const char *)a5;
        v12 = 2;
      }
      v93 = v48;
      v94 = v6;
      v95 = ".ldgsplit";
      LOBYTE(v96) = v12;
      HIBYTE(v96) = 3;
    }
  }
  else
  {
    v96 = 256;
  }
  v13 = sub_921130(
          (unsigned int **)a2,
          *(_QWORD *)(a1 + 8),
          *(_QWORD *)(a1 + 96),
          *(_BYTE ***)(a1 + 48),
          *(unsigned int *)(a1 + 56),
          (__int64)&v93,
          3u);
  v14 = v84[32];
  if ( v14 )
  {
    if ( v14 == 1 )
    {
      v89 = ".load";
      v92 = 259;
    }
    else
    {
      if ( v84[33] == 1 )
      {
        v7 = *((_QWORD *)v84 + 1);
        v15 = *(char **)v84;
      }
      else
      {
        v15 = v84;
        v14 = 2;
      }
      v89 = v15;
      v90 = v7;
      v91 = ".load";
      LOBYTE(v92) = v14;
      HIBYTE(v92) = 3;
    }
  }
  else
  {
    v92 = 256;
  }
  v49 = *(_DWORD *)(a1 + 104) | *(_DWORD *)(a1 + 108);
  v50 = v49 & (unsigned int)-v49;
  v51 = -1;
  if ( v50 )
  {
    _BitScanReverse64(&v50, v50);
    v51 = 63 - (v50 ^ 0x3F);
  }
  v96 = 257;
  v78 = v51;
  v52 = sub_BD2C40(80, 1u);
  v53 = v52;
  if ( v52 )
    sub_B4D190((__int64)v52, a3, v13, (__int64)&v93, 0, v78, 0, 0);
  (*(void (__fastcall **)(_QWORD, _BYTE *, char **, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
    *(_QWORD *)(a2 + 88),
    v53,
    &v89,
    *(_QWORD *)(a2 + 56),
    *(_QWORD *)(a2 + 64));
  v54 = *(unsigned int **)a2;
  v55 = *(_QWORD *)a2 + 16LL * *(unsigned int *)(a2 + 8);
  if ( *(_QWORD *)a2 != v55 )
  {
    do
    {
      v56 = *((_QWORD *)v54 + 1);
      v57 = *v54;
      v54 += 4;
      sub_B99FD0((__int64)v53, v57, v56);
    }
    while ( (unsigned int *)v55 != v54 );
  }
  v88 = v53;
  v58 = *(_BYTE **)(a6 + 8);
  if ( v58 == *(_BYTE **)(a6 + 16) )
  {
    sub_27D05B0(a6, v58, &v88);
    v53 = v88;
  }
  else
  {
    if ( v58 )
    {
      *(_QWORD *)v58 = v53;
      v53 = v88;
      v58 = *(_BYTE **)(a6 + 8);
    }
    *(_QWORD *)(a6 + 8) = v58 + 8;
  }
  v59 = v84[32];
  if ( v59 )
  {
    if ( v59 == 1 )
    {
      v89 = ".ldgsplitinsert";
      v92 = 259;
    }
    else
    {
      if ( v84[33] == 1 )
      {
        v76 = *((_QWORD *)v84 + 1);
        v84 = *(char **)v84;
      }
      else
      {
        v59 = 2;
      }
      LOBYTE(v92) = v59;
      HIBYTE(v92) = 3;
      v89 = v84;
      v90 = v76;
      v91 = ".ldgsplitinsert";
    }
  }
  else
  {
    v92 = 256;
  }
  v60 = *(_QWORD *)(a2 + 80);
  v61 = *(const void **)(v8 + 16);
  v62 = *(unsigned int *)(v8 + 24);
  v63 = *a4;
  v64 = *(__int64 (__fastcall **)(__int64, _BYTE *, _BYTE *, __int64, __int64))(*(_QWORD *)v60 + 88LL);
  if ( v64 == sub_9482E0 )
  {
    if ( *(_BYTE *)v63 > 0x15u || *v53 > 0x15u )
      goto LABEL_57;
    v85 = *(unsigned int *)(v8 + 24);
    v65 = sub_AAAE30(*a4, (__int64)v53, *(_DWORD **)(v8 + 16), v85);
    v62 = v85;
    v66 = v65;
  }
  else
  {
    v87 = *(unsigned int *)(v8 + 24);
    v73 = v64(v60, (_BYTE *)v63, v53, (__int64)v61, v62);
    v62 = v87;
    v66 = v73;
  }
  if ( !v66 )
  {
LABEL_57:
    v86 = v62;
    v96 = 257;
    v67 = sub_BD2C40(104, unk_3F148BC);
    v66 = (__int64)v67;
    if ( v67 )
    {
      sub_B44260((__int64)v67, *(_QWORD *)(v63 + 8), 65, 2u, 0, 0);
      *(_QWORD *)(v66 + 72) = v66 + 88;
      *(_QWORD *)(v66 + 80) = 0x400000000LL;
      sub_B4FD20(v66, v63, (__int64)v53, v61, v86, (__int64)&v93);
    }
    (*(void (__fastcall **)(_QWORD, __int64, char **, _QWORD, _QWORD))(**(_QWORD **)(a2 + 88) + 16LL))(
      *(_QWORD *)(a2 + 88),
      v66,
      &v89,
      *(_QWORD *)(a2 + 56),
      *(_QWORD *)(a2 + 64));
    v68 = 4LL * *(unsigned int *)(a2 + 8);
    v69 = *(unsigned int **)a2;
    v70 = &v69[v68];
    while ( v70 != v69 )
    {
      v71 = *((_QWORD *)v69 + 1);
      v72 = *v69;
      v69 += 4;
      sub_B99FD0(v66, v72, v71);
    }
  }
  *a4 = v66;
  return a4;
}
