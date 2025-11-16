// Function: sub_F6B230
// Address: 0xf6b230
//
__int64 __fastcall sub_F6B230(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  __int64 v7; // r15
  __int64 v8; // rax
  __int64 v9; // rcx
  __int64 v10; // r8
  int v11; // r11d
  unsigned int i; // eax
  __int64 v13; // rsi
  unsigned int v14; // eax
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 v18; // rsi
  int v19; // r10d
  unsigned int j; // eax
  __int64 v21; // r8
  unsigned int v22; // eax
  __int64 v23; // r14
  __int64 v24; // r13
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // r13
  int v28; // ebx
  __int64 v29; // rax
  __int64 v30; // r13
  __int64 *v31; // r12
  __int64 v32; // r14
  __int64 v33; // r15
  __int64 v34; // rcx
  __int64 v35; // r8
  __int64 v36; // r9
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // r8
  __int64 v48; // r9
  _QWORD *v49; // rbx
  _QWORD *v50; // r12
  __int64 v51; // rax
  _QWORD *v52; // rbx
  _QWORD *v53; // r12
  __int64 v54; // rax
  __int64 v56; // [rsp-8h] [rbp-C8h]
  __int64 v57; // [rsp+8h] [rbp-B8h]
  __int64 v58; // [rsp+18h] [rbp-A8h]
  __int64 v59; // [rsp+20h] [rbp-A0h]
  __int64 v60; // [rsp+28h] [rbp-98h]
  __int64 v61; // [rsp+30h] [rbp-90h] BYREF
  _BYTE *v62; // [rsp+38h] [rbp-88h]
  __int64 v63; // [rsp+40h] [rbp-80h]
  int v64; // [rsp+48h] [rbp-78h]
  char v65; // [rsp+4Ch] [rbp-74h]
  _BYTE v66[16]; // [rsp+50h] [rbp-70h] BYREF
  __int64 v67; // [rsp+60h] [rbp-60h] BYREF
  _BYTE *v68; // [rsp+68h] [rbp-58h]
  __int64 v69; // [rsp+70h] [rbp-50h]
  int v70; // [rsp+78h] [rbp-48h]
  char v71; // [rsp+7Ch] [rbp-44h]
  _BYTE v72[64]; // [rsp+80h] [rbp-40h] BYREF

  v6 = sub_BC1CD0(a4, &unk_4F875F0, a3);
  v7 = v6 + 8;
  v8 = sub_BC1CD0(a4, &unk_4F81450, a3);
  v9 = *(unsigned int *)(a4 + 88);
  v10 = *(_QWORD *)(a4 + 72);
  v59 = v8 + 8;
  if ( !(_DWORD)v9 )
    goto LABEL_53;
  v11 = 1;
  for ( i = (v9 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F881D0 >> 9) ^ ((unsigned int)&unk_4F881D0 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; i = (v9 - 1) & v14 )
  {
    v13 = v10 + 24LL * i;
    if ( *(_UNKNOWN **)v13 == &unk_4F881D0 && a3 == *(_QWORD *)(v13 + 8) )
      break;
    if ( *(_QWORD *)v13 == -4096 && *(_QWORD *)(v13 + 8) == -4096 )
      goto LABEL_53;
    v14 = v11 + i;
    ++v11;
  }
  if ( v13 == v10 + 24 * v9 )
  {
LABEL_53:
    v15 = 0;
  }
  else
  {
    v15 = *(_QWORD *)(*(_QWORD *)(v13 + 16) + 24LL);
    if ( v15 )
      v15 += 8;
  }
  v16 = sub_BC1CD0(a4, &unk_4F86630, a3);
  v17 = *(unsigned int *)(a4 + 88);
  v18 = *(_QWORD *)(a4 + 72);
  v60 = v16 + 8;
  if ( !(_DWORD)v17 )
    goto LABEL_55;
  v19 = 1;
  for ( j = (v17 - 1)
          & (((0xBF58476D1CE4E5B9LL
             * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)
              | ((unsigned __int64)(((unsigned int)&unk_4F8F810 >> 9) ^ ((unsigned int)&unk_4F8F810 >> 4)) << 32))) >> 31)
           ^ (484763065 * (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4)))); ; j = (v17 - 1) & v22 )
  {
    v21 = v18 + 24LL * j;
    if ( *(_UNKNOWN **)v21 == &unk_4F8F810 && a3 == *(_QWORD *)(v21 + 8) )
      break;
    if ( *(_QWORD *)v21 == -4096 && *(_QWORD *)(v21 + 8) == -4096 )
      goto LABEL_55;
    v22 = v19 + j;
    ++v19;
  }
  if ( v21 == v18 + 24 * v17 )
  {
LABEL_55:
    v57 = 0;
    v23 = 0;
  }
  else
  {
    v23 = *(_QWORD *)(*(_QWORD *)(v21 + 16) + 24LL);
    if ( v23 )
    {
      v24 = *(_QWORD *)(v23 + 8);
      v57 = v23 + 8;
      v25 = sub_22077B0(760);
      v23 = v25;
      if ( v25 )
      {
        *(_QWORD *)v25 = v24;
        *(_QWORD *)(v25 + 8) = v25 + 24;
        *(_QWORD *)(v25 + 16) = 0x1000000000LL;
        *(_QWORD *)(v25 + 416) = v25 + 440;
        *(_QWORD *)(v25 + 504) = v25 + 520;
        *(_QWORD *)(v25 + 512) = 0x800000000LL;
        *(_QWORD *)(v25 + 408) = 0;
        *(_QWORD *)(v25 + 424) = 8;
        *(_DWORD *)(v25 + 432) = 0;
        *(_BYTE *)(v25 + 436) = 1;
        *(_DWORD *)(v25 + 720) = 0;
        *(_QWORD *)(v25 + 728) = 0;
        *(_QWORD *)(v25 + 736) = v25 + 720;
        *(_QWORD *)(v25 + 744) = v25 + 720;
        *(_QWORD *)(v25 + 752) = 0;
      }
    }
    else
    {
      v57 = 0;
    }
  }
  v26 = *(_QWORD *)(v6 + 48);
  v27 = *(_QWORD *)(v6 + 40);
  v28 = 0;
  v58 = v26;
  if ( v26 == v27 )
    goto LABEL_51;
  v29 = v27;
  v30 = v15;
  v31 = (__int64 *)v23;
  v32 = v7;
  v33 = v29;
  do
  {
    v33 += 8;
    v18 = v59;
    v28 |= sub_F6AC10(*(char **)(v33 - 8), v59, v32, v30, v60, v31, 0);
  }
  while ( v58 != v33 );
  v23 = (__int64)v31;
  if ( !(_BYTE)v28 )
  {
LABEL_51:
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &unk_4F82400;
    *(_QWORD *)a1 = 1;
  }
  else
  {
    v61 = 0;
    v62 = v66;
    v63 = 2;
    v64 = 0;
    v65 = 1;
    v67 = 0;
    v68 = v72;
    v69 = 2;
    v70 = 0;
    v71 = 1;
    sub_F67B40((__int64)&v61, (__int64)&unk_4F81450, v56, v34, v35, v36);
    sub_F67B40((__int64)&v61, (__int64)&unk_4F875F0, v37, v38, v39, v40);
    sub_F67B40((__int64)&v61, (__int64)&unk_4F881D0, v41, v42, v43, v44);
    if ( v57 )
      sub_F67B40((__int64)&v61, (__int64)&unk_4F8F810, v45, v46, v47, v48);
    sub_F67B40((__int64)&v61, (__int64)&unk_4F8E5A8, v45, v46, v47, v48);
    sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v66, (__int64)&v61);
    v18 = a1 + 80;
    sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v72, (__int64)&v67);
    if ( !v71 )
      _libc_free(v68, v18);
    if ( !v65 )
      _libc_free(v62, v18);
  }
  if ( v23 )
  {
    sub_F67080(*(_QWORD **)(v23 + 728));
    v49 = *(_QWORD **)(v23 + 504);
    v50 = &v49[3 * *(unsigned int *)(v23 + 512)];
    if ( v49 != v50 )
    {
      do
      {
        v51 = *(v50 - 1);
        v50 -= 3;
        if ( v51 != 0 && v51 != -4096 && v51 != -8192 )
          sub_BD60C0(v50);
      }
      while ( v49 != v50 );
      v50 = *(_QWORD **)(v23 + 504);
    }
    if ( v50 != (_QWORD *)(v23 + 520) )
      _libc_free(v50, v18);
    if ( !*(_BYTE *)(v23 + 436) )
      _libc_free(*(_QWORD *)(v23 + 416), v18);
    v52 = *(_QWORD **)(v23 + 8);
    v53 = &v52[3 * *(unsigned int *)(v23 + 16)];
    if ( v52 != v53 )
    {
      do
      {
        v54 = *(v53 - 1);
        v53 -= 3;
        if ( v54 != 0 && v54 != -4096 && v54 != -8192 )
          sub_BD60C0(v53);
      }
      while ( v52 != v53 );
      v53 = *(_QWORD **)(v23 + 8);
    }
    if ( v53 != (_QWORD *)(v23 + 24) )
      _libc_free(v53, v18);
    j_j___libc_free_0(v23, 760);
  }
  return a1;
}
