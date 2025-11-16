// Function: sub_31AD3D0
// Address: 0x31ad3d0
//
__int64 __fastcall sub_31AD3D0(__int64 a1, __int64 *a2)
{
  __int64 v4; // r14
  unsigned int v5; // esi
  __int64 v6; // rcx
  int v7; // r10d
  __int64 *v8; // r12
  unsigned int v9; // edx
  __int64 *v10; // rax
  __int64 v11; // r8
  __int64 v12; // rax
  int v14; // eax
  int v15; // edx
  __int64 v16; // rax
  char *v17; // r14
  __int64 v18; // r8
  __int64 v19; // r9
  unsigned __int64 v20; // rcx
  __int64 v21; // rax
  unsigned __int64 v22; // rsi
  int v23; // edx
  __int64 v24; // r13
  __int64 v25; // r13
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdx
  int v29; // eax
  bool v30; // zf
  int v31; // eax
  int v32; // ecx
  __int64 v33; // rdi
  unsigned int v34; // eax
  __int64 v35; // rsi
  int v36; // r9d
  __int64 *v37; // r8
  unsigned __int64 v38; // r15
  __int64 v39; // rdi
  int v40; // eax
  int v41; // eax
  __int64 v42; // rsi
  int v43; // r8d
  unsigned int v44; // r15d
  __int64 *v45; // rdi
  __int64 v46; // rcx
  _QWORD v47[22]; // [rsp+0h] [rbp-1A0h] BYREF
  _QWORD v48[2]; // [rsp+B0h] [rbp-F0h] BYREF
  _QWORD v49[2]; // [rsp+C0h] [rbp-E0h] BYREF
  __int64 v50; // [rsp+D0h] [rbp-D0h]
  __int64 v51; // [rsp+D8h] [rbp-C8h]
  int v52; // [rsp+E0h] [rbp-C0h]
  int v53; // [rsp+E4h] [rbp-BCh]
  __int64 v54; // [rsp+E8h] [rbp-B8h]
  __int64 v55; // [rsp+F0h] [rbp-B0h]
  __int16 v56; // [rsp+F8h] [rbp-A8h]
  _BYTE v57[8]; // [rsp+100h] [rbp-A0h] BYREF
  unsigned __int64 v58; // [rsp+108h] [rbp-98h]
  char v59; // [rsp+11Ch] [rbp-84h]
  _BYTE v60[64]; // [rsp+120h] [rbp-80h] BYREF
  int v61; // [rsp+160h] [rbp-40h]

  v4 = *a2;
  v5 = *(_DWORD *)(a1 + 24);
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_36;
  }
  v6 = *(_QWORD *)(a1 + 8);
  v7 = 1;
  v8 = 0;
  v9 = (v5 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v10 = (__int64 *)(v6 + 16LL * v9);
  v11 = *v10;
  if ( v4 == *v10 )
  {
LABEL_3:
    v12 = *((unsigned int *)v10 + 2);
    return *(_QWORD *)(a1 + 32) + 184 * v12 + 8;
  }
  while ( v11 != -4096 )
  {
    if ( !v8 && v11 == -8192 )
      v8 = v10;
    v9 = (v5 - 1) & (v7 + v9);
    v10 = (__int64 *)(v6 + 16LL * v9);
    v11 = *v10;
    if ( v4 == *v10 )
      goto LABEL_3;
    ++v7;
  }
  if ( !v8 )
    v8 = v10;
  v14 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v15 = v14 + 1;
  if ( 4 * (v14 + 1) >= 3 * v5 )
  {
LABEL_36:
    sub_29906F0(a1, 2 * v5);
    v31 = *(_DWORD *)(a1 + 24);
    if ( v31 )
    {
      v32 = v31 - 1;
      v33 = *(_QWORD *)(a1 + 8);
      v34 = (v31 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v15 = *(_DWORD *)(a1 + 16) + 1;
      v8 = (__int64 *)(v33 + 16LL * v34);
      v35 = *v8;
      if ( v4 != *v8 )
      {
        v36 = 1;
        v37 = 0;
        while ( v35 != -4096 )
        {
          if ( !v37 && v35 == -8192 )
            v37 = v8;
          v34 = v32 & (v36 + v34);
          v8 = (__int64 *)(v33 + 16LL * v34);
          v35 = *v8;
          if ( v4 == *v8 )
            goto LABEL_15;
          ++v36;
        }
        if ( v37 )
          v8 = v37;
      }
      goto LABEL_15;
    }
    goto LABEL_63;
  }
  if ( v5 - *(_DWORD *)(a1 + 20) - v15 <= v5 >> 3 )
  {
    sub_29906F0(a1, v5);
    v40 = *(_DWORD *)(a1 + 24);
    if ( v40 )
    {
      v41 = v40 - 1;
      v42 = *(_QWORD *)(a1 + 8);
      v43 = 1;
      v44 = v41 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
      v15 = *(_DWORD *)(a1 + 16) + 1;
      v45 = 0;
      v8 = (__int64 *)(v42 + 16LL * v44);
      v46 = *v8;
      if ( v4 != *v8 )
      {
        while ( v46 != -4096 )
        {
          if ( !v45 && v46 == -8192 )
            v45 = v8;
          v44 = v41 & (v43 + v44);
          v8 = (__int64 *)(v42 + 16LL * v44);
          v46 = *v8;
          if ( v4 == *v8 )
            goto LABEL_15;
          ++v43;
        }
        if ( v45 )
          v8 = v45;
      }
      goto LABEL_15;
    }
LABEL_63:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v15;
  if ( *v8 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *v8 = v4;
  *((_DWORD *)v8 + 2) = 0;
  memset(v47, 0, sizeof(v47));
  v16 = *a2;
  v47[10] = &v47[13];
  v17 = (char *)v48;
  v48[0] = v16;
  v56 = 0;
  v47[1] = 6;
  v47[11] = 8;
  BYTE4(v47[12]) = 1;
  v48[1] = 0;
  v49[0] = 6;
  v49[1] = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  sub_C8CF70((__int64)v57, v60, 8, (__int64)&v47[13], (__int64)&v47[9]);
  v20 = *(unsigned int *)(a1 + 44);
  v61 = v47[21];
  v21 = *(unsigned int *)(a1 + 40);
  v22 = v21 + 1;
  v23 = v21;
  if ( v21 + 1 > v20 )
  {
    v38 = *(_QWORD *)(a1 + 32);
    v39 = a1 + 32;
    if ( v38 > (unsigned __int64)v48 || (unsigned __int64)v48 >= v38 + 184 * v21 )
    {
      sub_31AA620(v39, v22, v21, v20, v18, v19);
      v21 = *(unsigned int *)(a1 + 40);
      v24 = *(_QWORD *)(a1 + 32);
      v23 = *(_DWORD *)(a1 + 40);
    }
    else
    {
      sub_31AA620(v39, v22, v21, v20, v18, v19);
      v24 = *(_QWORD *)(a1 + 32);
      v21 = *(unsigned int *)(a1 + 40);
      v17 = (char *)v48 + v24 - v38;
      v23 = *(_DWORD *)(a1 + 40);
    }
  }
  else
  {
    v24 = *(_QWORD *)(a1 + 32);
  }
  v25 = 184 * v21 + v24;
  if ( v25 )
  {
    *(_QWORD *)v25 = *(_QWORD *)v17;
    v26 = *((_QWORD *)v17 + 1);
    *(_QWORD *)(v25 + 16) = 6;
    *(_QWORD *)(v25 + 8) = v26;
    v27 = *((_QWORD *)v17 + 4);
    *(_QWORD *)(v25 + 24) = 0;
    *(_QWORD *)(v25 + 32) = v27;
    if ( v27 != 0 && v27 != -4096 && v27 != -8192 )
      sub_BD6050((unsigned __int64 *)(v25 + 16), *((_QWORD *)v17 + 2) & 0xFFFFFFFFFFFFFFF8LL);
    v28 = *((_QWORD *)v17 + 7);
    *(_QWORD *)(v25 + 40) = *((_QWORD *)v17 + 5);
    v29 = *((_DWORD *)v17 + 12);
    *(_QWORD *)(v25 + 56) = v28;
    *(_DWORD *)(v25 + 48) = v29;
    *(_DWORD *)(v25 + 52) = *((_DWORD *)v17 + 13);
    *(_QWORD *)(v25 + 64) = *((_QWORD *)v17 + 8);
    *(_WORD *)(v25 + 72) = *((_WORD *)v17 + 36);
    sub_C8CF70(v25 + 80, (void *)(v25 + 112), 8, (__int64)(v17 + 112), (__int64)(v17 + 80));
    *(_DWORD *)(v25 + 176) = *((_DWORD *)v17 + 44);
    v23 = *(_DWORD *)(a1 + 40);
  }
  v30 = v59 == 0;
  *(_DWORD *)(a1 + 40) = v23 + 1;
  if ( v30 )
    _libc_free(v58);
  if ( v50 != 0 && v50 != -4096 && v50 != -8192 )
    sub_BD60C0(v49);
  if ( !BYTE4(v47[12]) )
    _libc_free(v47[10]);
  if ( v47[3] != -4096 && v47[3] != 0 && v47[3] != -8192 )
    sub_BD60C0(&v47[1]);
  v12 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
  *((_DWORD *)v8 + 2) = v12;
  return *(_QWORD *)(a1 + 32) + 184 * v12 + 8;
}
