// Function: sub_FBCEF0
// Address: 0xfbcef0
//
__int64 __fastcall sub_FBCEF0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rdx
  char v9; // di
  int v10; // edi
  __int64 v11; // r8
  int v12; // esi
  unsigned int v13; // ecx
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned int v17; // esi
  unsigned int v18; // eax
  int v19; // ecx
  __int64 v20; // r8
  __int64 v21; // r14
  __int64 v22; // rdx
  unsigned __int64 v23; // rcx
  __int64 v24; // rdx
  unsigned __int64 v25; // rsi
  int v26; // eax
  __int64 v27; // rcx
  char *v28; // rsi
  __int64 v29; // rdi
  char *v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rax
  __int64 v33; // rdx
  _BYTE *v34; // rdi
  int v35; // r11d
  __int64 v36; // r10
  unsigned __int64 v37; // r15
  __int64 v38; // rdi
  __int64 v39; // [rsp+0h] [rbp-D0h] BYREF
  int v40; // [rsp+8h] [rbp-C8h]
  __int64 v41; // [rsp+10h] [rbp-C0h]
  __int64 v42; // [rsp+18h] [rbp-B8h]
  __int64 v43; // [rsp+20h] [rbp-B0h]
  __int64 v44; // [rsp+28h] [rbp-A8h]
  char *v45; // [rsp+30h] [rbp-A0h]
  __int64 v46; // [rsp+38h] [rbp-98h]
  char v47; // [rsp+40h] [rbp-90h] BYREF
  _QWORD v48[2]; // [rsp+50h] [rbp-80h] BYREF
  __int64 v49; // [rsp+60h] [rbp-70h]
  __int64 v50; // [rsp+68h] [rbp-68h]
  __int64 v51; // [rsp+70h] [rbp-60h]
  _BYTE *v52; // [rsp+78h] [rbp-58h]
  __int64 v53; // [rsp+80h] [rbp-50h]
  _BYTE v54[72]; // [rsp+88h] [rbp-48h] BYREF

  v8 = *a2;
  v9 = *(_BYTE *)(a1 + 8);
  v40 = 0;
  v39 = v8;
  v10 = v9 & 1;
  if ( v10 )
  {
    v11 = a1 + 16;
    v12 = 1;
  }
  else
  {
    v17 = *(_DWORD *)(a1 + 24);
    v11 = *(_QWORD *)(a1 + 16);
    if ( !v17 )
    {
      v18 = *(_DWORD *)(a1 + 8);
      ++*(_QWORD *)a1;
      v48[0] = 0;
      v19 = (v18 >> 1) + 1;
LABEL_9:
      v20 = 3 * v17;
      goto LABEL_10;
    }
    v12 = v17 - 1;
  }
  v13 = v12 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v14 = v11 + 16LL * v13;
  a6 = *(_QWORD *)v14;
  if ( v8 == *(_QWORD *)v14 )
  {
LABEL_4:
    v15 = *(unsigned int *)(v14 + 8);
    return *(_QWORD *)(a1 + 48) + 72 * v15 + 8;
  }
  v35 = 1;
  v36 = 0;
  while ( a6 != -4096 )
  {
    if ( !v36 && a6 == -8192 )
      v36 = v14;
    v13 = v12 & (v35 + v13);
    v14 = v11 + 16LL * v13;
    a6 = *(_QWORD *)v14;
    if ( v8 == *(_QWORD *)v14 )
      goto LABEL_4;
    ++v35;
  }
  if ( !v36 )
    v36 = v14;
  v18 = *(_DWORD *)(a1 + 8);
  ++*(_QWORD *)a1;
  v48[0] = v36;
  v19 = (v18 >> 1) + 1;
  if ( !(_BYTE)v10 )
  {
    v17 = *(_DWORD *)(a1 + 24);
    goto LABEL_9;
  }
  v20 = 6;
  v17 = 2;
LABEL_10:
  if ( 4 * v19 >= (unsigned int)v20 )
  {
    sub_FBCAD0(a1, 2 * v17);
LABEL_30:
    sub_FA1D90(a1, &v39, v48);
    v8 = v39;
    v18 = *(_DWORD *)(a1 + 8);
    goto LABEL_12;
  }
  if ( v17 - *(_DWORD *)(a1 + 12) - v19 <= v17 >> 3 )
  {
    sub_FBCAD0(a1, v17);
    goto LABEL_30;
  }
LABEL_12:
  v21 = v48[0];
  *(_DWORD *)(a1 + 8) = (2 * (v18 >> 1) + 2) | v18 & 1;
  if ( *(_QWORD *)v21 != -4096 )
    --*(_DWORD *)(a1 + 12);
  *(_QWORD *)v21 = v8;
  v42 = 0;
  *(_DWORD *)(v21 + 8) = v40;
  v22 = *a2;
  v23 = *(unsigned int *)(a1 + 60);
  v52 = v54;
  v48[0] = v22;
  v24 = *(unsigned int *)(a1 + 56);
  v45 = &v47;
  v25 = v24 + 1;
  v53 = 0x200000000LL;
  v46 = 0x200000000LL;
  v26 = v24;
  v43 = 0;
  v44 = 0;
  v41 = 1;
  v48[1] = 1;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  if ( v24 + 1 > v23 )
  {
    v37 = *(_QWORD *)(a1 + 48);
    v38 = a1 + 48;
    if ( v37 > (unsigned __int64)v48 || (unsigned __int64)v48 >= v37 + 72 * v24 )
    {
      sub_FAA260(v38, v25, v24, v23, v20, a6);
      v24 = *(unsigned int *)(a1 + 56);
      v27 = *(_QWORD *)(a1 + 48);
      v28 = (char *)v48;
      v26 = *(_DWORD *)(a1 + 56);
    }
    else
    {
      sub_FAA260(v38, v25, v24, v23, v20, a6);
      v27 = *(_QWORD *)(a1 + 48);
      v24 = *(unsigned int *)(a1 + 56);
      v28 = (char *)v48 + v27 - v37;
      v26 = *(_DWORD *)(a1 + 56);
    }
  }
  else
  {
    v27 = *(_QWORD *)(a1 + 48);
    v28 = (char *)v48;
  }
  v29 = v27 + 72 * v24;
  if ( v29 )
  {
    v30 = *(char **)v28;
    *(_QWORD *)(v29 + 24) = 0;
    *(_QWORD *)(v29 + 16) = 0;
    *(_DWORD *)(v29 + 32) = 0;
    *(_QWORD *)v29 = v30;
    *(_QWORD *)(v29 + 8) = 1;
    v31 = *((_QWORD *)v28 + 2);
    ++*((_QWORD *)v28 + 1);
    v32 = *(_QWORD *)(v29 + 16);
    *(_QWORD *)(v29 + 16) = v31;
    LODWORD(v31) = *((_DWORD *)v28 + 6);
    *((_QWORD *)v28 + 2) = v32;
    LODWORD(v32) = *(_DWORD *)(v29 + 24);
    *(_DWORD *)(v29 + 24) = v31;
    LODWORD(v31) = *((_DWORD *)v28 + 7);
    *((_DWORD *)v28 + 6) = v32;
    LODWORD(v32) = *(_DWORD *)(v29 + 28);
    *(_DWORD *)(v29 + 28) = v31;
    v33 = *((unsigned int *)v28 + 8);
    *((_DWORD *)v28 + 7) = v32;
    LODWORD(v32) = *(_DWORD *)(v29 + 32);
    *(_DWORD *)(v29 + 32) = v33;
    *((_DWORD *)v28 + 8) = v32;
    *(_QWORD *)(v29 + 40) = v29 + 56;
    *(_QWORD *)(v29 + 48) = 0x200000000LL;
    if ( *((_DWORD *)v28 + 12) )
    {
      v28 += 40;
      sub_F8F130(v29 + 40, (char **)v28, v33, v27, v20, a6);
    }
    v26 = *(_DWORD *)(a1 + 56);
  }
  v34 = v52;
  *(_DWORD *)(a1 + 56) = v26 + 1;
  if ( v34 != v54 )
    _libc_free(v34, v28);
  sub_C7D6A0(v49, 8LL * (unsigned int)v51, 8);
  sub_C7D6A0(0, 0, 8);
  v15 = (unsigned int)(*(_DWORD *)(a1 + 56) - 1);
  *(_DWORD *)(v21 + 8) = v15;
  return *(_QWORD *)(a1 + 48) + 72 * v15 + 8;
}
