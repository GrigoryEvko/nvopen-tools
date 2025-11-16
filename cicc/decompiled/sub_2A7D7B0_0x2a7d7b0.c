// Function: sub_2A7D7B0
// Address: 0x2a7d7b0
//
void __fastcall sub_2A7D7B0(__int64 a1, _BYTE *a2, char a3)
{
  __int64 v3; // r14
  int v5; // eax
  int v6; // r13d
  __int64 v7; // r12
  __int64 *v8; // rax
  __int64 v9; // r12
  __int64 v10; // rsi
  __int64 v11; // r15
  __int64 v12; // r13
  unsigned __int64 v13; // rax
  char v14; // al
  _QWORD *v15; // rbx
  unsigned __int64 v16; // rdi
  unsigned __int64 v17; // rdi
  unsigned int v18; // eax
  unsigned int v19; // esi
  int v20; // eax
  int v21; // eax
  __int64 v22; // rax
  unsigned __int64 v23; // r13
  __int64 v24; // r13
  __int64 v25; // rax
  unsigned int v26; // eax
  unsigned int v27; // eax
  _QWORD *v28; // [rsp-110h] [rbp-110h] BYREF
  unsigned __int64 v29; // [rsp-108h] [rbp-108h] BYREF
  unsigned int v30; // [rsp-100h] [rbp-100h]
  unsigned __int64 v31; // [rsp-F8h] [rbp-F8h] BYREF
  unsigned int v32; // [rsp-F0h] [rbp-F0h]
  unsigned __int64 v33; // [rsp-E8h] [rbp-E8h] BYREF
  unsigned int v34; // [rsp-E0h] [rbp-E0h]
  unsigned __int64 v35; // [rsp-D8h] [rbp-D8h]
  unsigned int v36; // [rsp-D0h] [rbp-D0h]
  unsigned __int64 v37; // [rsp-C8h] [rbp-C8h] BYREF
  unsigned int v38; // [rsp-C0h] [rbp-C0h]
  unsigned __int64 v39; // [rsp-B8h] [rbp-B8h] BYREF
  unsigned int v40; // [rsp-B0h] [rbp-B0h]
  unsigned __int64 v41; // [rsp-A8h] [rbp-A8h] BYREF
  unsigned int v42; // [rsp-A0h] [rbp-A0h]
  unsigned __int64 v43; // [rsp-98h] [rbp-98h] BYREF
  unsigned int v44; // [rsp-90h] [rbp-90h]
  _QWORD *v45; // [rsp-88h] [rbp-88h] BYREF
  unsigned int v46; // [rsp-80h] [rbp-80h]
  __int64 v47; // [rsp-78h] [rbp-78h]
  int v48; // [rsp-70h] [rbp-70h]
  unsigned __int64 v49; // [rsp-68h] [rbp-68h] BYREF
  __int64 v50; // [rsp-60h] [rbp-60h]
  unsigned __int64 v51; // [rsp-58h] [rbp-58h]
  _QWORD v52[2]; // [rsp-50h] [rbp-50h] BYREF
  __int64 v53; // [rsp-40h] [rbp-40h]

  if ( *a2 != 82 )
    return;
  if ( **(_QWORD **)a1 != *((_QWORD *)a2 - 8) )
    return;
  v3 = *((_QWORD *)a2 - 4);
  if ( !v3 )
    return;
  v5 = sub_B53900((__int64)a2);
  v6 = v5;
  if ( !a3 )
    v6 = sub_B52870(v5);
  v7 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL);
  v8 = sub_DD8400(v7, v3);
  v9 = sub_DBB9F0(v7, (__int64)v8, 1u, 0);
  v30 = *(_DWORD *)(v9 + 8);
  if ( v30 > 0x40 )
    sub_C43780((__int64)&v29, (const void **)v9);
  else
    v29 = *(_QWORD *)v9;
  v32 = *(_DWORD *)(v9 + 24);
  if ( v32 > 0x40 )
    sub_C43780((__int64)&v31, (const void **)(v9 + 16));
  else
    v31 = *(_QWORD *)(v9 + 16);
  sub_AB15A0((__int64)&v33, v6, (__int64)&v29);
  v10 = **(_QWORD **)(a1 + 16);
  v46 = *(_DWORD *)(v10 + 8);
  if ( v46 > 0x40 )
    sub_C43780((__int64)&v45, (const void **)v10);
  else
    v45 = *(_QWORD **)v10;
  sub_AADBC0((__int64)&v49, (__int64 *)&v45);
  sub_ABA0E0((__int64)&v37, (__int64)&v33, (__int64)&v49, 2, 0);
  if ( LODWORD(v52[0]) > 0x40 && v51 )
    j_j___libc_free_0_0(v51);
  if ( (unsigned int)v50 > 0x40 && v49 )
    j_j___libc_free_0_0(v49);
  if ( v46 > 0x40 && v45 )
    j_j___libc_free_0_0((unsigned __int64)v45);
  v11 = *(_QWORD *)(a1 + 8);
  v42 = v38;
  if ( v38 > 0x40 )
    sub_C43780((__int64)&v41, (const void **)&v37);
  else
    v41 = v37;
  v44 = v40;
  if ( v40 > 0x40 )
    sub_C43780((__int64)&v43, (const void **)&v39);
  else
    v43 = v39;
  v12 = **(_QWORD **)(a1 + 32);
  v13 = **(_QWORD **)(a1 + 24);
  v49 = 0;
  v50 = 0;
  v51 = v13;
  if ( v13 != 0 && v13 != -4096 && v13 != -8192 )
    sub_BD73F0((__int64)&v49);
  v52[0] = 0;
  v52[1] = 0;
  v53 = v12;
  if ( v12 != -4096 && v12 != 0 && v12 != -8192 )
    sub_BD73F0((__int64)v52);
  v14 = sub_2A75400(v11 + 288, (__int64)&v49, &v28);
  v15 = v28;
  if ( !v14 )
  {
    v45 = v28;
    v19 = *(_DWORD *)(v11 + 312);
    v20 = *(_DWORD *)(v11 + 304);
    ++*(_QWORD *)(v11 + 288);
    v21 = v20 + 1;
    if ( 4 * v21 >= 3 * v19 )
    {
      v19 *= 2;
    }
    else if ( v19 - *(_DWORD *)(v11 + 308) - v21 > v19 >> 3 )
    {
LABEL_74:
      *(_DWORD *)(v11 + 304) = v21;
      if ( v15[2] == -4096 && v15[5] == -4096 )
      {
        v23 = v51;
        if ( v51 == -4096 )
          goto LABEL_82;
      }
      else
      {
        --*(_DWORD *)(v11 + 308);
        v22 = v15[2];
        v23 = v51;
        if ( v51 == v22 )
        {
LABEL_82:
          v24 = v53;
          v25 = v15[5];
          if ( v53 != v25 )
          {
            if ( v25 != 0 && v25 != -4096 && v25 != -8192 )
              sub_BD60C0(v15 + 3);
            v15[5] = v24;
            if ( v24 != 0 && v24 != -4096 && v24 != -8192 )
              sub_BD73F0((__int64)(v15 + 3));
          }
          v26 = v42;
          *((_DWORD *)v15 + 14) = v42;
          if ( v26 > 0x40 )
            sub_C43780((__int64)(v15 + 6), (const void **)&v41);
          else
            v15[6] = v41;
          v27 = v44;
          *((_DWORD *)v15 + 18) = v44;
          if ( v27 > 0x40 )
            sub_C43780((__int64)(v15 + 8), (const void **)&v43);
          else
            v15[8] = v43;
          goto LABEL_42;
        }
        if ( v22 != 0 && v22 != -4096 && v22 != -8192 )
          sub_BD60C0(v15);
      }
      v15[2] = v23;
      if ( v23 != 0 && v23 != -4096 && v23 != -8192 )
        sub_BD73F0((__int64)v15);
      goto LABEL_82;
    }
    sub_2A7D1F0(v11 + 288, v19);
    sub_2A75400(v11 + 288, (__int64)&v49, &v45);
    v15 = v45;
    v21 = *(_DWORD *)(v11 + 304) + 1;
    goto LABEL_74;
  }
  sub_AB2160((__int64)&v45, (__int64)&v41, (__int64)(v28 + 6), 0);
  if ( *((_DWORD *)v15 + 14) > 0x40u )
  {
    v16 = v15[6];
    if ( v16 )
      j_j___libc_free_0_0(v16);
  }
  v15[6] = v45;
  *((_DWORD *)v15 + 14) = v46;
  v46 = 0;
  if ( *((_DWORD *)v15 + 18) > 0x40u && (v17 = v15[8]) != 0 )
  {
    j_j___libc_free_0_0(v17);
    v18 = v46;
    v15[8] = v47;
    *((_DWORD *)v15 + 18) = v48;
    if ( v18 > 0x40 && v45 )
      j_j___libc_free_0_0((unsigned __int64)v45);
  }
  else
  {
    v15[8] = v47;
    *((_DWORD *)v15 + 18) = v48;
  }
LABEL_42:
  if ( v53 != 0 && v53 != -4096 && v53 != -8192 )
    sub_BD60C0(v52);
  if ( v51 != -4096 && v51 != 0 && v51 != -8192 )
    sub_BD60C0(&v49);
  if ( v44 > 0x40 && v43 )
    j_j___libc_free_0_0(v43);
  if ( v42 > 0x40 && v41 )
    j_j___libc_free_0_0(v41);
  if ( v40 > 0x40 && v39 )
    j_j___libc_free_0_0(v39);
  if ( v38 > 0x40 && v37 )
    j_j___libc_free_0_0(v37);
  if ( v36 > 0x40 && v35 )
    j_j___libc_free_0_0(v35);
  if ( v34 > 0x40 && v33 )
    j_j___libc_free_0_0(v33);
  if ( v32 > 0x40 && v31 )
    j_j___libc_free_0_0(v31);
  if ( v30 > 0x40 )
  {
    if ( v29 )
      j_j___libc_free_0_0(v29);
  }
}
