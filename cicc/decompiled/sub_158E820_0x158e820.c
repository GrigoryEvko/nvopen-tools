// Function: sub_158E820
// Address: 0x158e820
//
__int64 __fastcall sub_158E820(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 *v7; // r15
  unsigned int v8; // eax
  unsigned int v9; // edx
  __int64 v10; // rsi
  unsigned int v11; // ecx
  __int64 v12; // rax
  unsigned int v13; // eax
  int v14; // eax
  __int64 v15; // r13
  __int64 *v16; // r12
  __int64 *v17; // rbx
  __int64 v18; // rbx
  __int64 *v19; // r13
  unsigned int v20; // eax
  __int64 *v21; // r13
  __int64 *v22; // rbx
  unsigned int v23; // eax
  unsigned int v24; // eax
  __int64 *v25; // rbx
  __int64 v26; // [rsp+0h] [rbp-190h]
  unsigned int v27; // [rsp+8h] [rbp-188h]
  unsigned int v28; // [rsp+Ch] [rbp-184h]
  __int64 v29; // [rsp+38h] [rbp-158h]
  __int64 v30; // [rsp+40h] [rbp-150h] BYREF
  unsigned int v31; // [rsp+48h] [rbp-148h]
  __int64 v32; // [rsp+50h] [rbp-140h] BYREF
  unsigned int v33; // [rsp+58h] [rbp-138h]
  __int64 v34; // [rsp+60h] [rbp-130h] BYREF
  unsigned int v35; // [rsp+68h] [rbp-128h]
  __int64 v36; // [rsp+70h] [rbp-120h] BYREF
  unsigned int v37; // [rsp+78h] [rbp-118h]
  __int64 v38; // [rsp+80h] [rbp-110h] BYREF
  unsigned int v39; // [rsp+88h] [rbp-108h]
  __int64 v40; // [rsp+90h] [rbp-100h] BYREF
  unsigned int v41; // [rsp+98h] [rbp-F8h]
  __int64 v42; // [rsp+A0h] [rbp-F0h] BYREF
  unsigned int v43; // [rsp+A8h] [rbp-E8h]
  __int64 v44; // [rsp+B0h] [rbp-E0h]
  unsigned int v45; // [rsp+B8h] [rbp-D8h]
  __int64 v46; // [rsp+C0h] [rbp-D0h] BYREF
  unsigned int v47; // [rsp+C8h] [rbp-C8h]
  __int64 v48; // [rsp+D0h] [rbp-C0h] BYREF
  unsigned int v49; // [rsp+D8h] [rbp-B8h]
  __int64 v50; // [rsp+E0h] [rbp-B0h] BYREF
  unsigned int v51; // [rsp+E8h] [rbp-A8h]
  __int64 v52; // [rsp+F0h] [rbp-A0h]
  unsigned int v53; // [rsp+F8h] [rbp-98h]
  __int64 v54; // [rsp+100h] [rbp-90h] BYREF
  unsigned int v55; // [rsp+108h] [rbp-88h]
  __int64 v56; // [rsp+110h] [rbp-80h]
  unsigned int v57; // [rsp+118h] [rbp-78h]
  __int64 v58; // [rsp+120h] [rbp-70h] BYREF
  unsigned int v59; // [rsp+128h] [rbp-68h]
  _BYTE v60[16]; // [rsp+130h] [rbp-60h] BYREF
  char v61[16]; // [rsp+140h] [rbp-50h] BYREF
  char v62[16]; // [rsp+150h] [rbp-40h] BYREF
  _BYTE v63[48]; // [rsp+160h] [rbp-30h] BYREF

  v4 = a1;
  if ( sub_158A120(a2) || sub_158A120(a3) )
  {
    sub_15897D0(a1, *(_DWORD *)(a2 + 8), 0);
    return v4;
  }
  sub_158AAD0((__int64)&v58, a2);
  sub_16A5C50(&v30, &v58, (unsigned int)(2 * *(_DWORD *)(a2 + 8)));
  if ( v59 > 0x40 && v58 )
    j_j___libc_free_0_0(v58);
  sub_158A9F0((__int64)&v58, a2);
  sub_16A5C50(&v32, &v58, (unsigned int)(2 * *(_DWORD *)(a2 + 8)));
  if ( v59 > 0x40 && v58 )
    j_j___libc_free_0_0(v58);
  sub_158AAD0((__int64)&v58, a3);
  sub_16A5C50(&v34, &v58, (unsigned int)(2 * *(_DWORD *)(a2 + 8)));
  if ( v59 > 0x40 && v58 )
    j_j___libc_free_0_0(v58);
  sub_158A9F0((__int64)&v58, a3);
  sub_16A5C50(&v36, &v58, (unsigned int)(2 * *(_DWORD *)(a2 + 8)));
  if ( v59 > 0x40 && v58 )
    j_j___libc_free_0_0(v58);
  v7 = &v54;
  sub_16A7B50(&v54, &v32, &v36);
  sub_16A7490(&v54, 1);
  v8 = v55;
  v55 = 0;
  v59 = v8;
  v58 = v54;
  sub_16A7B50(&v50, &v30, &v34);
  sub_15898E0((__int64)&v42, (__int64)&v50, &v58);
  if ( v51 > 0x40 && v50 )
    j_j___libc_free_0_0(v50);
  if ( v59 > 0x40 && v58 )
    j_j___libc_free_0_0(v58);
  if ( v55 > 0x40 && v54 )
    j_j___libc_free_0_0(v54);
  sub_158D430((__int64)&v46, (__int64)&v42, *(_DWORD *)(a2 + 8));
  if ( sub_158A670((__int64)&v46) )
    goto LABEL_50;
  v9 = v49;
  v10 = v48;
  v11 = v49 - 1;
  v12 = 1LL << ((unsigned __int8)v49 - 1);
  if ( v49 > 0x40 )
  {
    if ( (*(_QWORD *)(v48 + 8LL * (v11 >> 6)) & v12) != 0 )
    {
      v26 = v48;
      v27 = v49 - 1;
      v28 = v49;
      v14 = sub_16A58A0(&v48);
      v9 = v28;
      v10 = v26;
      if ( v27 != v14 )
        goto LABEL_50;
    }
  }
  else if ( (v12 & v48) != 0 && 1LL << v11 != v48 )
  {
LABEL_50:
    sub_158ACE0((__int64)&v54, a2);
    sub_16A5B10(&v58, &v54, (unsigned int)(2 * *(_DWORD *)(a2 + 8)));
    if ( v31 > 0x40 && v30 )
      j_j___libc_free_0_0(v30);
    v30 = v58;
    v31 = v59;
    if ( v55 > 0x40 && v54 )
      j_j___libc_free_0_0(v54);
    sub_158ABC0((__int64)&v54, a2);
    sub_16A5B10(&v58, &v54, (unsigned int)(2 * *(_DWORD *)(a2 + 8)));
    if ( v33 > 0x40 && v32 )
      j_j___libc_free_0_0(v32);
    v32 = v58;
    v33 = v59;
    if ( v55 > 0x40 && v54 )
      j_j___libc_free_0_0(v54);
    sub_158ACE0((__int64)&v54, a3);
    sub_16A5B10(&v58, &v54, (unsigned int)(2 * *(_DWORD *)(a2 + 8)));
    if ( v35 > 0x40 && v34 )
      j_j___libc_free_0_0(v34);
    v34 = v58;
    v35 = v59;
    if ( v55 > 0x40 && v54 )
      j_j___libc_free_0_0(v54);
    sub_158ABC0((__int64)&v54, a3);
    sub_16A5B10(&v58, &v54, (unsigned int)(2 * *(_DWORD *)(a2 + 8)));
    if ( v37 > 0x40 && v36 )
      j_j___libc_free_0_0(v36);
    v36 = v58;
    v37 = v59;
    if ( v55 > 0x40 && v54 )
      j_j___libc_free_0_0(v54);
    sub_16A7B50(&v58, &v30, &v34);
    sub_16A7B50(v60, &v30, &v36);
    sub_16A7B50(v61, &v32, &v34);
    v15 = a2;
    sub_16A7B50(v62, &v32, &v36);
    v16 = &v58;
    v17 = (__int64 *)v60;
    do
    {
      if ( (int)sub_16AEA10(v16, v17) < 0 )
        v16 = v17;
      v17 += 2;
    }
    while ( v17 != (__int64 *)v63 );
    v18 = v15;
    v19 = v16;
    v4 = a1;
    v41 = *((_DWORD *)v19 + 2);
    if ( v41 > 0x40 )
      sub_16A4FD0(&v40, v19);
    else
      v40 = *v19;
    sub_16A7490(&v40, 1);
    v20 = v41;
    v41 = 0;
    v21 = (__int64 *)v60;
    v29 = v18;
    v22 = &v58;
    v55 = v20;
    v54 = v40;
    do
    {
      if ( (int)sub_16AEA10(v21, v22) < 0 )
        v22 = v21;
      v21 += 2;
    }
    while ( v21 != (__int64 *)v63 );
    v39 = *((_DWORD *)v22 + 2);
    if ( v39 > 0x40 )
      sub_16A4FD0(&v38, v22);
    else
      v38 = *v22;
    sub_15898E0((__int64)&v50, (__int64)&v38, &v54);
    if ( v39 > 0x40 && v38 )
      j_j___libc_free_0_0(v38);
    if ( v55 > 0x40 && v54 )
      j_j___libc_free_0_0(v54);
    if ( v41 > 0x40 && v40 )
      j_j___libc_free_0_0(v40);
    sub_158D430((__int64)&v54, (__int64)&v50, *(_DWORD *)(v29 + 8));
    if ( sub_158A690((__int64)&v46, (__int64)&v54) )
      v7 = &v46;
    v23 = *((_DWORD *)v7 + 2);
    *(_DWORD *)(a1 + 8) = v23;
    if ( v23 > 0x40 )
      sub_16A4FD0(a1, v7);
    else
      *(_QWORD *)a1 = *v7;
    v24 = *((_DWORD *)v7 + 6);
    *(_DWORD *)(a1 + 24) = v24;
    if ( v24 > 0x40 )
      sub_16A4FD0(a1 + 16, v7 + 2);
    else
      *(_QWORD *)(a1 + 16) = v7[2];
    if ( v57 > 0x40 && v56 )
      j_j___libc_free_0_0(v56);
    if ( v55 > 0x40 && v54 )
      j_j___libc_free_0_0(v54);
    if ( v53 > 0x40 && v52 )
      j_j___libc_free_0_0(v52);
    if ( v51 > 0x40 && v50 )
      j_j___libc_free_0_0(v50);
    v25 = (__int64 *)v63;
    do
    {
      v25 -= 2;
      if ( *((_DWORD *)v25 + 2) > 0x40u && *v25 )
        j_j___libc_free_0_0(*v25);
    }
    while ( v25 != &v58 );
    if ( v49 > 0x40 && v48 )
      j_j___libc_free_0_0(v48);
    if ( v47 > 0x40 && v46 )
      j_j___libc_free_0_0(v46);
    goto LABEL_30;
  }
  v13 = v47;
  *(_DWORD *)(a1 + 24) = v9;
  *(_QWORD *)(a1 + 16) = v10;
  *(_DWORD *)(a1 + 8) = v13;
  *(_QWORD *)a1 = v46;
LABEL_30:
  if ( v45 > 0x40 && v44 )
    j_j___libc_free_0_0(v44);
  if ( v43 > 0x40 && v42 )
    j_j___libc_free_0_0(v42);
  if ( v37 > 0x40 && v36 )
    j_j___libc_free_0_0(v36);
  if ( v35 > 0x40 && v34 )
    j_j___libc_free_0_0(v34);
  if ( v33 > 0x40 && v32 )
    j_j___libc_free_0_0(v32);
  if ( v31 > 0x40 && v30 )
    j_j___libc_free_0_0(v30);
  return v4;
}
