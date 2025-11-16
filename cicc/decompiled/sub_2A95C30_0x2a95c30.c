// Function: sub_2A95C30
// Address: 0x2a95c30
//
__int64 __fastcall sub_2A95C30(__int64 a1, __int64 a2, unsigned int *a3, __int64 *a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  __int64 v7; // r15
  __int64 v8; // rsi
  __int64 v9; // rsi
  __int64 *v10; // rdi
  _BYTE *v11; // rax
  __int64 v12; // rcx
  _QWORD *v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // rbx
  __int64 (__fastcall *v17)(__int64, __int64, _BYTE *, _BYTE **, __int64, int); // rax
  _BYTE **v18; // rax
  _BYTE **v19; // rcx
  __int64 v20; // r13
  __int64 v21; // r15
  _QWORD *v22; // rax
  __int64 v23; // rbx
  __int64 v24; // r15
  __int64 v25; // r13
  __int64 v26; // rdx
  unsigned int v27; // esi
  __int64 v28; // rdi
  __int64 v29; // rcx
  __int64 v30; // rax
  unsigned int v31; // edx
  __int64 *v32; // rsi
  __int64 v33; // r9
  unsigned int v34; // esi
  __int64 v35; // rdi
  unsigned int v36; // eax
  __int64 *v37; // rcx
  __int64 v38; // rdx
  __int64 v39; // rax
  __int64 v41; // r11
  __int64 v42; // rax
  __int64 v43; // rcx
  __int64 v44; // r12
  __int64 v45; // rbx
  __int64 v46; // rdx
  unsigned int v47; // esi
  __int64 v48; // rdx
  int v49; // eax
  char v50; // cl
  bool v51; // cc
  __int64 *v52; // rax
  int v53; // eax
  int v54; // esi
  int v55; // r10d
  int v56; // r10d
  __int64 *v57; // r9
  int v58; // eax
  int v59; // edx
  __int64 v60; // rcx
  __int64 v61; // [rsp+0h] [rbp-F0h]
  __int64 v62; // [rsp+8h] [rbp-E8h]
  __int64 v67; // [rsp+30h] [rbp-C0h]
  _BYTE *v69; // [rsp+48h] [rbp-A8h] BYREF
  __int64 v70; // [rsp+50h] [rbp-A0h] BYREF
  unsigned int v71; // [rsp+58h] [rbp-98h]
  __int64 v72; // [rsp+60h] [rbp-90h] BYREF
  __m128i v73; // [rsp+68h] [rbp-88h] BYREF
  __int16 v74; // [rsp+80h] [rbp-70h]
  __int64 *v75; // [rsp+90h] [rbp-60h] BYREF
  unsigned int v76; // [rsp+98h] [rbp-58h]
  __int64 v77; // [rsp+A0h] [rbp-50h] BYREF
  __int16 v78; // [rsp+B0h] [rbp-40h]

  v6 = a2;
  v7 = *(_QWORD *)a3;
  v67 = a2 + 56;
  v8 = *(_QWORD *)(*(_QWORD *)a3 + 32LL);
  if ( v8 == *(_QWORD *)(*(_QWORD *)a3 + 40LL) + 48LL || !v8 )
    v9 = 0;
  else
    v9 = v8 - 24;
  sub_D5F1F0(v67, v9);
  v10 = *(__int64 **)(v6 + 128);
  v74 = 773;
  v72 = a5;
  v73.m128i_i64[0] = a6;
  v73.m128i_i64[1] = (__int64)"GEP";
  v11 = (_BYTE *)sub_ACCFD0(v10, (__int64)a4);
  v12 = *(_QWORD *)(v7 - 32);
  v13 = *(_QWORD **)(v6 + 128);
  v69 = v11;
  v62 = v12;
  v14 = sub_BCB2B0(v13);
  v15 = *(_QWORD *)(v6 + 136);
  v16 = v14;
  v17 = *(__int64 (__fastcall **)(__int64, __int64, _BYTE *, _BYTE **, __int64, int))(*(_QWORD *)v15 + 64LL);
  if ( v17 == sub_920540 )
  {
    if ( sub_BCEA30(v16) )
      goto LABEL_26;
    if ( *(_BYTE *)v62 > 0x15u )
      goto LABEL_26;
    v18 = sub_2A8A560(&v69, (__int64)&v70);
    if ( v19 != v18 )
      goto LABEL_26;
    LOBYTE(v78) = 0;
    v20 = sub_AD9FD0(v16, (unsigned __int8 *)v62, (__int64 *)&v69, 1, 0, (__int64)&v75, 0);
    if ( (_BYTE)v78 )
    {
      LOBYTE(v78) = 0;
      sub_969240(&v77);
      sub_969240((__int64 *)&v75);
    }
  }
  else
  {
    v20 = v17(v15, v16, (_BYTE *)v62, &v69, 1, 0);
  }
  if ( v20 )
    goto LABEL_11;
LABEL_26:
  v78 = 257;
  v20 = (__int64)sub_BD2C40(88, 2u);
  if ( !v20 )
    goto LABEL_29;
  v41 = *(_QWORD *)(v62 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v41 + 8) - 17 > 1 )
  {
    v48 = *((_QWORD *)v69 + 1);
    v49 = *(unsigned __int8 *)(v48 + 8);
    if ( v49 == 17 )
    {
      v50 = 0;
    }
    else
    {
      v50 = 1;
      if ( v49 != 18 )
        goto LABEL_28;
    }
    v53 = *(_DWORD *)(v48 + 32);
    BYTE4(v70) = v50;
    LODWORD(v70) = v53;
    v41 = sub_BCE1B0((__int64 *)v41, v70);
  }
LABEL_28:
  sub_B44260(v20, v41, 34, 2u, 0, 0);
  *(_QWORD *)(v20 + 72) = v16;
  *(_QWORD *)(v20 + 80) = sub_B4DC50(v16, (__int64)&v69, 1);
  sub_B4D9A0(v20, v62, (__int64 *)&v69, 1, (__int64)&v75);
LABEL_29:
  sub_B4DDE0(v20, 0);
  (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v6 + 144) + 16LL))(
    *(_QWORD *)(v6 + 144),
    v20,
    &v72,
    *(_QWORD *)(v67 + 56),
    *(_QWORD *)(v67 + 64));
  v42 = *(_QWORD *)(v6 + 56);
  v43 = v42 + 16LL * *(unsigned int *)(v6 + 64);
  if ( v42 != v43 )
  {
    v61 = v6;
    v44 = *(_QWORD *)(v6 + 56);
    v45 = v43;
    do
    {
      v46 = *(_QWORD *)(v44 + 8);
      v47 = *(_DWORD *)v44;
      v44 += 16;
      sub_B99FD0(v20, v47, v46);
    }
    while ( v45 != v44 );
    v6 = v61;
  }
LABEL_11:
  v74 = 261;
  v72 = a5;
  v73.m128i_i64[0] = a6;
  v21 = *(_QWORD *)(v7 + 8);
  v78 = 257;
  v22 = sub_BD2C40(80, 1u);
  v23 = (__int64)v22;
  if ( v22 )
    sub_B4D190((__int64)v22, v21, v20, (__int64)&v75, 0, 0, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(v6 + 144) + 16LL))(
    *(_QWORD *)(v6 + 144),
    v23,
    &v72,
    *(_QWORD *)(v67 + 56),
    *(_QWORD *)(v67 + 64));
  v24 = *(_QWORD *)(v6 + 56);
  v25 = v24 + 16LL * *(unsigned int *)(v6 + 64);
  while ( v25 != v24 )
  {
    v26 = *(_QWORD *)(v24 + 8);
    v27 = *(_DWORD *)v24;
    v24 += 16;
    sub_B99FD0(v23, v27, v26);
  }
  sub_CE85E0(v23, 0);
  v76 = a3[4];
  if ( v76 > 0x40 )
    sub_C43780((__int64)&v75, (const void **)a3 + 1);
  else
    v75 = (__int64 *)*((_QWORD *)a3 + 1);
  sub_C45EE0((__int64)&v75, a4);
  v28 = *(_QWORD *)(v6 + 1312);
  v71 = v76;
  v70 = (__int64)v75;
  v29 = *(_QWORD *)a3;
  v30 = *(unsigned int *)(v6 + 1328);
  if ( (_DWORD)v30 )
  {
    v31 = (v30 - 1) & (((unsigned int)v29 >> 9) ^ ((unsigned int)v29 >> 4));
    v32 = (__int64 *)(v28 + 24LL * v31);
    v33 = *v32;
    if ( v29 == *v32 )
    {
LABEL_19:
      if ( v32 != (__int64 *)(v28 + 24 * v30) )
      {
        v72 = v23;
        v73 = _mm_loadu_si128((const __m128i *)(v32 + 1));
        sub_2A8CE20((__int64)&v75, v6 + 1304, &v72, &v73);
      }
    }
    else
    {
      v54 = 1;
      while ( v33 != -4096 )
      {
        v55 = v54 + 1;
        v31 = (v30 - 1) & (v54 + v31);
        v32 = (__int64 *)(v28 + 24LL * v31);
        v33 = *v32;
        if ( v29 == *v32 )
          goto LABEL_19;
        v54 = v55;
      }
    }
  }
  v34 = *(_DWORD *)(v6 + 1296);
  v72 = v23;
  if ( !v34 )
  {
    ++*(_QWORD *)(v6 + 1272);
    v75 = 0;
LABEL_58:
    v34 *= 2;
    goto LABEL_59;
  }
  v35 = *(_QWORD *)(v6 + 1280);
  v36 = (v34 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
  v37 = (__int64 *)(v35 + 8LL * v36);
  v38 = *v37;
  if ( v23 == *v37 )
    goto LABEL_23;
  v56 = 1;
  v57 = 0;
  while ( v38 != -4096 )
  {
    if ( v38 != -8192 || v57 )
      v37 = v57;
    v36 = (v34 - 1) & (v56 + v36);
    v38 = *(_QWORD *)(v35 + 8LL * v36);
    if ( v23 == v38 )
      goto LABEL_23;
    ++v56;
    v57 = v37;
    v37 = (__int64 *)(v35 + 8LL * v36);
  }
  v58 = *(_DWORD *)(v6 + 1288);
  if ( !v57 )
    v57 = v37;
  ++*(_QWORD *)(v6 + 1272);
  v59 = v58 + 1;
  v75 = v57;
  if ( 4 * (v58 + 1) >= 3 * v34 )
    goto LABEL_58;
  v60 = v23;
  if ( v34 - *(_DWORD *)(v6 + 1292) - v59 <= v34 >> 3 )
  {
LABEL_59:
    sub_CF4090(v6 + 1272, v34);
    sub_23FDF60(v6 + 1272, &v72, &v75);
    v60 = v72;
    v57 = v75;
    v59 = *(_DWORD *)(v6 + 1288) + 1;
  }
  *(_DWORD *)(v6 + 1288) = v59;
  if ( *v57 != -4096 )
    --*(_DWORD *)(v6 + 1292);
  *v57 = v60;
LABEL_23:
  v76 = v71;
  if ( v71 > 0x40 )
  {
    sub_C43780((__int64)&v75, (const void **)&v70);
    v51 = v71 <= 0x40;
    *(_DWORD *)(a1 + 16) = v76;
    v52 = v75;
    *(_QWORD *)a1 = v23;
    *(_QWORD *)(a1 + 8) = v52;
    if ( !v51 && v70 )
      j_j___libc_free_0_0(v70);
  }
  else
  {
    *(_DWORD *)(a1 + 16) = v71;
    v39 = v70;
    *(_QWORD *)a1 = v23;
    *(_QWORD *)(a1 + 8) = v39;
  }
  return a1;
}
