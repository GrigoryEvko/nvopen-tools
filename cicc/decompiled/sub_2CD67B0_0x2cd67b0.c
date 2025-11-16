// Function: sub_2CD67B0
// Address: 0x2cd67b0
//
__int64 __fastcall sub_2CD67B0(__int64 a1, unsigned __int64 a2, __int64 a3, char a4)
{
  __int64 v4; // rax
  __int64 v6; // rsi
  __int64 *v8; // rax
  __int64 **v9; // r15
  const char *v10; // rax
  __int64 *v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r9
  __int64 v14; // r8
  __int64 (__fastcall *v15)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v16; // r12
  __int64 *v17; // rax
  __int64 *v18; // rdx
  __int64 *v19; // r13
  __int64 v20; // rbx
  unsigned __int8 v21; // al
  unsigned int v22; // r15d
  unsigned __int8 v23; // r14
  _QWORD *v24; // rbx
  char *v25; // r15
  char *v26; // r13
  __int64 v27; // rdx
  unsigned int v28; // esi
  __int64 *v29; // rdx
  __int64 v30; // r13
  __int64 v31; // rax
  char v32; // r9
  _QWORD *v33; // rax
  __int64 v34; // r9
  __int64 v35; // r15
  char *v36; // r13
  char *v37; // r12
  __int64 v38; // rdx
  unsigned int v39; // esi
  char v40; // r12
  _QWORD *v41; // rax
  __int64 v42; // r9
  __int64 v43; // r13
  char *v44; // r12
  char *v45; // r14
  __int64 v46; // rdx
  unsigned int v47; // esi
  int v49; // r15d
  char *v50; // r15
  char *v51; // rbx
  __int64 v52; // rdx
  unsigned int v53; // esi
  const char *v54; // rax
  __int64 *v55; // rdx
  __int64 v56; // rax
  __int64 v57; // [rsp-10h] [rbp-140h]
  __int16 v58; // [rsp+0h] [rbp-130h]
  char v59; // [rsp+0h] [rbp-130h]
  char v60; // [rsp+5h] [rbp-12Bh]
  const char *v62; // [rsp+10h] [rbp-120h] BYREF
  __int64 *v63; // [rsp+18h] [rbp-118h]
  char *v64; // [rsp+20h] [rbp-110h]
  __int16 v65; // [rsp+30h] [rbp-100h]
  _BYTE v66[32]; // [rsp+40h] [rbp-F0h] BYREF
  __int16 v67; // [rsp+60h] [rbp-D0h]
  char *v68; // [rsp+70h] [rbp-C0h] BYREF
  int v69; // [rsp+78h] [rbp-B8h]
  char v70; // [rsp+80h] [rbp-B0h] BYREF
  __int64 v71; // [rsp+A0h] [rbp-90h]
  __int64 v72; // [rsp+A8h] [rbp-88h]
  __int64 v73; // [rsp+B0h] [rbp-80h]
  __int64 v74; // [rsp+C0h] [rbp-70h]
  __int64 v75; // [rsp+C8h] [rbp-68h]
  __int64 v76; // [rsp+D0h] [rbp-60h]
  int v77; // [rsp+D8h] [rbp-58h]
  void *v78; // [rsp+F0h] [rbp-40h]

  v4 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  if ( !v4 )
    BUG();
  v6 = *(_QWORD *)(v4 + 32);
  if ( v6 )
    v6 -= 24;
  sub_23D0AB0((__int64)&v68, v6, 0, 0, 0);
  v8 = (__int64 *)sub_BD5C60(a2);
  v9 = (__int64 **)sub_BCE3C0(v8, 101);
  v10 = sub_BD5D20(a2);
  v14 = 773;
  v62 = v10;
  v65 = 773;
  v63 = v11;
  v64 = ".param";
  if ( v9 == *(__int64 ***)(a2 + 8) )
  {
    v16 = a2;
    goto LABEL_10;
  }
  v15 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, __int64))(*(_QWORD *)v74 + 120LL);
  if ( v15 != sub_920130 )
  {
    v16 = v15(v74, 50u, (_BYTE *)a2, (__int64)v9);
    goto LABEL_9;
  }
  if ( *(_BYTE *)a2 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC4810(0x32u) )
      v16 = sub_ADAB70(50, a2, v9, 0);
    else
      v16 = sub_AA93C0(0x32u, a2, (__int64)v9);
LABEL_9:
    if ( v16 )
      goto LABEL_10;
  }
  v67 = 257;
  v16 = sub_B51D30(50, a2, (__int64)v9, (__int64)v66, 0, 0);
  if ( (unsigned __int8)sub_920620(v16) )
  {
    v49 = v77;
    if ( v76 )
      sub_B99FD0(v16, 3u, v76);
    sub_B45150(v16, v49);
  }
  (*(void (__fastcall **)(__int64, __int64, const char **, __int64, __int64))(*(_QWORD *)v75 + 16LL))(
    v75,
    v16,
    &v62,
    v72,
    v73);
  v50 = &v68[16 * v69];
  if ( v68 != v50 )
  {
    v51 = v68;
    do
    {
      v52 = *((_QWORD *)v51 + 1);
      v53 = *(_DWORD *)v51;
      v51 += 16;
      sub_B99FD0(v16, v53, v52);
    }
    while ( v50 != v51 );
    if ( !*(_BYTE *)(a3 + 28) )
      goto LABEL_44;
    goto LABEL_11;
  }
LABEL_10:
  if ( !*(_BYTE *)(a3 + 28) )
    goto LABEL_44;
LABEL_11:
  v17 = *(__int64 **)(a3 + 8);
  v12 = *(unsigned int *)(a3 + 20);
  v11 = &v17[v12];
  if ( v17 == v11 )
  {
LABEL_45:
    if ( (unsigned int)v12 < *(_DWORD *)(a3 + 16) )
    {
      *(_DWORD *)(a3 + 20) = v12 + 1;
      *v11 = v16;
      ++*(_QWORD *)a3;
      goto LABEL_15;
    }
LABEL_44:
    sub_C8CC70(a3, v16, (__int64)v11, v12, v14, v13);
    goto LABEL_15;
  }
  while ( v16 != *v17 )
  {
    if ( v11 == ++v17 )
      goto LABEL_45;
  }
LABEL_15:
  if ( a4 )
  {
    v58 = sub_B2BD00(a2);
    v60 = v58;
    v62 = sub_BD5D20(a2);
    v65 = 773;
    v63 = v18;
    v64 = ".copy";
    v19 = (__int64 *)sub_B2BD20(a2);
    v20 = sub_AA4E30(v71);
    v21 = sub_AE5260(v20, (__int64)v19);
    v22 = *(_DWORD *)(v20 + 4);
    v23 = v21;
    v67 = 257;
    v24 = sub_BD2C40(80, 1u);
    if ( v24 )
      sub_B4CCA0((__int64)v24, v19, v22, 0, v23, (__int64)v66, 0, 0);
    (*(void (__fastcall **)(__int64, _QWORD *, const char **, __int64, __int64))(*(_QWORD *)v75 + 16LL))(
      v75,
      v24,
      &v62,
      v72,
      v73);
    v25 = v68;
    v26 = &v68[16 * v69];
    if ( v68 != v26 )
    {
      do
      {
        v27 = *((_QWORD *)v25 + 1);
        v28 = *(_DWORD *)v25;
        v25 += 16;
        sub_B99FD0((__int64)v24, v28, v27);
      }
      while ( v26 != v25 );
    }
    if ( HIBYTE(v58) )
    {
      *((_WORD *)v24 + 1) = *((_WORD *)v24 + 1) & 0xFFC0 | (unsigned __int8)v58;
      v54 = sub_BD5D20(v16);
      v65 = 773;
      v32 = v58;
      v62 = v54;
      v63 = v55;
      v64 = ".copy";
      v30 = v24[9];
    }
    else
    {
      *((_WORD *)v24 + 1) &= 0xFFC0u;
      v62 = sub_BD5D20(v16);
      v65 = 773;
      v63 = v29;
      v64 = ".copy";
      v30 = v24[9];
      v31 = sub_AA4E30(v71);
      v32 = sub_AE5020(v31, v30);
    }
    v59 = v32;
    v67 = 257;
    v33 = sub_BD2C40(80, 1u);
    v35 = (__int64)v33;
    if ( v33 )
    {
      sub_B4D190((__int64)v33, v30, v16, (__int64)v66, 0, v59, 0, 0);
      v34 = v57;
    }
    (*(void (__fastcall **)(__int64, __int64, const char **, __int64, __int64, __int64))(*(_QWORD *)v75 + 16LL))(
      v75,
      v35,
      &v62,
      v72,
      v73,
      v34);
    v36 = v68;
    v37 = &v68[16 * v69];
    if ( v68 != v37 )
    {
      do
      {
        v38 = *((_QWORD *)v36 + 1);
        v39 = *(_DWORD *)v36;
        v36 += 16;
        sub_B99FD0(v35, v39, v38);
      }
      while ( v37 != v36 );
    }
    v40 = v60;
    if ( !HIBYTE(v58) )
    {
      v56 = sub_AA4E30(v71);
      v40 = sub_AE5020(v56, *(_QWORD *)(v35 + 8));
    }
    v67 = 257;
    v41 = sub_BD2C40(80, unk_3F10A10);
    v43 = (__int64)v41;
    if ( v41 )
      sub_B4D3C0((__int64)v41, v35, (__int64)v24, 0, v40, v42, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v75 + 16LL))(
      v75,
      v43,
      v66,
      v72,
      v73);
    v44 = v68;
    v45 = &v68[16 * v69];
    if ( v68 != v45 )
    {
      do
      {
        v46 = *((_QWORD *)v44 + 1);
        v47 = *(_DWORD *)v44;
        v44 += 16;
        sub_B99FD0(v43, v47, v46);
      }
      while ( v45 != v44 );
    }
    v16 = (__int64)v24;
  }
  nullsub_61();
  v78 = &unk_49DA100;
  nullsub_63();
  if ( v68 != &v70 )
    _libc_free((unsigned __int64)v68);
  return v16;
}
