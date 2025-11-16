// Function: sub_11EA710
// Address: 0x11ea710
//
__int64 __fastcall sub_11EA710(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  __int64 v5; // r14
  __int64 v6; // r12
  __int64 v7; // rcx
  __int64 v8; // r8
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rsi
  __int64 v12; // r14
  __int64 v13; // r12
  __int64 v14; // rax
  unsigned __int8 *v15; // r9
  __int64 v16; // r13
  __int64 v17; // rax
  __int64 v18; // r15
  unsigned int v19; // ebx
  __int64 v20; // rdi
  __int64 (__fastcall *v21)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  _QWORD *v22; // rax
  __int64 v23; // r9
  __int64 v24; // r13
  _QWORD **v25; // rdx
  int v26; // ecx
  int v27; // eax
  __int64 *v28; // rax
  __int64 v29; // rax
  __int64 v30; // r12
  __int64 v31; // rbx
  __int64 v32; // rdx
  unsigned int v33; // esi
  char v35; // r14
  __int64 v36; // rax
  _WORD *v37; // rsi
  __int64 v38; // r14
  _QWORD *v39; // rdi
  int v40; // eax
  __int64 v41; // r15
  __int64 v42; // r12
  __int64 v43; // rax
  _BYTE *v44; // rax
  __int64 v45; // rdi
  __int64 (__fastcall *v46)(__int64, __int64, _BYTE *, _BYTE **, __int64, int); // rax
  _BYTE **v47; // rax
  _BYTE **v48; // rcx
  __int64 v49; // r10
  __int64 v50; // rbx
  __int64 v51; // r12
  __int64 v52; // rdx
  unsigned int v53; // esi
  __int64 v54; // rax
  int v55; // edx
  char v56; // dl
  int v57; // eax
  __int64 v58; // [rsp+8h] [rbp-E8h]
  unsigned __int8 *v59; // [rsp+18h] [rbp-D8h]
  __int64 v60; // [rsp+18h] [rbp-D8h]
  __int64 v61; // [rsp+18h] [rbp-D8h]
  unsigned __int8 *v62; // [rsp+18h] [rbp-D8h]
  __int64 v63; // [rsp+20h] [rbp-D0h]
  _BYTE *v65; // [rsp+30h] [rbp-C0h] BYREF
  __int64 v66; // [rsp+38h] [rbp-B8h] BYREF
  __int64 v67[2]; // [rsp+40h] [rbp-B0h] BYREF
  _WORD *v68; // [rsp+50h] [rbp-A0h] BYREF
  size_t v69; // [rsp+58h] [rbp-98h]
  _QWORD v70[4]; // [rsp+60h] [rbp-90h] BYREF
  char v71; // [rsp+80h] [rbp-70h]
  char v72; // [rsp+81h] [rbp-6Fh]
  __int64 v73; // [rsp+90h] [rbp-60h] BYREF
  unsigned int v74; // [rsp+98h] [rbp-58h]
  __int64 v75; // [rsp+A0h] [rbp-50h]
  unsigned int v76; // [rsp+A8h] [rbp-48h]
  __int16 v77; // [rsp+B0h] [rbp-40h]

  v4 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v5 = *(_QWORD *)(a2 - 32 * v4);
  if ( *(_QWORD *)(a2 + 32 * (1 - v4)) == v5 )
    return v5;
  v6 = a2;
  if ( !(unsigned __int8)sub_11D9DE0(*(_QWORD *)(a2 + 16), v5) )
  {
    v67[0] = 0;
    v67[1] = 0;
    v68 = 0;
    v69 = 0;
    v35 = sub_98B0F0(v5, v67, 1u);
    if ( !(unsigned __int8)sub_98B0F0(*(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF))), &v68, 1u) )
    {
LABEL_29:
      v73 = 0x100000000LL;
      sub_11DA4B0(a2, (int *)&v73, 2);
      return 0;
    }
    if ( !v69 )
      return *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    if ( !v35 )
    {
      if ( v69 == 1 )
        return sub_11CA130(
                 *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)),
                 *(_BYTE *)v68,
                 a3,
                 *(__int64 **)(a1 + 24));
      goto LABEL_29;
    }
    v37 = v68;
    v38 = sub_C931B0(v67, v68, v69, 0);
    if ( v38 == -1 )
      return sub_AD6530(*(_QWORD *)(v6 + 8), (__int64)v37);
    v39 = *(_QWORD **)(a3 + 72);
    v72 = 1;
    v70[0] = "strstr";
    v40 = *(_DWORD *)(v6 + 4);
    v71 = 3;
    v41 = *(_QWORD *)(v6 - 32LL * (v40 & 0x7FFFFFF));
    v42 = sub_BCB2B0(v39);
    v43 = sub_BCB2E0(*(_QWORD **)(a3 + 72));
    v44 = (_BYTE *)sub_ACD640(v43, v38, 0);
    v45 = *(_QWORD *)(a3 + 80);
    v65 = v44;
    v46 = *(__int64 (__fastcall **)(__int64, __int64, _BYTE *, _BYTE **, __int64, int))(*(_QWORD *)v45 + 64LL);
    if ( v46 == sub_920540 )
    {
      if ( sub_BCEA30(v42) )
        goto LABEL_41;
      if ( *(_BYTE *)v41 > 0x15u )
        goto LABEL_41;
      v47 = sub_11D9D10(&v65, (__int64)&v66);
      if ( v48 != v47 )
        goto LABEL_41;
      LOBYTE(v77) = 0;
      v5 = sub_AD9FD0(v42, (unsigned __int8 *)v41, (__int64 *)&v65, 1, 3u, (__int64)&v73, 0);
      if ( (_BYTE)v77 )
      {
        LOBYTE(v77) = 0;
        if ( v76 > 0x40 && v75 )
          j_j___libc_free_0_0(v75);
        if ( v74 > 0x40 && v73 )
          j_j___libc_free_0_0(v73);
      }
    }
    else
    {
      v5 = v46(v45, v42, (_BYTE *)v41, &v65, 1, 3);
    }
    if ( v5 )
      return v5;
LABEL_41:
    v77 = 257;
    v5 = (__int64)sub_BD2C40(88, 2u);
    if ( !v5 )
      goto LABEL_44;
    v49 = *(_QWORD *)(v41 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v49 + 8) - 17 <= 1 )
    {
LABEL_43:
      sub_B44260(v5, v49, 34, 2u, 0, 0);
      *(_QWORD *)(v5 + 72) = v42;
      *(_QWORD *)(v5 + 80) = sub_B4DC50(v42, (__int64)&v65, 1);
      sub_B4D9A0(v5, v41, (__int64 *)&v65, 1, (__int64)&v73);
LABEL_44:
      sub_B4DDE0(v5, 3);
      (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
        *(_QWORD *)(a3 + 88),
        v5,
        v70,
        *(_QWORD *)(a3 + 56),
        *(_QWORD *)(a3 + 64));
      v50 = *(_QWORD *)a3;
      v51 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
      if ( *(_QWORD *)a3 != v51 )
      {
        do
        {
          v52 = *(_QWORD *)(v50 + 8);
          v53 = *(_DWORD *)v50;
          v50 += 16;
          sub_B99FD0(v5, v53, v52);
        }
        while ( v51 != v50 );
      }
      return v5;
    }
    v54 = *((_QWORD *)v65 + 1);
    v55 = *(unsigned __int8 *)(v54 + 8);
    if ( v55 == 17 )
    {
      v56 = 0;
    }
    else
    {
      if ( v55 != 18 )
        goto LABEL_43;
      v56 = 1;
    }
    v57 = *(_DWORD *)(v54 + 32);
    BYTE4(v66) = v56;
    LODWORD(v66) = v57;
    v49 = sub_BCE1B0((__int64 *)v49, v66);
    goto LABEL_43;
  }
  v9 = sub_11CA050(v8, a3, *(_QWORD *)(v7 + 16), *(__int64 **)(v7 + 24));
  if ( !v9 )
    return 0;
  v10 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  v11 = *(_QWORD *)(a2 + 32 * (1 - v10));
  v12 = sub_11CA1D0(*(_QWORD *)(v6 - 32 * v10), v11, v9, a3, *(_QWORD *)(a1 + 16), *(__int64 **)(a1 + 24));
  if ( !v12 )
    return 0;
  if ( !*(_QWORD *)(v6 + 16) )
    return v6;
  v58 = v6;
  v13 = *(_QWORD *)(v6 + 16);
  v63 = a3;
  do
  {
    v17 = v13;
    v13 = *(_QWORD *)(v13 + 8);
    v18 = *(_QWORD *)(v17 + 24);
    v72 = 1;
    v70[0] = "cmp";
    v71 = 3;
    v15 = (unsigned __int8 *)sub_AD6530(*(_QWORD *)(v12 + 8), v11);
    v19 = *(_WORD *)(v18 + 2) & 0x3F;
    v20 = *(_QWORD *)(v63 + 80);
    v21 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *))(*(_QWORD *)v20 + 56LL);
    if ( v21 != sub_928890 )
    {
      v62 = v15;
      v36 = v21(v20, v19, (_BYTE *)v12, v15);
      v15 = v62;
      v16 = v36;
LABEL_9:
      if ( v16 )
        goto LABEL_10;
      goto LABEL_13;
    }
    if ( *(_BYTE *)v12 <= 0x15u && *v15 <= 0x15u )
    {
      v59 = v15;
      v14 = sub_AAB310(v19, (unsigned __int8 *)v12, v15);
      v15 = v59;
      v16 = v14;
      goto LABEL_9;
    }
LABEL_13:
    v60 = (__int64)v15;
    v77 = 257;
    v22 = sub_BD2C40(72, unk_3F10FD0);
    v23 = v60;
    v24 = (__int64)v22;
    if ( v22 )
    {
      v25 = *(_QWORD ***)(v12 + 8);
      v26 = *((unsigned __int8 *)v25 + 8);
      if ( (unsigned int)(v26 - 17) > 1 )
      {
        v29 = sub_BCB2A0(*v25);
      }
      else
      {
        v27 = *((_DWORD *)v25 + 8);
        BYTE4(v68) = (_BYTE)v26 == 18;
        LODWORD(v68) = v27;
        v28 = (__int64 *)sub_BCB2A0(*v25);
        v29 = sub_BCE1B0(v28, (__int64)v68);
      }
      sub_B523C0(v24, v29, 53, v19, v12, v60, (__int64)&v73, 0, 0, 0);
    }
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD, __int64))(**(_QWORD **)(v63 + 88) + 16LL))(
      *(_QWORD *)(v63 + 88),
      v24,
      v70,
      *(_QWORD *)(v63 + 56),
      *(_QWORD *)(v63 + 64),
      v23);
    if ( *(_QWORD *)v63 != *(_QWORD *)v63 + 16LL * *(unsigned int *)(v63 + 8) )
    {
      v61 = v13;
      v30 = *(_QWORD *)v63;
      v31 = *(_QWORD *)v63 + 16LL * *(unsigned int *)(v63 + 8);
      do
      {
        v32 = *(_QWORD *)(v30 + 8);
        v33 = *(_DWORD *)v30;
        v30 += 16;
        sub_B99FD0(v24, v33, v32);
      }
      while ( v31 != v30 );
      v13 = v61;
    }
LABEL_10:
    v11 = v18;
    sub_11EA700(a1);
  }
  while ( v13 );
  return v58;
}
