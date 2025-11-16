// Function: sub_2436FF0
// Address: 0x2436ff0
//
unsigned __int64 __fastcall sub_2436FF0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rsi
  __int64 v7; // rdi
  __int64 v8; // rax
  __int64 v9; // rdi
  unsigned __int8 *v10; // rbx
  __int64 (__fastcall *v11)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8); // rax
  _BYTE *v12; // r15
  _QWORD *v13; // rdi
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rdi
  __int64 v17; // r14
  __int64 (__fastcall *v18)(__int64, __int64, _BYTE *, _BYTE **, __int64, int); // rax
  _BYTE **v19; // rax
  _BYTE **v20; // rcx
  __int64 v21; // r13
  __int64 v23; // rbx
  __int64 v24; // r14
  __int64 v25; // rdx
  unsigned int v26; // esi
  __int64 v27; // r10
  __int64 v28; // rbx
  __int64 v29; // r12
  __int64 v30; // rdx
  unsigned int v31; // esi
  __int64 **v32; // rcx
  __int64 v33; // rdx
  int v34; // eax
  char v35; // al
  int v36; // edx
  _BYTE *v37; // [rsp+0h] [rbp-A0h] BYREF
  __int64 v38; // [rsp+8h] [rbp-98h] BYREF
  int v39[8]; // [rsp+10h] [rbp-90h] BYREF
  __int16 v40; // [rsp+30h] [rbp-70h]
  unsigned __int64 v41; // [rsp+40h] [rbp-60h] BYREF
  unsigned int v42; // [rsp+48h] [rbp-58h]
  unsigned __int64 v43; // [rsp+50h] [rbp-50h]
  unsigned int v44; // [rsp+58h] [rbp-48h]
  __int16 v45; // [rsp+60h] [rbp-40h]

  v6 = *(unsigned __int8 *)(a1 + 104);
  v7 = *(_QWORD *)(a2 + 8);
  v40 = 257;
  v8 = sub_AD64C0(v7, v6, 0);
  v9 = *(_QWORD *)(a3 + 80);
  v10 = (unsigned __int8 *)v8;
  v11 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8))(*(_QWORD *)v9 + 24LL);
  if ( v11 != sub_920250 )
  {
    v12 = (_BYTE *)v11(v9, 26u, (_BYTE *)a2, v10, 0);
    goto LABEL_6;
  }
  if ( *(_BYTE *)a2 <= 0x15u && *v10 <= 0x15u )
  {
    if ( (unsigned __int8)sub_AC47B0(26) )
      v12 = (_BYTE *)sub_AD5570(26, a2, v10, 0, 0);
    else
      v12 = (_BYTE *)sub_AABE40(0x1Au, (unsigned __int8 *)a2, v10);
LABEL_6:
    if ( v12 )
      goto LABEL_7;
  }
  v45 = 257;
  v12 = (_BYTE *)sub_B504D0(26, a2, (__int64)v10, (__int64)&v41, 0, 0);
  (*(void (__fastcall **)(_QWORD, _BYTE *, int *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
    *(_QWORD *)(a3 + 88),
    v12,
    v39,
    *(_QWORD *)(a3 + 56),
    *(_QWORD *)(a3 + 64));
  v23 = *(_QWORD *)a3;
  v24 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
  if ( *(_QWORD *)a3 != v24 )
  {
    do
    {
      v25 = *(_QWORD *)(v23 + 8);
      v26 = *(_DWORD *)v23;
      v23 += 16;
      sub_B99FD0((__int64)v12, v26, v25);
    }
    while ( v24 != v23 );
  }
LABEL_7:
  if ( !*(_DWORD *)(a1 + 88) && !*(_QWORD *)(a1 + 96) )
  {
    v32 = *(__int64 ***)(a1 + 128);
    v45 = 257;
    return sub_2436E50((__int64 *)a3, 0x30u, (unsigned __int64)v12, v32, (__int64)&v41, 0, v39[0], 0);
  }
  v13 = *(_QWORD **)(a3 + 72);
  v37 = v12;
  v40 = 257;
  v14 = *(_QWORD *)(a1 + 512);
  v15 = sub_BCB2B0(v13);
  v16 = *(_QWORD *)(a3 + 80);
  v17 = v15;
  v18 = *(__int64 (__fastcall **)(__int64, __int64, _BYTE *, _BYTE **, __int64, int))(*(_QWORD *)v16 + 64LL);
  if ( v18 == sub_920540 )
  {
    if ( sub_BCEA30(v17) )
      goto LABEL_19;
    if ( *(_BYTE *)v14 > 0x15u )
      goto LABEL_19;
    v19 = sub_2433F20(&v37, (__int64)&v38);
    if ( v20 != v19 )
      goto LABEL_19;
    LOBYTE(v45) = 0;
    v21 = sub_AD9FD0(v17, (unsigned __int8 *)v14, (__int64 *)&v37, 1, 0, (__int64)&v41, 0);
    if ( (_BYTE)v45 )
    {
      LOBYTE(v45) = 0;
      if ( v44 > 0x40 && v43 )
        j_j___libc_free_0_0(v43);
      if ( v42 > 0x40 && v41 )
        j_j___libc_free_0_0(v41);
    }
  }
  else
  {
    v21 = v18(v16, v17, (_BYTE *)v14, &v37, 1, 0);
  }
  if ( v21 )
    return v21;
LABEL_19:
  v45 = 257;
  v21 = (__int64)sub_BD2C40(88, 2u);
  if ( !v21 )
    goto LABEL_22;
  v27 = *(_QWORD *)(v14 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v27 + 8) - 17 > 1 )
  {
    v33 = *((_QWORD *)v37 + 1);
    v34 = *(unsigned __int8 *)(v33 + 8);
    if ( v34 == 17 )
    {
      v35 = 0;
    }
    else
    {
      if ( v34 != 18 )
        goto LABEL_21;
      v35 = 1;
    }
    v36 = *(_DWORD *)(v33 + 32);
    BYTE4(v38) = v35;
    LODWORD(v38) = v36;
    v27 = sub_BCE1B0((__int64 *)v27, v38);
  }
LABEL_21:
  sub_B44260(v21, v27, 34, 2u, 0, 0);
  *(_QWORD *)(v21 + 72) = v17;
  *(_QWORD *)(v21 + 80) = sub_B4DC50(v17, (__int64)&v37, 1);
  sub_B4D9A0(v21, v14, (__int64 *)&v37, 1, (__int64)&v41);
LABEL_22:
  sub_B4DDE0(v21, 0);
  (*(void (__fastcall **)(_QWORD, __int64, int *, _QWORD, _QWORD))(**(_QWORD **)(a3 + 88) + 16LL))(
    *(_QWORD *)(a3 + 88),
    v21,
    v39,
    *(_QWORD *)(a3 + 56),
    *(_QWORD *)(a3 + 64));
  v28 = *(_QWORD *)a3;
  v29 = *(_QWORD *)a3 + 16LL * *(unsigned int *)(a3 + 8);
  while ( v29 != v28 )
  {
    v30 = *(_QWORD *)(v28 + 8);
    v31 = *(_DWORD *)v28;
    v28 += 16;
    sub_B99FD0(v21, v31, v30);
  }
  return v21;
}
