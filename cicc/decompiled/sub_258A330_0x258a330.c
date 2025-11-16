// Function: sub_258A330
// Address: 0x258a330
//
__int64 __fastcall sub_258A330(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  void *v7; // rax
  int v8; // edi
  signed int v9; // r15d
  __int64 v10; // rdx
  unsigned int v11; // r13d
  __int64 v13; // rsi
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 (__fastcall *v16)(__int64); // rax
  __int64 v17; // r15
  __int64 v18; // rsi
  _DWORD *v19; // rdi
  __int64 (__fastcall *v20)(__int64); // rax
  int v21; // eax
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  __int64 v26; // rax
  __int64 v27; // rax
  bool v28; // cc
  unsigned __int64 v29; // rdi
  unsigned __int64 v30; // rdi
  unsigned __int64 v31; // rdi
  unsigned __int64 v32; // rdi
  unsigned int v33; // edx
  unsigned int v34; // edx
  __int64 v35; // [rsp-10h] [rbp-F0h]
  __int64 v36; // [rsp+0h] [rbp-E0h]
  __int64 v37; // [rsp+0h] [rbp-E0h]
  __int64 v38; // [rsp+0h] [rbp-E0h]
  __int64 v39; // [rsp+0h] [rbp-E0h]
  __int64 v40; // [rsp+0h] [rbp-E0h]
  __int64 v41; // [rsp+0h] [rbp-E0h]
  __int64 v42; // [rsp+0h] [rbp-E0h]
  __int64 v43; // [rsp+8h] [rbp-D8h]
  __int64 v44; // [rsp+8h] [rbp-D8h]
  __int64 v45; // [rsp+10h] [rbp-D0h] BYREF
  __int64 v46; // [rsp+18h] [rbp-C8h]
  void *v47; // [rsp+20h] [rbp-C0h]
  char *v48; // [rsp+28h] [rbp-B8h] BYREF
  __int64 v49; // [rsp+30h] [rbp-B0h]
  char v50; // [rsp+38h] [rbp-A8h] BYREF
  const void *v51; // [rsp+40h] [rbp-A0h] BYREF
  unsigned int v52; // [rsp+48h] [rbp-98h]
  const void *v53; // [rsp+50h] [rbp-90h] BYREF
  unsigned int v54; // [rsp+58h] [rbp-88h]
  void *v55; // [rsp+60h] [rbp-80h] BYREF
  unsigned int *v56; // [rsp+68h] [rbp-78h] BYREF
  const void *v57; // [rsp+70h] [rbp-70h] BYREF
  unsigned int v58; // [rsp+78h] [rbp-68h] BYREF
  const void *v59; // [rsp+80h] [rbp-60h] BYREF
  unsigned int v60; // [rsp+88h] [rbp-58h]
  __int64 v61; // [rsp+90h] [rbp-50h] BYREF
  int v62; // [rsp+98h] [rbp-48h]
  __int64 v63; // [rsp+A0h] [rbp-40h]
  int v64; // [rsp+A8h] [rbp-38h]

  v7 = *(void **)a2;
  v8 = *(_DWORD *)(a2 + 16);
  v48 = &v50;
  v49 = 0;
  v47 = v7;
  if ( v8 )
  {
    sub_2538240((__int64)&v48, (char **)(a2 + 8), a3, a4, a5, a6);
    v9 = **(_DWORD **)a1;
    v56 = &v58;
    v57 = 0;
    v55 = v47;
    if ( (_DWORD)v49 )
      sub_2538550((__int64)&v56, (__int64)&v48, v22, v23, v24, v25);
  }
  else
  {
    v9 = **(_DWORD **)a1;
    v55 = v7;
    v56 = &v58;
    v57 = 0;
  }
  v45 = sub_254CA10((__int64)&v55, v9);
  v46 = v10;
  if ( v56 != &v58 )
    _libc_free((unsigned __int64)v56);
  if ( !(unsigned __int8)sub_2509800(&v45) )
    goto LABEL_6;
  v13 = v45;
  v14 = sub_2589400(*(_QWORD *)(a1 + 8), v45, v46, *(_QWORD *)(a1 + 16), 0, 0, 1);
  v15 = v14;
  if ( !v14 )
    goto LABEL_6;
  v16 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v14 + 48LL);
  if ( v16 == sub_2534AC0 )
    v17 = v15 + 88;
  else
    v17 = ((__int64 (__fastcall *)(__int64, __int64, __int64))v16)(v15, v13, v35);
  v18 = *(_QWORD *)(a1 + 24);
  if ( !*(_BYTE *)(v18 + 80) )
  {
    sub_AADB10((__int64)&v51, *(_DWORD *)(v17 + 8), 0);
    v26 = *(_QWORD *)(a1 + 24);
    if ( *(_BYTE *)(v26 + 80) )
    {
      v55 = &unk_4A16D38;
      LODWORD(v56) = v52;
      v58 = v52;
      if ( v52 > 0x40 )
      {
        v43 = v26;
        sub_C43780((__int64)&v57, &v51);
        v26 = v43;
      }
      else
      {
        v57 = v51;
      }
      v60 = v54;
      if ( v54 > 0x40 )
      {
        v41 = v26;
        sub_C43780((__int64)&v59, &v53);
        v26 = v41;
      }
      else
      {
        v59 = v53;
      }
      v36 = v26;
      sub_AADB10((__int64)&v61, v52, 1);
      v27 = v36;
      v28 = *(_DWORD *)(v36 + 24) <= 0x40u;
      *(_DWORD *)(v36 + 8) = (_DWORD)v56;
      if ( !v28 )
      {
        v29 = *(_QWORD *)(v36 + 16);
        if ( v29 )
        {
          j_j___libc_free_0_0(v29);
          v27 = v36;
        }
      }
      *(_QWORD *)(v27 + 16) = v57;
      *(_DWORD *)(v27 + 24) = v58;
      v58 = 0;
      if ( *(_DWORD *)(v27 + 40) > 0x40u )
      {
        v30 = *(_QWORD *)(v27 + 32);
        if ( v30 )
        {
          v37 = v27;
          j_j___libc_free_0_0(v30);
          v27 = v37;
        }
      }
      *(_QWORD *)(v27 + 32) = v59;
      *(_DWORD *)(v27 + 40) = v60;
      v60 = 0;
      if ( *(_DWORD *)(v27 + 56) > 0x40u )
      {
        v31 = *(_QWORD *)(v27 + 48);
        if ( v31 )
        {
          v38 = v27;
          j_j___libc_free_0_0(v31);
          v27 = v38;
        }
      }
      *(_QWORD *)(v27 + 48) = v61;
      *(_DWORD *)(v27 + 56) = v62;
      v62 = 0;
      if ( *(_DWORD *)(v27 + 72) > 0x40u )
      {
        v32 = *(_QWORD *)(v27 + 64);
        if ( v32 )
        {
          v39 = v27;
          j_j___libc_free_0_0(v32);
          v27 = v39;
        }
      }
      *(_QWORD *)(v27 + 64) = v63;
      *(_DWORD *)(v27 + 72) = v64;
      v64 = 0;
      sub_253FFA0((__int64)&v55);
    }
    else
    {
      *(_QWORD *)v26 = &unk_4A16D38;
      v33 = v52;
      *(_DWORD *)(v26 + 8) = v52;
      *(_DWORD *)(v26 + 24) = v33;
      if ( v33 > 0x40 )
      {
        v44 = v26;
        sub_C43780(v26 + 16, &v51);
        v26 = v44;
      }
      else
      {
        *(_QWORD *)(v26 + 16) = v51;
      }
      v34 = v54;
      *(_DWORD *)(v26 + 40) = v54;
      if ( v34 > 0x40 )
      {
        v42 = v26;
        sub_C43780(v26 + 32, &v53);
        v26 = v42;
      }
      else
      {
        *(_QWORD *)(v26 + 32) = v53;
      }
      v40 = v26;
      sub_AADB10(v26 + 48, v52, 1);
      *(_BYTE *)(v40 + 80) = 1;
    }
    sub_969240((__int64 *)&v53);
    sub_969240((__int64 *)&v51);
    v18 = *(_QWORD *)(a1 + 24);
  }
  sub_254FA20((__int64)&v55, v18, v17);
  sub_253FFA0((__int64)&v55);
  v19 = *(_DWORD **)(a1 + 24);
  v20 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v19 + 16LL);
  if ( v20 == sub_2535A50 )
  {
    if ( v19[2] )
    {
      LOBYTE(v21) = sub_AAF760((__int64)(v19 + 4));
      v11 = v21 ^ 1;
      goto LABEL_7;
    }
LABEL_6:
    v11 = 0;
    goto LABEL_7;
  }
  v11 = ((__int64 (*)(void))v20)();
LABEL_7:
  if ( v48 != &v50 )
    _libc_free((unsigned __int64)v48);
  return v11;
}
