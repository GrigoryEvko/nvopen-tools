// Function: sub_27EB050
// Address: 0x27eb050
//
__int64 __fastcall sub_27EB050(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  _QWORD *v8; // r14
  __int64 v9; // rbx
  _BYTE *v10; // r10
  size_t v11; // r8
  _QWORD *v12; // rax
  __int64 v13; // r14
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rsi
  char v17; // al
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  __int64 v22; // r13
  __int64 v23; // rdx
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  _QWORD *v27; // rbx
  _QWORD *v28; // r14
  void (__fastcall *v29)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v30; // rax
  __int64 v31; // r13
  __int64 v32; // rdx
  __int64 v33; // rcx
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  __int64 v41; // rax
  _QWORD *v42; // rdi
  __int64 v43; // [rsp+0h] [rbp-B0h]
  size_t n; // [rsp+8h] [rbp-A8h]
  size_t na; // [rsp+8h] [rbp-A8h]
  void *src; // [rsp+10h] [rbp-A0h]
  char srca; // [rsp+10h] [rbp-A0h]
  _BYTE *srcb; // [rsp+10h] [rbp-A0h]
  __int64 v50; // [rsp+28h] [rbp-88h] BYREF
  __int128 v51; // [rsp+30h] [rbp-80h] BYREF
  __int128 v52; // [rsp+40h] [rbp-70h] BYREF
  _QWORD v53[2]; // [rsp+50h] [rbp-60h] BYREF
  __int64 v54; // [rsp+60h] [rbp-50h]
  __int64 v55; // [rsp+68h] [rbp-48h]
  __int64 v56; // [rsp+70h] [rbp-40h]

  v7 = sub_BC1CD0(a4, &unk_4F89C30, a3);
  v8 = *(_QWORD **)(a3 + 40);
  *(_QWORD *)&v52 = v53;
  v9 = v7 + 8;
  v10 = (_BYTE *)v8[29];
  v11 = v8[30];
  if ( &v10[v11] && !v10 )
    sub_426248((__int64)"basic_string::_M_construct null not valid");
  *(_QWORD *)&v51 = v8[30];
  if ( v11 > 0xF )
  {
    na = v11;
    srcb = v10;
    v41 = sub_22409D0((__int64)&v52, (unsigned __int64 *)&v51, 0);
    v10 = srcb;
    v11 = na;
    *(_QWORD *)&v52 = v41;
    v42 = (_QWORD *)v41;
    v53[0] = v51;
  }
  else
  {
    if ( v11 == 1 )
    {
      LOBYTE(v53[0]) = *v10;
      v12 = v53;
      goto LABEL_6;
    }
    if ( !v11 )
    {
      v12 = v53;
      goto LABEL_6;
    }
    v42 = v53;
  }
  memcpy(v42, v10, v11);
  v11 = v51;
  v12 = (_QWORD *)v52;
LABEL_6:
  *((_QWORD *)&v52 + 1) = v11;
  *((_BYTE *)v12 + v11) = 0;
  v54 = v8[33];
  v55 = v8[34];
  v56 = v8[35];
  if ( (unsigned int)(v54 - 42) > 1 )
  {
    if ( (_QWORD *)v52 != v53 )
      j_j___libc_free_0(v52);
    if ( (unsigned __int8)sub_DF9710(v9) )
      goto LABEL_35;
  }
  else if ( (_QWORD *)v52 != v53 )
  {
    j_j___libc_free_0(v52);
  }
  v13 = sub_BC1CD0(a4, &unk_4F6D3F8, a3) + 8;
  v43 = sub_BC1CD0(a4, &unk_4FDBCC8, a3) + 8;
  n = sub_BC1CD0(a4, &unk_4F86540, a3) + 8;
  v14 = sub_BC1CD0(a4, &unk_4F81450, a3);
  BYTE8(v52) = 0;
  BYTE8(v51) = 0;
  src = (void *)(v14 + 8);
  v15 = sub_22077B0(0x2B8u);
  if ( v15 )
  {
    *(_QWORD *)(v15 + 528) = 0;
    *(_QWORD *)v15 = v15 + 16;
    *(_QWORD *)(v15 + 576) = v15 + 600;
    *(_QWORD *)(v15 + 8) = 0x1000000000LL;
    *(_QWORD *)(v15 + 536) = 0;
    *(_QWORD *)(v15 + 544) = src;
    *(_QWORD *)(v15 + 552) = 0;
    *(_BYTE *)(v15 + 560) = 1;
    *(_QWORD *)(v15 + 568) = 0;
    *(_QWORD *)(v15 + 584) = 8;
    *(_DWORD *)(v15 + 592) = 0;
    *(_BYTE *)(v15 + 596) = 1;
    *(_WORD *)(v15 + 664) = 0;
    *(_QWORD *)(v15 + 672) = 0;
    *(_QWORD *)(v15 + 680) = 0;
    *(_QWORD *)(v15 + 688) = 0;
  }
  v50 = v15;
  v16 = a3;
  v17 = sub_27EA5F0(a2, a3, a4, v13, v9, v43, n, &v50, v51, v52);
  v22 = v50;
  srca = v17;
  if ( v50 )
  {
    sub_FFCE90(v50, v16, v18, v19, v20, v21);
    sub_FFD870(v22, v16, v23, v24, v25, v26);
    sub_FFBC40(v22, v16);
    v27 = *(_QWORD **)(v22 + 680);
    v28 = *(_QWORD **)(v22 + 672);
    if ( v27 != v28 )
    {
      do
      {
        v29 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v28[7];
        *v28 = &unk_49E5048;
        if ( v29 )
          v29(v28 + 5, v28 + 5, 3);
        *v28 = &unk_49DB368;
        v30 = v28[3];
        if ( v30 != 0 && v30 != -4096 && v30 != -8192 )
          sub_BD60C0(v28 + 1);
        v28 += 9;
      }
      while ( v27 != v28 );
      v28 = *(_QWORD **)(v22 + 672);
    }
    if ( v28 )
      j_j___libc_free_0((unsigned __int64)v28);
    if ( !*(_BYTE *)(v22 + 596) )
      _libc_free(*(_QWORD *)(v22 + 576));
    if ( *(_QWORD *)v22 != v22 + 16 )
      _libc_free(*(_QWORD *)v22);
    v16 = 696;
    j_j___libc_free_0(v22);
  }
  if ( !srca )
  {
LABEL_35:
    *(_BYTE *)(a1 + 76) = 1;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  v31 = *(_QWORD *)(a2 + 48);
  sub_FFCE90(v31, v16, v18, v19, v20, v21);
  sub_FFD870(v31, v16, v32, v33, v34, v35);
  sub_FFBC40(v31, v16);
  sub_27DD0B0(a1, a2, v36, v37, v38, v39);
  return a1;
}
