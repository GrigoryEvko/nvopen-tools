// Function: sub_24E8700
// Address: 0x24e8700
//
__int64 __fastcall sub_24E8700(__int64 a1)
{
  int v2; // eax
  __int64 result; // rax
  _QWORD *v4; // rax
  __int64 v5; // rcx
  __int64 v6; // r14
  unsigned int v7; // eax
  _QWORD *v8; // r13
  __int64 v9; // r14
  unsigned __int64 v10; // rsi
  int v11; // eax
  __int64 v12; // rsi
  __int64 v13; // rax
  __int64 v14; // rdx
  int v15; // edx
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rdi
  unsigned __int8 *v20; // r9
  __int64 v21; // r8
  __int64 (__fastcall *v22)(__int64, unsigned int, _BYTE *, unsigned __int8 *); // rax
  __int64 v23; // rax
  __int64 v24; // r10
  _QWORD *v25; // rax
  _QWORD *v26; // r9
  __int64 v27; // rsi
  unsigned __int64 v28; // rdi
  int v29; // eax
  _QWORD *v30; // rdi
  _QWORD *v31; // rax
  __int64 v32; // r14
  __int64 v33; // rbx
  __int64 v34; // r12
  __int64 v35; // rdx
  unsigned int v36; // esi
  _QWORD *v37; // rax
  unsigned __int8 *v38; // r9
  _QWORD *v39; // r10
  __int64 v40; // rdx
  __int64 *v41; // rax
  __int64 v42; // rax
  __int64 v43; // r8
  __int64 v44; // r9
  __int64 v45; // r10
  __int64 v46; // rax
  _QWORD *v47; // [rsp+0h] [rbp-D0h]
  unsigned __int8 *v48; // [rsp+8h] [rbp-C8h]
  __int64 v49; // [rsp+8h] [rbp-C8h]
  unsigned __int8 *v50; // [rsp+8h] [rbp-C8h]
  unsigned __int8 *v51; // [rsp+8h] [rbp-C8h]
  __int64 v52; // [rsp+10h] [rbp-C0h]
  __int64 v53; // [rsp+10h] [rbp-C0h]
  __int64 v54; // [rsp+10h] [rbp-C0h]
  __int64 v55; // [rsp+10h] [rbp-C0h]
  __int64 v56; // [rsp+28h] [rbp-A8h]
  _QWORD *v57; // [rsp+28h] [rbp-A8h]
  __int64 v58; // [rsp+28h] [rbp-A8h]
  __int64 v59; // [rsp+38h] [rbp-98h]
  char v60[32]; // [rsp+40h] [rbp-90h] BYREF
  __int16 v61; // [rsp+60h] [rbp-70h]
  __int64 v62[4]; // [rsp+70h] [rbp-60h] BYREF
  __int16 v63; // [rsp+90h] [rbp-40h]

  v2 = *(_DWORD *)(a1 + 32);
  if ( v2 > 2 )
  {
    if ( (unsigned int)(v2 - 3) > 1 )
      goto LABEL_48;
    goto LABEL_29;
  }
  if ( v2 <= 0 )
  {
    if ( v2 )
      goto LABEL_48;
LABEL_29:
    result = *(_QWORD *)(a1 + 24);
    goto LABEL_4;
  }
  result = *(_QWORD *)(a1 + 24);
  if ( *(_BYTE *)(result + 365) )
    return result;
LABEL_4:
  v62[0] = *(_QWORD *)(result + 328);
  v4 = sub_24E84F0(a1 + 200, v62);
  v5 = 32;
  v6 = v4[2];
  v7 = (*(_DWORD *)(v6 + 4) & 0x7FFFFFFu) >> 1;
  if ( v7 )
    v5 = 32LL * (2 * (v7 - 1) + 1);
  v56 = *(_QWORD *)(*(_QWORD *)(v6 - 8) + v5);
  sub_B53C80(v6, v6, v7 - 2);
  result = *(unsigned int *)(a1 + 32);
  if ( (int)result > 2 )
  {
    result = (unsigned int)(result - 3);
    if ( (unsigned int)result <= 1 )
      return result;
LABEL_48:
    BUG();
  }
  if ( (int)result <= 0 )
  {
    if ( !(_DWORD)result )
      return result;
    goto LABEL_48;
  }
  v8 = *(_QWORD **)(v6 + 40);
  v62[0] = (__int64)"Switch";
  v63 = 259;
  v9 = sub_AA8550(v8, (__int64 *)(v6 + 24), 0, (__int64)v62, 0);
  v10 = v8[6] & 0xFFFFFFFFFFFFFFF8LL;
  if ( v8 + 6 == (_QWORD *)v10 )
  {
    v12 = 0;
  }
  else
  {
    if ( !v10 )
      goto LABEL_48;
    v11 = *(unsigned __int8 *)(v10 - 24);
    v12 = v10 - 24;
    if ( (unsigned int)(v11 - 30) >= 0xB )
      v12 = 0;
  }
  sub_D5F1F0(a1 + 40, v12);
  if ( !(unsigned __int8)sub_B2D610(*(_QWORD *)(a1 + 280), 7) )
  {
    v62[0] = (__int64)"ResumeFn.addr";
    v13 = *(_QWORD *)(a1 + 24);
    v14 = *(_QWORD *)(a1 + 288);
    v63 = 259;
    v15 = sub_9213A0((unsigned int **)(a1 + 40), *(_QWORD *)(v13 + 288), v14, 0, 0, (__int64)v62, 7u);
    v16 = *(_QWORD *)(a1 + 24);
    v63 = 257;
    v17 = **(_QWORD **)(*(_QWORD *)(v16 + 288) + 16LL);
    v61 = 257;
    v52 = sub_A82CA0((unsigned int **)(a1 + 40), v17, v15, 0, 0, (__int64)v62);
    v18 = sub_AD6530(*(_QWORD *)(v52 + 8), v17);
    v19 = *(_QWORD *)(a1 + 120);
    v20 = (unsigned __int8 *)v18;
    v21 = v52;
    v22 = *(__int64 (__fastcall **)(__int64, unsigned int, _BYTE *, unsigned __int8 *))(*(_QWORD *)v19 + 56LL);
    if ( v22 == sub_928890 )
    {
      if ( *(_BYTE *)v52 > 0x15u || *v20 > 0x15u )
        goto LABEL_39;
      v48 = v20;
      v23 = sub_AAB310(0x20u, (unsigned __int8 *)v52, v20);
      v21 = v52;
      v20 = v48;
      v24 = v23;
    }
    else
    {
      v51 = v20;
      v46 = v22(v19, 32u, (_BYTE *)v52, v20);
      v20 = v51;
      v21 = v52;
      v24 = v46;
    }
    if ( v24 )
    {
LABEL_18:
      v63 = 257;
      v49 = v24;
      v25 = sub_BD2C40(72, 3u);
      v26 = v25;
      if ( v25 )
      {
        v27 = v56;
        v57 = v25;
        sub_B4C9A0((__int64)v25, v27, v9, v49, 3u, (__int64)v25, 0, 0);
        v26 = v57;
      }
      v58 = (__int64)v26;
      (*(void (__fastcall **)(_QWORD, _QWORD *, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 128) + 16LL))(
        *(_QWORD *)(a1 + 128),
        v26,
        v62,
        *(_QWORD *)(a1 + 96),
        *(_QWORD *)(a1 + 104));
      sub_94AAF0((unsigned int **)(a1 + 40), v58);
      goto LABEL_21;
    }
LABEL_39:
    v50 = v20;
    v63 = 257;
    v53 = v21;
    v37 = sub_BD2C40(72, unk_3F10FD0);
    v38 = v50;
    v39 = v37;
    if ( v37 )
    {
      v40 = *(_QWORD *)(v53 + 8);
      v47 = v37;
      if ( (unsigned int)*(unsigned __int8 *)(v40 + 8) - 17 > 1 )
      {
        v42 = sub_BCB2A0(*(_QWORD **)v40);
        v45 = (__int64)v47;
        v44 = (__int64)v50;
        v43 = v53;
      }
      else
      {
        BYTE4(v59) = *(_BYTE *)(v40 + 8) == 18;
        LODWORD(v59) = *(_DWORD *)(v40 + 32);
        v41 = (__int64 *)sub_BCB2A0(*(_QWORD **)v40);
        v42 = sub_BCE1B0(v41, v59);
        v43 = v53;
        v44 = (__int64)v50;
        v45 = (__int64)v47;
      }
      v54 = v45;
      sub_B523C0(v45, v42, 53, 32, v43, v44, (__int64)v62, 0, 0, 0);
      v39 = (_QWORD *)v54;
    }
    v55 = (__int64)v39;
    (*(void (__fastcall **)(_QWORD, _QWORD *, char *, _QWORD, _QWORD, unsigned __int8 *))(**(_QWORD **)(a1 + 128) + 16LL))(
      *(_QWORD *)(a1 + 128),
      v39,
      v60,
      *(_QWORD *)(a1 + 96),
      *(_QWORD *)(a1 + 104),
      v38);
    sub_94AAF0((unsigned int **)(a1 + 40), v55);
    v24 = v55;
    goto LABEL_18;
  }
  v63 = 257;
  v31 = sub_BD2C40(72, 1u);
  v32 = (__int64)v31;
  if ( v31 )
    sub_B4C8F0((__int64)v31, v56, 1u, 0, 0);
  (*(void (__fastcall **)(_QWORD, __int64, __int64 *, _QWORD, _QWORD))(**(_QWORD **)(a1 + 128) + 16LL))(
    *(_QWORD *)(a1 + 128),
    v32,
    v62,
    *(_QWORD *)(a1 + 96),
    *(_QWORD *)(a1 + 104));
  v33 = *(_QWORD *)(a1 + 40);
  v34 = v33 + 16LL * *(unsigned int *)(a1 + 48);
  while ( v34 != v33 )
  {
    v35 = *(_QWORD *)(v33 + 8);
    v36 = *(_DWORD *)v33;
    v33 += 16;
    sub_B99FD0(v32, v36, v35);
  }
LABEL_21:
  v28 = v8[6] & 0xFFFFFFFFFFFFFFF8LL;
  if ( v8 + 6 == (_QWORD *)v28 )
    return sub_B43D60(0);
  if ( !v28 )
    goto LABEL_48;
  v29 = *(unsigned __int8 *)(v28 - 24);
  v30 = (_QWORD *)(v28 - 24);
  if ( (unsigned int)(v29 - 30) >= 0xB )
    v30 = 0;
  return sub_B43D60(v30);
}
