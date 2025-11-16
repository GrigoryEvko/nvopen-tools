// Function: sub_17CD290
// Address: 0x17cd290
//
__int64 __fastcall sub_17CD290(__int64 a1, _QWORD **a2)
{
  __int64 v4; // r13
  unsigned __int64 v5; // rsi
  _QWORD *v6; // rax
  _DWORD *v7; // rdi
  __int64 v8; // rcx
  __int64 v9; // rdx
  unsigned __int64 v10; // rdx
  _QWORD *v11; // rax
  bool v12; // r15
  _DWORD *v13; // rdi
  __int64 v14; // rsi
  __int64 v15; // rcx
  __int64 v16; // rax
  _DWORD *v17; // r8
  _DWORD *v18; // rdi
  __int64 v19; // rcx
  __int64 v20; // rdx
  _QWORD *v21; // rsi
  __int64 v22; // rax
  _QWORD *v23; // rdi
  int v24; // r9d
  __int64 v25; // rax
  bool v26; // zf
  __int64 v28; // rax
  _DWORD *v29; // r8
  _DWORD *v30; // rdi
  __int64 v31; // rcx
  __int64 v32; // rdx
  __int64 v33; // r13
  __int64 v34; // rax
  _QWORD *v35; // rdi
  __int64 v36; // rax
  __int64 v37; // rbx
  __int64 v38; // r13
  __int64 v39; // rax
  __int64 v40; // rbx
  _QWORD *v41; // rdi
  __int64 v42; // [rsp+8h] [rbp-A8h]
  __int64 v43; // [rsp+8h] [rbp-A8h]
  _QWORD v44[2]; // [rsp+10h] [rbp-A0h] BYREF
  __int16 v45; // [rsp+20h] [rbp-90h]
  __int64 v46[2]; // [rsp+30h] [rbp-80h] BYREF
  __int64 v47; // [rsp+40h] [rbp-70h] BYREF
  _QWORD *v48; // [rsp+48h] [rbp-68h]
  __int64 v49; // [rsp+50h] [rbp-60h]
  int v50; // [rsp+58h] [rbp-58h]
  int v51; // [rsp+5Ch] [rbp-54h]
  __int64 v52; // [rsp+60h] [rbp-50h]
  __int64 v53; // [rsp+68h] [rbp-48h]

  v4 = sub_1632FA0((__int64)a2);
  v5 = sub_16D5D50();
  v6 = *(_QWORD **)&dword_4FA0208[2];
  if ( !*(_QWORD *)&dword_4FA0208[2] )
    goto LABEL_8;
  v7 = dword_4FA0208;
  do
  {
    while ( 1 )
    {
      v8 = v6[2];
      v9 = v6[3];
      if ( v5 <= v6[4] )
        break;
      v6 = (_QWORD *)v6[3];
      if ( !v9 )
        goto LABEL_6;
    }
    v7 = v6;
    v6 = (_QWORD *)v6[2];
  }
  while ( v8 );
LABEL_6:
  if ( v7 == dword_4FA0208 )
    goto LABEL_8;
  if ( v5 < *((_QWORD *)v7 + 4) )
    goto LABEL_8;
  v28 = *((_QWORD *)v7 + 7);
  v29 = v7 + 12;
  if ( !v28 )
    goto LABEL_8;
  v30 = v7 + 12;
  do
  {
    while ( 1 )
    {
      v31 = *(_QWORD *)(v28 + 16);
      v32 = *(_QWORD *)(v28 + 24);
      if ( *(_DWORD *)(v28 + 32) >= dword_4FA3F48 )
        break;
      v28 = *(_QWORD *)(v28 + 24);
      if ( !v32 )
        goto LABEL_34;
    }
    v30 = (_DWORD *)v28;
    v28 = *(_QWORD *)(v28 + 16);
  }
  while ( v31 );
LABEL_34:
  if ( v29 == v30 || dword_4FA3F48 < v30[8] )
  {
LABEL_8:
    v10 = sub_16D5D50();
    v11 = *(_QWORD **)&dword_4FA0208[2];
    if ( !*(_QWORD *)&dword_4FA0208[2] )
      goto LABEL_39;
    v12 = 0;
  }
  else
  {
    v12 = v30[9] > 0;
    v10 = sub_16D5D50();
    v11 = *(_QWORD **)&dword_4FA0208[2];
    if ( !*(_QWORD *)&dword_4FA0208[2] )
      goto LABEL_23;
  }
  v13 = dword_4FA0208;
  do
  {
    while ( 1 )
    {
      v14 = v11[2];
      v15 = v11[3];
      if ( v10 <= v11[4] )
        break;
      v11 = (_QWORD *)v11[3];
      if ( !v15 )
        goto LABEL_14;
    }
    v13 = v11;
    v11 = (_QWORD *)v11[2];
  }
  while ( v14 );
LABEL_14:
  if ( v13 != dword_4FA0208 && v10 >= *((_QWORD *)v13 + 4) )
  {
    v16 = *((_QWORD *)v13 + 7);
    v17 = v13 + 12;
    if ( v16 )
    {
      v18 = v13 + 12;
      do
      {
        while ( 1 )
        {
          v19 = *(_QWORD *)(v16 + 16);
          v20 = *(_QWORD *)(v16 + 24);
          if ( *(_DWORD *)(v16 + 32) >= dword_4FA3E68 )
            break;
          v16 = *(_QWORD *)(v16 + 24);
          if ( !v20 )
            goto LABEL_21;
        }
        v18 = (_DWORD *)v16;
        v16 = *(_QWORD *)(v16 + 16);
      }
      while ( v19 );
LABEL_21:
      if ( v17 != v18 && dword_4FA3E68 >= v18[8] )
      {
        if ( v18[9] > 0 || v12 )
          goto LABEL_24;
        goto LABEL_39;
      }
    }
  }
LABEL_23:
  if ( v12 )
  {
LABEL_24:
    *(_QWORD *)(a1 + 384) = qword_4FA41A0;
    *(_QWORD *)(a1 + 392) = qword_4FA40C0;
    *(_QWORD *)(a1 + 400) = qword_4FA3FE0;
    *(_QWORD *)(a1 + 408) = qword_4FA3F00;
    *(_QWORD *)(a1 + 376) = a1 + 384;
    goto LABEL_25;
  }
LABEL_39:
  v45 = 260;
  v44[0] = a2 + 30;
  sub_16E1010((__int64)v46, (__int64)v44);
  if ( v51 == 9 )
  {
    switch ( (int)v49 )
    {
      case 3:
      case 4:
        *(_QWORD *)(a1 + 376) = &unk_42B5FE0;
        goto LABEL_50;
      case 12:
      case 13:
        *(_QWORD *)(a1 + 376) = &unk_42B6020;
        goto LABEL_50;
      case 17:
      case 18:
        *(_QWORD *)(a1 + 376) = &unk_42B6000;
        goto LABEL_50;
      case 31:
        *(_QWORD *)(a1 + 376) = &unk_42B6060;
        goto LABEL_50;
      case 32:
        *(_QWORD *)(a1 + 376) = &unk_42B6040;
        goto LABEL_50;
      default:
        goto LABEL_57;
    }
  }
  if ( v51 == 12 )
  {
    if ( (_DWORD)v49 == 32 )
    {
      *(_QWORD *)(a1 + 376) = &unk_42B5F80;
      goto LABEL_50;
    }
LABEL_57:
    sub_16BD130("unsupported architecture", 1u);
  }
  if ( v51 != 5 )
    sub_16BD130("unsupported operating system", 1u);
  if ( (_DWORD)v49 == 31 )
  {
    *(_QWORD *)(a1 + 376) = &unk_42B5FC0;
    goto LABEL_50;
  }
  if ( (_DWORD)v49 != 32 )
    goto LABEL_57;
  *(_QWORD *)(a1 + 376) = &unk_42B5FA0;
LABEL_50:
  if ( (__int64 *)v46[0] != &v47 )
    j_j___libc_free_0(v46[0], v47 + 1);
LABEL_25:
  v21 = *a2;
  v50 = 0;
  v46[0] = 0;
  *(_QWORD *)(a1 + 168) = v21;
  v48 = v21;
  v47 = 0;
  v49 = 0;
  v52 = 0;
  v53 = 0;
  v46[1] = 0;
  v22 = sub_15A9620(v4, (__int64)v21, 0);
  v23 = v48;
  *(_QWORD *)(a1 + 176) = v22;
  *(_QWORD *)(a1 + 184) = sub_1643350(v23);
  v44[0] = *(_QWORD *)(a1 + 168);
  *(_QWORD *)(a1 + 416) = sub_161BE60(v44, 1u, 0x3E8u);
  v44[0] = *(_QWORD *)(a1 + 168);
  *(_QWORD *)(a1 + 424) = sub_161BE60(v44, 1u, 0x3E8u);
  v25 = sub_1B281E0(
          (_DWORD)a2,
          (unsigned int)"msan.module_ctor",
          16,
          (unsigned int)"__msan_init",
          11,
          v24,
          0,
          0,
          0,
          0,
          0,
          0);
  v26 = byte_4FA4280 == 0;
  *(_QWORD *)(a1 + 440) = v25;
  if ( v26 )
  {
    sub_1B28000(a2, v25, 0, 0);
    if ( !*(_DWORD *)(a1 + 156) )
      goto LABEL_27;
  }
  else
  {
    *(_QWORD *)(*(_QWORD *)(a1 + 440) + 48LL) = sub_1633B90((__int64)a2, "msan.module_ctor", 0x10u);
    sub_1B28000(a2, *(_QWORD *)(a1 + 440), 0, *(_QWORD *)(a1 + 440));
    if ( !*(_DWORD *)(a1 + 156) )
      goto LABEL_27;
  }
  v33 = sub_1643350(v48);
  v42 = *(unsigned int *)(a1 + 156);
  v34 = sub_1643350(v48);
  v43 = sub_159C470(v34, v42, 0);
  v44[0] = "__msan_track_origins";
  v45 = 259;
  v35 = sub_1648A60(88, 1u);
  if ( !v35 )
  {
LABEL_27:
    if ( !*(_BYTE *)(a1 + 160) )
      goto LABEL_28;
    goto LABEL_46;
  }
  sub_15E51E0((__int64)v35, (__int64)a2, v33, 1, 5, v43, (__int64)v44, 0, 0, 0, 0);
  if ( !*(_BYTE *)(a1 + 160) )
    goto LABEL_28;
LABEL_46:
  v36 = sub_1643350(v48);
  v37 = *(unsigned __int8 *)(a1 + 160);
  v38 = v36;
  v39 = sub_1643350(v48);
  v40 = sub_159C470(v39, v37, 0);
  v45 = 259;
  v44[0] = "__msan_keep_going";
  v41 = sub_1648A60(88, 1u);
  if ( v41 )
    sub_15E51E0((__int64)v41, (__int64)a2, v38, 1, 5, v40, (__int64)v44, 0, 0, 0, 0);
LABEL_28:
  sub_17CD270(v46);
  return 1;
}
