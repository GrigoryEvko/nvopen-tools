// Function: sub_34CFD30
// Address: 0x34cfd30
//
__int64 __fastcall sub_34CFD30(__int64 a1, __int64 a2, _DWORD *a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r12
  _BYTE *v6; // rax
  __int64 v7; // rdx
  bool (__fastcall *v8)(__int64, __int64); // rax
  __int64 v9; // rdx
  unsigned int v10; // eax
  unsigned __int64 *v11; // rdx
  __int64 v12; // rbx
  __int64 v13; // r13
  __int64 v14; // r12
  _DWORD *v15; // rax
  __int64 v16; // r8
  __int64 v17; // r9
  __int64 v18; // rcx
  int v19; // eax
  unsigned int v20; // eax
  __int64 v21; // r13
  char v22; // di
  __int64 v23; // rbx
  __int64 v24; // rax
  __int64 *v25; // rdx
  __int64 v26; // rsi
  __int64 *v27; // rax
  int v28; // ebx
  unsigned __int64 v29; // r13
  unsigned int v30; // r14d
  unsigned __int64 *v31; // r15
  unsigned int v32; // ebx
  __int64 *v33; // r13
  __int64 v34; // rbx
  bool v35; // al
  unsigned __int64 v37; // r14
  unsigned __int64 *v38; // rdx
  unsigned __int64 v39; // rdx
  _BYTE *v43; // [rsp+18h] [rbp-D8h]
  __int64 v44; // [rsp+20h] [rbp-D0h]
  char v45; // [rsp+2Bh] [rbp-C5h]
  unsigned int v46; // [rsp+2Ch] [rbp-C4h]
  __int64 v47; // [rsp+30h] [rbp-C0h]
  unsigned __int64 *v48; // [rsp+40h] [rbp-B0h] BYREF
  unsigned int v49; // [rsp+48h] [rbp-A8h]
  unsigned __int64 v50; // [rsp+50h] [rbp-A0h] BYREF
  unsigned int v51; // [rsp+58h] [rbp-98h]
  unsigned __int64 *v52; // [rsp+60h] [rbp-90h] BYREF
  unsigned int v53; // [rsp+68h] [rbp-88h]
  unsigned __int64 *v54; // [rsp+70h] [rbp-80h] BYREF
  unsigned int v55; // [rsp+78h] [rbp-78h]
  char *v56; // [rsp+80h] [rbp-70h] BYREF
  __int64 *v57; // [rsp+88h] [rbp-68h]
  __int64 v58; // [rsp+90h] [rbp-60h]
  int v59; // [rsp+98h] [rbp-58h]
  char v60; // [rsp+9Ch] [rbp-54h]
  char v61; // [rsp+A0h] [rbp-50h] BYREF

  v5 = a2;
  v44 = *(_QWORD *)(a1 + 8);
  v46 = ((*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) >> 1) - 1;
  v6 = *(_BYTE **)(a1 + 24);
  *a3 = 0;
  v7 = *(_QWORD *)(a2 + 40);
  v43 = v6;
  v8 = *(bool (__fastcall **)(__int64, __int64))(*(_QWORD *)v6 + 664LL);
  if ( v8 != sub_2FE3D70 )
  {
    v35 = v8((__int64)v43, *(_QWORD *)(v7 + 72));
    if ( !v46 )
      return 0;
    if ( !v35 )
      goto LABEL_55;
    goto LABEL_5;
  }
  v56 = (char *)sub_B2D7E0(*(_QWORD *)(v7 + 72), "no-jump-tables", 0xEu);
  if ( !(unsigned __int8)sub_A72A30((__int64 *)&v56) && ((v43[7217] & 0xFB) == 0 || (v43[7216] & 0xFB) == 0) )
  {
    if ( !v46 )
      return 0;
LABEL_5:
    v45 = 1;
    goto LABEL_6;
  }
  if ( !v46 )
    return 0;
LABEL_55:
  if ( sub_AE2980(v44, 0)[3] < v46 )
    return v46;
  v45 = 0;
LABEL_6:
  v9 = *(_QWORD *)(*(_QWORD *)(a2 - 8) + 64LL);
  v10 = *(_DWORD *)(v9 + 32);
  v49 = v10;
  if ( v10 <= 0x40 )
  {
    v11 = *(unsigned __int64 **)(v9 + 24);
    v51 = v10;
    v48 = v11;
LABEL_8:
    v50 = (unsigned __int64)v48;
    goto LABEL_9;
  }
  sub_C43780((__int64)&v48, (const void **)(v9 + 24));
  v51 = v49;
  if ( v49 <= 0x40 )
    goto LABEL_8;
  sub_C43780((__int64)&v50, (const void **)&v48);
LABEL_9:
  v12 = 0;
  v47 = ((*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) >> 1) - 1;
  if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFFu) >> 1 == 1 )
    goto LABEL_21;
  do
  {
    while ( 1 )
    {
      v13 = *(_QWORD *)(*(_QWORD *)(a2 - 8) + 32LL * (unsigned int)(2 * ++v12));
      v14 = v13 + 24;
      if ( (int)sub_C4C880(v13 + 24, (__int64)&v48) <= 0 )
        goto LABEL_11;
      if ( v49 <= 0x40 && *(_DWORD *)(v13 + 32) <= 0x40u )
        break;
      sub_C43990((__int64)&v48, v14);
      if ( (int)sub_C4C880(v14, (__int64)&v50) < 0 )
        goto LABEL_17;
LABEL_12:
      if ( v47 == v12 )
        goto LABEL_20;
    }
    v38 = *(unsigned __int64 **)(v13 + 24);
    v49 = *(_DWORD *)(v13 + 32);
    v48 = v38;
LABEL_11:
    if ( (int)sub_C4C880(v14, (__int64)&v50) >= 0 )
      goto LABEL_12;
LABEL_17:
    if ( v51 <= 0x40 && *(_DWORD *)(v13 + 32) <= 0x40u )
    {
      v39 = *(_QWORD *)(v13 + 24);
      v51 = *(_DWORD *)(v13 + 32);
      v50 = v39;
      goto LABEL_12;
    }
    sub_C43990((__int64)&v50, v14);
  }
  while ( v47 != v12 );
LABEL_20:
  v5 = a2;
LABEL_21:
  v15 = sub_AE2980(v44, 0);
  v18 = v46;
  if ( v46 <= v15[3] )
  {
    v56 = 0;
    v57 = (__int64 *)&v61;
    v19 = *(_DWORD *)(v5 + 4);
    v58 = 4;
    v59 = 0;
    v20 = (v19 & 0x7FFFFFFu) >> 1;
    v60 = 1;
    v21 = v20 - 1;
    if ( v20 != 1 )
    {
      v22 = 1;
      v23 = 0;
      while ( 1 )
      {
        v24 = 4;
        if ( (_DWORD)v23 != -2 )
          v24 = 4LL * (unsigned int)(2 * v23 + 3);
        v25 = *(__int64 **)(v5 - 8);
        v26 = v25[v24];
        if ( !v22 )
          goto LABEL_60;
        v27 = v57;
        v18 = HIDWORD(v58);
        v25 = &v57[HIDWORD(v58)];
        if ( v57 == v25 )
        {
LABEL_62:
          if ( HIDWORD(v58) >= (unsigned int)v58 )
          {
LABEL_60:
            ++v23;
            sub_C8CC70((__int64)&v56, v26, (__int64)v25, v18, v16, v17);
            v22 = v60;
            if ( v21 == v23 )
              goto LABEL_32;
          }
          else
          {
            v18 = (unsigned int)(HIDWORD(v58) + 1);
            ++v23;
            ++HIDWORD(v58);
            *v25 = v26;
            v22 = v60;
            ++v56;
            if ( v21 == v23 )
              goto LABEL_32;
          }
        }
        else
        {
          while ( v26 != *v27 )
          {
            if ( v25 == ++v27 )
              goto LABEL_62;
          }
          if ( v21 == ++v23 )
          {
LABEL_32:
            v28 = HIDWORD(v58) - v59;
            goto LABEL_33;
          }
        }
      }
    }
    v28 = 0;
LABEL_33:
    v29 = (unsigned int)sub_AE2980(v44, 0)[3];
    v55 = v49;
    if ( v49 > 0x40 )
      sub_C43780((__int64)&v54, (const void **)&v48);
    else
      v54 = v48;
    sub_C46B40((__int64)&v54, (__int64 *)&v50);
    v30 = v55;
    v31 = v54;
    v55 = 0;
    v53 = v30;
    v52 = v54;
    if ( v30 <= 0x40 )
    {
      if ( v54 == (unsigned __int64 *)-1LL )
      {
LABEL_37:
        if ( !v60 )
          _libc_free((unsigned __int64)v57);
        goto LABEL_39;
      }
      v37 = (unsigned __int64)v54 + 1;
LABEL_71:
      if ( v29 < v37 || (v28 != 1 || v46 <= 2) && (v28 != 2 || v46 <= 4) && (v28 != 3 || v46 <= 5) )
        goto LABEL_37;
      if ( !v60 )
        _libc_free((unsigned __int64)v57);
LABEL_76:
      v46 = 1;
      goto LABEL_77;
    }
    if ( v30 - (unsigned int)sub_C444A0((__int64)&v52) > 0x40 )
    {
      v37 = -1;
      goto LABEL_67;
    }
    v37 = *v31;
    if ( *v31 == -1 )
    {
      j_j___libc_free_0_0((unsigned __int64)v31);
      if ( v55 <= 0x40 )
        goto LABEL_37;
    }
    else
    {
      ++v37;
LABEL_67:
      if ( !v31 )
        goto LABEL_71;
      j_j___libc_free_0_0((unsigned __int64)v31);
      if ( v55 <= 0x40 )
        goto LABEL_71;
    }
    if ( v54 )
      j_j___libc_free_0_0((unsigned __int64)v54);
    goto LABEL_71;
  }
LABEL_39:
  if ( !v45 || v46 == 1 || (*(unsigned int (__fastcall **)(_BYTE *))(*(_QWORD *)v43 + 856LL))(v43) > v46 )
    goto LABEL_77;
  v55 = v49;
  if ( v49 > 0x40 )
    sub_C43780((__int64)&v54, (const void **)&v48);
  else
    v54 = v48;
  sub_C46B40((__int64)&v54, (__int64 *)&v50);
  v32 = v55;
  v33 = (__int64 *)v54;
  v55 = 0;
  LODWORD(v57) = v32;
  v56 = (char *)v54;
  if ( v32 <= 0x40 )
  {
    v34 = (__int64)v54 + 1;
    if ( v54 == (unsigned __int64 *)-1LL )
      v34 = -1;
  }
  else
  {
    if ( v32 - (unsigned int)sub_C444A0((__int64)&v56) > 0x40 )
    {
      v34 = -1;
      goto LABEL_47;
    }
    v34 = *v33;
    if ( *v33 == -1 )
      goto LABEL_48;
    ++v34;
LABEL_47:
    if ( v33 )
    {
LABEL_48:
      j_j___libc_free_0_0((unsigned __int64)v33);
      if ( v55 > 0x40 && v54 )
        j_j___libc_free_0_0((unsigned __int64)v54);
    }
  }
  if ( (*(unsigned __int8 (__fastcall **)(_BYTE *, __int64, _QWORD, __int64, __int64, __int64))(*(_QWORD *)v43 + 672LL))(
         v43,
         v5,
         v46,
         v34,
         a4,
         a5) )
  {
    *a3 = v34;
    goto LABEL_76;
  }
LABEL_77:
  if ( v51 > 0x40 && v50 )
    j_j___libc_free_0_0(v50);
  if ( v49 > 0x40 && v48 )
    j_j___libc_free_0_0((unsigned __int64)v48);
  return v46;
}
