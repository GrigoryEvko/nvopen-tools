// Function: sub_117DEC0
// Address: 0x117dec0
//
_QWORD *__fastcall sub_117DEC0(__int64 a1, __int64 *a2)
{
  __int64 v3; // rax
  __int64 v4; // r15
  __int64 v5; // rdi
  __int64 v6; // r13
  __int64 *v7; // r13
  _QWORD *v8; // r12
  __int64 v10; // rdx
  _BYTE *v11; // rax
  __int64 v12; // rdx
  _BYTE *v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 v16; // rax
  __int64 v17; // rax
  __int64 v18; // rdi
  __int64 v19; // rsi
  int v20; // r9d
  int v21; // r8d
  unsigned __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // r14
  _BYTE *v25; // rax
  __int64 v26; // r15
  __int64 v27; // r15
  __int64 *v28; // rax
  __int64 v29; // rax
  __int64 v30; // r15
  __int64 v31; // r14
  __int64 v32; // rax
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // rbx
  __int64 v36; // r14
  __int64 v37; // rdx
  unsigned int v38; // esi
  bool v39; // [rsp+Fh] [rbp-111h]
  int v40; // [rsp+10h] [rbp-110h]
  _BYTE *v41; // [rsp+18h] [rbp-108h]
  _QWORD *v43; // [rsp+28h] [rbp-F8h]
  __int64 v44; // [rsp+30h] [rbp-F0h]
  char v45; // [rsp+47h] [rbp-D9h] BYREF
  __int64 v46; // [rsp+48h] [rbp-D8h] BYREF
  __int64 v47[2]; // [rsp+50h] [rbp-D0h] BYREF
  char v48; // [rsp+64h] [rbp-BCh]
  _QWORD v49[4]; // [rsp+70h] [rbp-B0h] BYREF
  _QWORD *v50; // [rsp+90h] [rbp-90h] BYREF
  __int64 v51; // [rsp+98h] [rbp-88h]
  char v52; // [rsp+A4h] [rbp-7Ch]
  __int16 v53; // [rsp+B0h] [rbp-70h]
  __int64 v54[2]; // [rsp+C0h] [rbp-60h] BYREF
  __int64 *v55; // [rsp+D0h] [rbp-50h] BYREF
  char v56; // [rsp+D8h] [rbp-48h]
  __int16 v57; // [rsp+E0h] [rbp-40h]

  v3 = *(_QWORD *)(a1 + 8);
  v4 = *(_QWORD *)(a1 - 96);
  v5 = *(_QWORD *)(a1 - 64);
  v6 = *(_QWORD *)(a1 - 32);
  v44 = v3;
  if ( *(_BYTE *)v5 == 18 )
  {
    v41 = (_BYTE *)(v5 + 24);
  }
  else
  {
    v10 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v5 + 8) + 8LL) - 17;
    if ( (unsigned int)v10 > 1 )
      return 0;
    if ( *(_BYTE *)v5 > 0x15u )
      return 0;
    v11 = sub_AD7630(v5, 1, v10);
    if ( !v11 || *v11 != 18 )
      return 0;
    v41 = v11 + 24;
  }
  if ( *(_BYTE *)v6 == 18 )
  {
    v7 = (__int64 *)(v6 + 24);
  }
  else
  {
    v12 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v6 + 8) + 8LL) - 17;
    if ( (unsigned int)v12 > 1 )
      return 0;
    if ( *(_BYTE *)v6 > 0x15u )
      return 0;
    v13 = sub_AD7630(v6, 1, v12);
    if ( !v13 || *v13 != 18 )
      return 0;
    v7 = (__int64 *)(v13 + 24);
  }
  v43 = sub_C33340();
  if ( *(_QWORD **)v41 == v43 )
    sub_C3C790(v47, (_QWORD **)v41);
  else
    sub_C33EB0(v47, (__int64 *)v41);
  if ( v43 == (_QWORD *)v47[0] )
  {
    if ( (*(_BYTE *)(v47[1] + 20) & 8) == 0 )
      goto LABEL_31;
    sub_C3CCB0((__int64)v47);
    v14 = v47[0];
  }
  else
  {
    if ( (v48 & 8) == 0 )
    {
LABEL_9:
      sub_C338E0((__int64)v49, (__int64)v47);
      goto LABEL_10;
    }
    sub_C34440((unsigned __int8 *)v47);
    v14 = v47[0];
  }
  if ( v43 != (_QWORD *)v14 )
    goto LABEL_9;
LABEL_31:
  sub_C3C840(v49, v47);
LABEL_10:
  if ( (_QWORD *)*v7 == v43 )
    sub_C3C790(&v50, (_QWORD **)v7);
  else
    sub_C33EB0(&v50, v7);
  if ( v43 == v50 )
  {
    if ( (*(_BYTE *)(v51 + 20) & 8) == 0 )
      goto LABEL_34;
    sub_C3CCB0((__int64)&v50);
    v15 = (__int64)v50;
  }
  else
  {
    if ( (v52 & 8) == 0 )
    {
LABEL_14:
      sub_C338E0((__int64)v54, (__int64)&v50);
      goto LABEL_15;
    }
    sub_C34440((unsigned __int8 *)&v50);
    v15 = (__int64)v50;
  }
  if ( (_QWORD *)v15 != v43 )
    goto LABEL_14;
LABEL_34:
  sub_C3C840(v54, &v50);
LABEL_15:
  if ( v49[0] != v54[0] )
  {
    sub_91D830(v54);
    sub_91D830(&v50);
    sub_91D830(v49);
    sub_91D830(v47);
    return 0;
  }
  if ( (_QWORD *)v49[0] == v43 )
    v39 = sub_C3E590((__int64)v49, (__int64)v54);
  else
    v39 = sub_C33D00((__int64)v49, (__int64)v54);
  sub_91D830(v54);
  sub_91D830(&v50);
  sub_91D830(v49);
  sub_91D830(v47);
  if ( !v39 )
    return 0;
  BYTE4(v49[0]) = 0;
  LODWORD(v49[0]) = 42;
  v54[0] = (__int64)v49;
  v54[1] = (__int64)&v46;
  v55 = v47;
  v56 = 0;
  v16 = *(_QWORD *)(v4 + 16);
  if ( !v16 )
    return 0;
  if ( *(_QWORD *)(v16 + 8) )
    return 0;
  if ( *(_BYTE *)v4 != 82 )
    return 0;
  v17 = *(_QWORD *)(v4 - 64);
  if ( *(_BYTE *)v17 != 78 )
    return 0;
  v18 = *(_QWORD *)(v17 + 8);
  v19 = *(_QWORD *)(*(_QWORD *)(v17 - 32) + 8LL);
  v20 = *(unsigned __int8 *)(v18 + 8);
  v21 = *(unsigned __int8 *)(v19 + 8);
  if ( (unsigned int)(v20 - 17) <= 1 != (unsigned int)(v21 - 17) <= 1
    || (unsigned int)(v21 - 17) <= 1
    && (((_BYTE)v20 == 18) != ((_BYTE)v21 == 18) || *(_DWORD *)(v18 + 32) != *(_DWORD *)(v19 + 32)) )
  {
    return 0;
  }
  v46 = *(_QWORD *)(v17 - 32);
  if ( !(unsigned __int8)sub_991580((__int64)&v55, *(_QWORD *)(v4 - 32)) )
    return 0;
  if ( v54[0] )
  {
    v22 = sub_B53900(v4);
    v23 = v54[0];
    *(_DWORD *)v54[0] = v22;
    *(_BYTE *)(v23 + 4) = BYTE4(v22);
  }
  if ( !sub_9893F0(v49[0], v47[0], &v45) )
    return 0;
  v24 = v46;
  if ( v44 != *(_QWORD *)(v46 + 8) )
    return 0;
  v25 = v41;
  if ( v43 == *(_QWORD **)v41 )
    v25 = (_BYTE *)*((_QWORD *)v41 + 1);
  if ( v45 != ((v25[20] & 8) != 0) )
  {
    v53 = 257;
    v26 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD))(*(_QWORD *)a2[10] + 48LL))(
            a2[10],
            12,
            v46,
            *((unsigned int *)a2 + 26));
    if ( !v26 )
    {
      v57 = 257;
      v40 = *((_DWORD *)a2 + 26);
      v33 = sub_B50340(12, v24, (__int64)v54, 0, 0);
      v26 = v33;
      v34 = a2[12];
      if ( v34 )
        sub_B99FD0(v33, 3u, v34);
      sub_B45150(v26, v40);
      (*(void (__fastcall **)(__int64, __int64, _QWORD **, __int64, __int64))(*(_QWORD *)a2[11] + 16LL))(
        a2[11],
        v26,
        &v50,
        a2[7],
        a2[8]);
      if ( *a2 != *a2 + 16LL * *((unsigned int *)a2 + 2) )
      {
        v35 = *a2;
        v36 = *a2 + 16LL * *((unsigned int *)a2 + 2);
        do
        {
          v37 = *(_QWORD *)(v35 + 8);
          v38 = *(_DWORD *)v35;
          v35 += 16;
          sub_B99FD0(v26, v38, v37);
        }
        while ( v36 != v35 );
      }
    }
    v46 = v26;
  }
  if ( v43 == *(_QWORD **)v41 )
    sub_C3C790(&v50, (_QWORD **)v41);
  else
    sub_C33EB0(&v50, (__int64 *)v41);
  if ( v43 == v50 )
  {
    if ( (*(_BYTE *)(v51 + 20) & 8) == 0 )
      goto LABEL_66;
    sub_C3CCB0((__int64)&v50);
    v32 = (__int64)v50;
LABEL_69:
    if ( (_QWORD *)v32 != v43 )
      goto LABEL_60;
LABEL_66:
    sub_C3C840(v54, &v50);
    goto LABEL_61;
  }
  if ( (v52 & 8) != 0 )
  {
    sub_C34440((unsigned __int8 *)&v50);
    v32 = (__int64)v50;
    goto LABEL_69;
  }
LABEL_60:
  sub_C338E0((__int64)v54, (__int64)&v50);
LABEL_61:
  v27 = sub_AD8F10(v44, v54);
  sub_91D830(v54);
  sub_91D830(&v50);
  v54[0] = *(_QWORD *)(a1 + 8);
  v28 = (__int64 *)sub_B43CA0(a1);
  v29 = sub_B6E160(v28, 0x1Au, (__int64)v54, 1);
  v50 = (_QWORD *)v27;
  v30 = 0;
  v31 = v29;
  v57 = 257;
  v51 = v46;
  if ( v29 )
    v30 = *(_QWORD *)(v29 + 24);
  v8 = sub_BD2CC0(88, 3u);
  if ( v8 )
  {
    sub_B44260((__int64)v8, **(_QWORD **)(v30 + 16), 56, 3u, 0, 0);
    v8[9] = 0;
    sub_B4A290((__int64)v8, v30, v31, (__int64 *)&v50, 2, (__int64)v54, 0, 0);
  }
  return v8;
}
