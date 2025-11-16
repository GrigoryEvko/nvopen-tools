// Function: sub_320DBD0
// Address: 0x320dbd0
//
char __fastcall sub_320DBD0(_QWORD *a1, __int64 *a2)
{
  _QWORD *v2; // rax
  __int64 v4; // rax
  unsigned __int8 v5; // dl
  unsigned int v6; // r13d
  unsigned int v7; // eax
  int v8; // r13d
  unsigned __int16 v9; // r13
  __int64 v10; // rax
  __int64 v11; // rax
  unsigned __int8 v12; // dl
  _QWORD *v13; // rax
  _BYTE *v14; // r14
  unsigned __int8 v15; // al
  _QWORD *v16; // r14
  __int64 v17; // rax
  unsigned __int8 v18; // dl
  _BYTE **v19; // rax
  _BYTE *v20; // rax
  unsigned __int8 v21; // dl
  _BYTE **v22; // rax
  __int64 v23; // rax
  unsigned __int8 v24; // dl
  _BYTE **v25; // rax
  _BYTE *v26; // rsi
  unsigned __int8 v27; // al
  _BYTE **v28; // rsi
  __int64 v29; // r15
  __int64 v30; // rsi
  __int64 v31; // rsi
  __int64 v32; // rax
  unsigned __int8 v33; // dl
  __int64 v34; // r13
  __int64 (__fastcall *v35)(__int64, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, const char *, __int64, _QWORD); // rbx
  __int64 v36; // rax
  _BYTE *v37; // rax
  _BYTE *v38; // rdi
  const char *v39; // rax
  __int64 v40; // rdx
  unsigned int v41; // r14d
  unsigned int v42; // eax
  __int64 v43; // rax
  __int64 v44; // r13
  char v45; // r15
  __int64 v46; // r14
  unsigned __int8 **v47; // rax
  unsigned __int8 *v48; // rax
  __int64 v49; // rdx
  __int64 v50; // rcx
  __int64 v51; // r8
  __int64 v52; // r9
  unsigned __int8 v53; // al
  __int64 v54; // r13
  unsigned __int8 *v55; // rax
  __int64 v56; // rax
  const char *v58; // [rsp-68h] [rbp-68h]
  __int64 v59; // [rsp-60h] [rbp-60h]
  unsigned int v60; // [rsp-50h] [rbp-50h]
  unsigned int v61; // [rsp-4Ch] [rbp-4Ch]
  int v62; // [rsp-3Ch] [rbp-3Ch] BYREF

  v2 = (_QWORD *)*a2;
  if ( !*a2 || (_QWORD *)a1[3] == v2 )
    return (char)v2;
  v4 = sub_B10CD0((__int64)a2);
  v5 = *(_BYTE *)(v4 - 16);
  v2 = (v5 & 2) != 0 ? *(_QWORD **)(v4 - 32) : (_QWORD *)(v4 - 16 - 8LL * ((v5 >> 2) & 0xF));
  if ( !*v2 )
    return (char)v2;
  v6 = sub_B10CE0((__int64)a2);
  v7 = sub_B10CE0((__int64)a2);
  sub_3708840(&v62, v7, v6, 1);
  v8 = v62 & 0xFFFFFF;
  LODWORD(v2) = sub_B10CE0((__int64)a2);
  if ( (_DWORD)v2 != v8 )
    return (char)v2;
  LOBYTE(v2) = (v62 & 0xFFFFFF) == (_DWORD)&loc_F00F00;
  if ( (unsigned __int8)v2 | ((v62 & 0xFFFFFF) == 16707566) )
    return (char)v2;
  v9 = sub_B10CF0((__int64)a2);
  LODWORD(v2) = sub_B10CF0((__int64)a2);
  if ( (_DWORD)v2 != v9 )
    return (char)v2;
  v10 = a1[99];
  if ( !*(_BYTE *)(v10 + 489) )
    *(_BYTE *)(v10 + 489) = 1;
  if ( !sub_B10CD0((__int64)(a1 + 3)) )
    goto LABEL_25;
  v11 = sub_B10CD0((__int64)(a1 + 3));
  v12 = *(_BYTE *)(v11 - 16);
  if ( (v12 & 2) != 0 )
    v13 = *(_QWORD **)(v11 - 32);
  else
    v13 = (_QWORD *)(v11 - 16 - 8LL * ((v12 >> 2) & 0xF));
  v14 = (_BYTE *)*v13;
  if ( *(_BYTE *)*v13 != 16 )
  {
    v15 = *(v14 - 16);
    if ( (v15 & 2) != 0 )
      v16 = (_QWORD *)*((_QWORD *)v14 - 4);
    else
      v16 = &v14[-8 * ((v15 >> 2) & 0xF) - 16];
    v14 = (_BYTE *)*v16;
  }
  v17 = sub_B10CD0((__int64)a2);
  v18 = *(_BYTE *)(v17 - 16);
  if ( (v18 & 2) != 0 )
    v19 = *(_BYTE ***)(v17 - 32);
  else
    v19 = (_BYTE **)(v17 - 16 - 8LL * ((v18 >> 2) & 0xF));
  v20 = *v19;
  if ( *v20 != 16 )
  {
    v21 = *(v20 - 16);
    if ( (v21 & 2) != 0 )
      v22 = (_BYTE **)*((_QWORD *)v20 - 4);
    else
      v22 = (_BYTE **)&v20[-8 * ((v21 >> 2) & 0xF) - 16];
    v20 = *v22;
  }
  if ( v14 == v20 )
  {
    v60 = *(_DWORD *)(a1[99] + 460LL);
  }
  else
  {
LABEL_25:
    v23 = sub_B10CD0((__int64)a2);
    v24 = *(_BYTE *)(v23 - 16);
    if ( (v24 & 2) != 0 )
      v25 = *(_BYTE ***)(v23 - 32);
    else
      v25 = (_BYTE **)(v23 - 16 - 8LL * ((v24 >> 2) & 0xF));
    v26 = *v25;
    if ( **v25 != 16 )
    {
      v27 = *(v26 - 16);
      if ( (v27 & 2) != 0 )
        v28 = (_BYTE **)*((_QWORD *)v26 - 4);
      else
        v28 = (_BYTE **)&v26[-8 * ((v27 >> 2) & 0xF) - 16];
      v26 = *v28;
    }
    v29 = a1[99];
    v60 = sub_31FF830((__int64)a1, (unsigned __int64)v26);
    *(_DWORD *)(v29 + 460) = v60;
  }
  if ( a1 + 3 != a2 )
  {
    v30 = a1[3];
    if ( v30 )
      sub_B91220((__int64)(a1 + 3), v30);
    v31 = *a2;
    a1[3] = *a2;
    if ( v31 )
      sub_B96E90((__int64)(a1 + 3), v31, 1);
  }
  v61 = *(_DWORD *)(a1[99] + 456LL);
  v32 = sub_B10CD0((__int64)a2);
  v33 = *(_BYTE *)(v32 - 16);
  if ( (v33 & 2) != 0 )
  {
    if ( *(_DWORD *)(v32 - 24) != 2 )
      goto LABEL_39;
    v43 = *(_QWORD *)(v32 - 32);
LABEL_51:
    v44 = *(_QWORD *)(v43 + 8);
    if ( v44 )
    {
      v45 = 1;
      v46 = sub_B10CD0((__int64)a2);
      v47 = (unsigned __int8 **)sub_A17150((_BYTE *)(v46 - 16));
      v48 = sub_AF34D0(*v47);
      v61 = *(_DWORD *)(sub_320C1A0((__int64)a1, v44, (__int64)v48) + 136);
      while ( 1 )
      {
        v53 = *(_BYTE *)(v46 - 16);
        if ( (v53 & 2) != 0 )
        {
          if ( *(_DWORD *)(v46 - 24) != 2 )
            goto LABEL_55;
          v49 = *(_QWORD *)(v46 - 32);
          v54 = *(_QWORD *)(v49 + 8);
          if ( !v54 )
            goto LABEL_55;
        }
        else
        {
          v49 = (*(_WORD *)(v46 - 16) >> 6) & 0xF;
          if ( ((*(_WORD *)(v46 - 16) >> 6) & 0xF) != 2
            || (v49 = v46 - 16 - 8LL * ((v53 >> 2) & 0xF), (v54 = *(_QWORD *)(v49 + 8)) == 0) )
          {
LABEL_55:
            sub_31F42C0(a1[99] + 56LL, v46, v49, v50, v51, v52);
            goto LABEL_39;
          }
        }
        v55 = sub_AF34D0(*(unsigned __int8 **)v49);
        v56 = sub_320C1A0((__int64)a1, v54, (__int64)v55);
        if ( !v45 )
          sub_31F42C0(v56 + 104, v46, v49, v50, v51, v52);
        v45 = 0;
        v46 = v54;
      }
    }
    goto LABEL_39;
  }
  if ( ((*(_WORD *)(v32 - 16) >> 6) & 0xF) == 2 )
  {
    v43 = v32 - 16 - 8LL * ((v33 >> 2) & 0xF);
    goto LABEL_51;
  }
LABEL_39:
  v34 = a1[66];
  v35 = *(__int64 (__fastcall **)(__int64, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, const char *, __int64, _QWORD))(*(_QWORD *)v34 + 736LL);
  v36 = sub_B10CD0((__int64)a2);
  v37 = sub_A17150((_BYTE *)(v36 - 16));
  v38 = *(_BYTE **)v37;
  if ( **(_BYTE **)v37 == 16 || (v38 = *(_BYTE **)sub_A17150(v38 - 16)) != 0 )
  {
    v39 = (const char *)sub_A547D0((__int64)v38, 0);
  }
  else
  {
    v40 = 0;
    v39 = byte_3F871B3;
  }
  v59 = v40;
  v58 = v39;
  v41 = sub_B10CF0((__int64)a2);
  v42 = sub_B10CE0((__int64)a2);
  LOBYTE(v2) = v35(v34, v61, v60, v42, v41, 0, 0, v58, v59, 0);
  return (char)v2;
}
