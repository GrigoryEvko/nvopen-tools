// Function: sub_2D47520
// Address: 0x2d47520
//
__int64 __fastcall sub_2D47520(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rcx
  __int64 v5; // r12
  __int64 v6; // r10
  __int64 v7; // r14
  bool v8; // zf
  __int64 v9; // rbx
  unsigned __int16 v10; // cx
  unsigned __int64 v11; // rax
  _QWORD *v12; // rax
  __int64 v13; // r12
  char *v14; // rbx
  char *v15; // r14
  __int64 v16; // rdx
  unsigned int v17; // esi
  __int64 v18; // rbx
  __int64 v19; // r14
  int v21; // r15d
  char *v22; // rbx
  char *v23; // r15
  __int64 v24; // rdx
  unsigned int v25; // esi
  int v26; // r12d
  __int64 v27; // r12
  char *v28; // r12
  __int64 v29; // rdx
  unsigned int v30; // esi
  int v31; // r15d
  char *v32; // rbx
  char *v33; // r15
  __int64 v34; // rdx
  unsigned int v35; // esi
  int v36; // r12d
  __int64 v37; // r12
  char *v38; // r12
  __int64 v39; // rdx
  unsigned int v40; // esi
  char v41; // [rsp+Ch] [rbp-1B4h]
  __int16 v42; // [rsp+10h] [rbp-1B0h]
  unsigned __int8 v43; // [rsp+14h] [rbp-1ACh]
  __int64 v44; // [rsp+18h] [rbp-1A8h]
  __int64 v45; // [rsp+18h] [rbp-1A8h]
  char *v46; // [rsp+18h] [rbp-1A8h]
  char *v47; // [rsp+18h] [rbp-1A8h]
  _BYTE v48[32]; // [rsp+20h] [rbp-1A0h] BYREF
  __int16 v49; // [rsp+40h] [rbp-180h]
  _BYTE v50[32]; // [rsp+50h] [rbp-170h] BYREF
  __int16 v51; // [rsp+70h] [rbp-150h]
  char *v52; // [rsp+80h] [rbp-140h] BYREF
  unsigned int v53; // [rsp+88h] [rbp-138h]
  char v54; // [rsp+90h] [rbp-130h] BYREF
  __int64 v55; // [rsp+B8h] [rbp-108h]
  __int64 v56; // [rsp+C0h] [rbp-100h]
  __int64 v57; // [rsp+D0h] [rbp-F0h]
  __int64 v58; // [rsp+D8h] [rbp-E8h]
  __int64 v59; // [rsp+E0h] [rbp-E0h]
  int v60; // [rsp+E8h] [rbp-D8h]
  void *v61; // [rsp+100h] [rbp-C0h]
  void *v62; // [rsp+108h] [rbp-B8h]
  _QWORD v63[12]; // [rsp+160h] [rbp-60h] BYREF

  v3 = sub_B43CA0(a2);
  v5 = sub_2D43EB0(*(_QWORD *)a1, *(__int64 **)(a2 + 8), v3 + 312, v4);
  sub_2D46B10((__int64)&v52, a2, *(_QWORD *)(a1 + 8));
  v6 = *(_QWORD *)(a2 - 32);
  v7 = *(_QWORD *)(a2 - 64);
  v8 = *(_BYTE *)(*(_QWORD *)(v6 + 8) + 8LL) == 14;
  v49 = 257;
  if ( !v8 )
  {
    if ( v5 != *(_QWORD *)(v6 + 8) )
    {
      v44 = v6;
      v9 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v57 + 120LL))(v57, 49, v6, v5);
      if ( !v9 )
      {
        v51 = 257;
        v9 = sub_B51D30(49, v44, v5, (__int64)v50, 0, 0);
        if ( (unsigned __int8)sub_920620(v9) )
        {
          v36 = v60;
          if ( v59 )
            sub_B99FD0(v9, 3u, v59);
          sub_B45150(v9, v36);
        }
        (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v58 + 16LL))(
          v58,
          v9,
          v48,
          v55,
          v56);
        v37 = 16LL * v53;
        v47 = &v52[v37];
        if ( v52 != &v52[v37] )
        {
          v38 = v52;
          do
          {
            v39 = *((_QWORD *)v38 + 1);
            v40 = *(_DWORD *)v38;
            v38 += 16;
            sub_B99FD0(v9, v40, v39);
          }
          while ( v47 != v38 );
        }
      }
      goto LABEL_4;
    }
LABEL_48:
    v9 = v6;
    goto LABEL_4;
  }
  if ( v5 == *(_QWORD *)(v6 + 8) )
    goto LABEL_48;
  v45 = v6;
  v9 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v57 + 120LL))(v57, 47, v6, v5);
  if ( !v9 )
  {
    v51 = 257;
    v9 = sub_B51D30(47, v45, v5, (__int64)v50, 0, 0);
    if ( (unsigned __int8)sub_920620(v9) )
    {
      v26 = v60;
      if ( v59 )
        sub_B99FD0(v9, 3u, v59);
      sub_B45150(v9, v26);
    }
    (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v58 + 16LL))(
      v58,
      v9,
      v48,
      v55,
      v56);
    v27 = 16LL * v53;
    v46 = &v52[v27];
    if ( v52 != &v52[v27] )
    {
      v28 = v52;
      do
      {
        v29 = *((_QWORD *)v28 + 1);
        v30 = *(_DWORD *)v28;
        v28 += 16;
        sub_B99FD0(v9, v30, v29);
      }
      while ( v46 != v28 );
    }
  }
LABEL_4:
  v10 = *(_WORD *)(a2 + 2);
  v51 = 257;
  v41 = *(_BYTE *)(a2 + 72);
  _BitScanReverse64(&v11, 1LL << (v10 >> 9));
  v42 = (v10 >> 1) & 7;
  v43 = 63 - (v11 ^ 0x3F);
  v12 = sub_BD2C40(80, unk_3F148C0);
  v13 = (__int64)v12;
  if ( v12 )
    sub_B4D750((__int64)v12, 0, v7, v9, v43, v42, v41, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v58 + 16LL))(v58, v13, v50, v55, v56);
  v14 = v52;
  v15 = &v52[16 * v53];
  if ( v52 != v15 )
  {
    do
    {
      v16 = *((_QWORD *)v14 + 1);
      v17 = *(_DWORD *)v14;
      v14 += 16;
      sub_B99FD0(v13, v17, v16);
    }
    while ( v15 != v14 );
  }
  *(_WORD *)(v13 + 2) = *(_WORD *)(a2 + 2) & 1 | *(_WORD *)(v13 + 2) & 0xFFFE;
  sub_2D42CA0(v13, a2);
  v18 = *(_QWORD *)(a2 + 8);
  v8 = *(_BYTE *)(v18 + 8) == 14;
  v49 = 257;
  if ( !v8 )
  {
    if ( v18 != *(_QWORD *)(v13 + 8) )
    {
      v19 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v57 + 120LL))(v57, 49, v13, v18);
      if ( !v19 )
      {
        v51 = 257;
        v19 = sub_B51D30(49, v13, v18, (__int64)v50, 0, 0);
        if ( (unsigned __int8)sub_920620(v19) )
        {
          v31 = v60;
          if ( v59 )
            sub_B99FD0(v19, 3u, v59);
          sub_B45150(v19, v31);
        }
        (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v58 + 16LL))(
          v58,
          v19,
          v48,
          v55,
          v56);
        v32 = v52;
        v33 = &v52[16 * v53];
        if ( v52 != v33 )
        {
          do
          {
            v34 = *((_QWORD *)v32 + 1);
            v35 = *(_DWORD *)v32;
            v32 += 16;
            sub_B99FD0(v19, v35, v34);
          }
          while ( v33 != v32 );
        }
      }
      goto LABEL_11;
    }
LABEL_49:
    v19 = v13;
    goto LABEL_11;
  }
  if ( v18 == *(_QWORD *)(v13 + 8) )
    goto LABEL_49;
  v19 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v57 + 120LL))(v57, 48, v13, v18);
  if ( !v19 )
  {
    v51 = 257;
    v19 = sub_B51D30(48, v13, v18, (__int64)v50, 0, 0);
    if ( (unsigned __int8)sub_920620(v19) )
    {
      v21 = v60;
      if ( v59 )
        sub_B99FD0(v19, 3u, v59);
      sub_B45150(v19, v21);
    }
    (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v58 + 16LL))(
      v58,
      v19,
      v48,
      v55,
      v56);
    v22 = v52;
    v23 = &v52[16 * v53];
    if ( v52 != v23 )
    {
      do
      {
        v24 = *((_QWORD *)v22 + 1);
        v25 = *(_DWORD *)v22;
        v22 += 16;
        sub_B99FD0(v19, v25, v24);
      }
      while ( v23 != v22 );
    }
  }
LABEL_11:
  sub_BD84D0(a2, v19);
  sub_B43D60((_QWORD *)a2);
  sub_B32BF0(v63);
  v61 = &unk_49E5698;
  v62 = &unk_49D94D0;
  nullsub_63();
  nullsub_63();
  if ( v52 != &v54 )
    _libc_free((unsigned __int64)v52);
  return v13;
}
