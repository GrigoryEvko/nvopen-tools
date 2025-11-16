// Function: sub_23DFD50
// Address: 0x23dfd50
//
void __fastcall sub_23DFD50(_QWORD *a1, __int64 a2, __int64 a3)
{
  _BYTE *v4; // rbx
  __int64 v5; // r15
  __int64 v6; // r10
  __int64 v7; // r14
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 v10; // rax
  char v11; // al
  _QWORD *v12; // rax
  __int64 v13; // rbx
  unsigned int *v14; // r15
  unsigned int *v15; // r12
  __int64 v16; // rdx
  unsigned int v17; // esi
  __int64 v18; // rdx
  unsigned __int64 v19; // rsi
  __int64 v20; // rax
  __int64 v21; // r9
  __int64 v22; // rax
  __int64 v23; // rbx
  _BYTE *v24; // r14
  _BYTE *v25; // r15
  __int64 v26; // rax
  __int64 v27; // rdx
  int v28; // r15d
  __int64 v29; // r15
  unsigned int *v30; // r15
  unsigned int *v31; // rbx
  __int64 v32; // rdx
  unsigned int v33; // esi
  int v34; // r12d
  unsigned int *v35; // r12
  unsigned int *v36; // rbx
  __int64 v37; // rdx
  unsigned int v38; // esi
  char v39; // [rsp+8h] [rbp-198h]
  __int64 v40; // [rsp+38h] [rbp-168h]
  __int64 v42; // [rsp+48h] [rbp-158h]
  __int64 v43; // [rsp+48h] [rbp-158h]
  _QWORD v44[4]; // [rsp+50h] [rbp-150h] BYREF
  __int16 v45; // [rsp+70h] [rbp-130h]
  _DWORD v46[8]; // [rsp+80h] [rbp-120h] BYREF
  __int16 v47; // [rsp+A0h] [rbp-100h]
  _QWORD v48[4]; // [rsp+B0h] [rbp-F0h] BYREF
  __int16 v49; // [rsp+D0h] [rbp-D0h]
  unsigned int *v50; // [rsp+E0h] [rbp-C0h] BYREF
  __int64 v51; // [rsp+E8h] [rbp-B8h]
  _BYTE v52[32]; // [rsp+F0h] [rbp-B0h] BYREF
  __int64 v53; // [rsp+110h] [rbp-90h]
  __int64 v54; // [rsp+118h] [rbp-88h]
  __int64 v55; // [rsp+120h] [rbp-80h]
  __int64 v56; // [rsp+128h] [rbp-78h]
  void **v57; // [rsp+130h] [rbp-70h]
  void **v58; // [rsp+138h] [rbp-68h]
  __int64 v59; // [rsp+140h] [rbp-60h]
  int v60; // [rsp+148h] [rbp-58h]
  __int16 v61; // [rsp+14Ch] [rbp-54h]
  char v62; // [rsp+14Eh] [rbp-52h]
  __int64 v63; // [rsp+150h] [rbp-50h]
  __int64 v64; // [rsp+158h] [rbp-48h]
  void *v65; // [rsp+160h] [rbp-40h] BYREF
  void *v66; // [rsp+168h] [rbp-38h] BYREF

  v4 = (_BYTE *)a2;
  v56 = sub_BD5C60(a2);
  v57 = &v65;
  v58 = &v66;
  v50 = (unsigned int *)v52;
  v65 = &unk_49DA100;
  v61 = 512;
  v51 = 0x200000000LL;
  v59 = 0;
  v60 = 0;
  v62 = 7;
  v63 = 0;
  v64 = 0;
  v53 = 0;
  v54 = 0;
  LOWORD(v55) = 0;
  v66 = &unk_49DA0B0;
  sub_D5F1F0((__int64)&v50, a2);
  v5 = *(_QWORD *)(a3 + 8);
  v6 = a1[58];
  v47 = 257;
  if ( v6 == v5 )
  {
    v7 = a3;
  }
  else
  {
    v42 = v6;
    v7 = (*((__int64 (__fastcall **)(void **, __int64, __int64, __int64))*v57 + 15))(v57, 47, a3, v6);
    if ( v7 )
    {
      v5 = a1[58];
    }
    else
    {
      v49 = 257;
      v7 = sub_B51D30(47, a3, v42, (__int64)v48, 0, 0);
      if ( (unsigned __int8)sub_920620(v7) )
      {
        v28 = v60;
        if ( v59 )
          sub_B99FD0(v7, 3u, v59);
        sub_B45150(v7, v28);
      }
      (*((void (__fastcall **)(void **, __int64, _DWORD *, __int64, __int64))*v58 + 2))(v58, v7, v46, v54, v55);
      v29 = 4LL * (unsigned int)v51;
      if ( v50 != &v50[v29] )
      {
        v30 = &v50[v29];
        v31 = v50;
        do
        {
          v32 = *((_QWORD *)v31 + 1);
          v33 = *v31;
          v31 += 4;
          sub_B99FD0(v7, v33, v32);
        }
        while ( v30 != v31 );
        v4 = (_BYTE *)a2;
      }
      v5 = a1[58];
    }
  }
  if ( *v4 != 30 )
  {
    v46[1] = 0;
    v49 = 257;
    v44[0] = v5;
    v22 = sub_B33D10((__int64)&v50, 0xBAu, (__int64)v44, 1, 0, 0, v46[0], (__int64)v48);
    v47 = 257;
    v23 = a1[58];
    v24 = (_BYTE *)v22;
    v45 = 257;
    if ( v23 == *(_QWORD *)(a3 + 8) )
    {
      v25 = (_BYTE *)a3;
    }
    else
    {
      v25 = (_BYTE *)(*((__int64 (__fastcall **)(void **, __int64, __int64, __int64))*v57 + 15))(v57, 47, a3, v23);
      if ( !v25 )
      {
        v49 = 257;
        v25 = (_BYTE *)sub_B51D30(47, a3, v23, (__int64)v48, 0, 0);
        if ( (unsigned __int8)sub_920620((__int64)v25) )
        {
          v34 = v60;
          if ( v59 )
            sub_B99FD0((__int64)v25, 3u, v59);
          sub_B45150((__int64)v25, v34);
        }
        (*((void (__fastcall **)(void **, _BYTE *, _QWORD *, __int64, __int64))*v58 + 2))(v58, v25, v44, v54, v55);
        v35 = v50;
        v36 = &v50[4 * (unsigned int)v51];
        if ( v50 != v36 )
        {
          do
          {
            v37 = *((_QWORD *)v35 + 1);
            v38 = *v35;
            v35 += 4;
            sub_B99FD0((__int64)v25, v38, v37);
          }
          while ( v36 != v35 );
        }
      }
    }
    v26 = sub_929C50(&v50, v25, v24, (__int64)v46, 0, 0);
    v5 = a1[58];
    v7 = v26;
  }
  v8 = a1[2];
  v47 = 257;
  v9 = a1[748];
  v45 = 257;
  v40 = v8;
  v10 = sub_AA4E30(v53);
  v11 = sub_AE5020(v10, v5);
  v49 = 257;
  v39 = v11;
  v12 = sub_BD2C40(80, unk_3F10A14);
  v13 = (__int64)v12;
  if ( v12 )
    sub_B4D190((__int64)v12, v5, v9, (__int64)v48, 0, v39, 0, 0);
  (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v58 + 2))(v58, v13, v44, v54, v55);
  v14 = v50;
  v15 = &v50[4 * (unsigned int)v51];
  if ( v50 != v15 )
  {
    do
    {
      v16 = *((_QWORD *)v14 + 1);
      v17 = *v14;
      v14 += 4;
      sub_B99FD0(v13, v17, v16);
    }
    while ( v15 != v14 );
  }
  v18 = a1[672];
  v19 = a1[671];
  v48[0] = v13;
  v48[1] = v7;
  v20 = sub_921880(&v50, v19, v18, (int)v48, 2, (__int64)v46, 0);
  if ( *(_BYTE *)(v40 + 8) )
  {
    v27 = *(unsigned int *)(v40 + 24);
    if ( v27 + 1 > (unsigned __int64)*(unsigned int *)(v40 + 28) )
    {
      v43 = v20;
      sub_C8D5F0(v40 + 16, (const void *)(v40 + 32), v27 + 1, 8u, v27 + 1, v21);
      v27 = *(unsigned int *)(v40 + 24);
      v20 = v43;
    }
    *(_QWORD *)(*(_QWORD *)(v40 + 16) + 8 * v27) = v20;
    ++*(_DWORD *)(v40 + 24);
  }
  nullsub_61();
  v65 = &unk_49DA100;
  nullsub_63();
  if ( v50 != (unsigned int *)v52 )
    _libc_free((unsigned __int64)v50);
}
