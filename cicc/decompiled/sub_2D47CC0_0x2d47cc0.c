// Function: sub_2D47CC0
// Address: 0x2d47cc0
//
__int64 __fastcall sub_2D47CC0(unsigned __int64 *a1, __int64 a2)
{
  unsigned __int64 v3; // rax
  __int64 v4; // r14
  __int64 v5; // rbx
  __int64 v6; // r15
  __int64 v7; // r14
  unsigned __int16 v8; // r9
  char v9; // cl
  _QWORD *v10; // rax
  __int64 v11; // rbx
  char *v12; // r15
  char *v13; // r14
  __int64 v14; // rdx
  unsigned int v15; // esi
  __int64 v16; // rsi
  __int64 v18; // rbx
  char *v19; // r15
  char *v20; // rbx
  __int64 v21; // rdx
  unsigned int v22; // esi
  __int64 v23; // rbx
  __int64 v24; // r9
  _QWORD *v25; // rax
  char *v26; // rbx
  char *v27; // r14
  __int64 v28; // rdx
  unsigned int v29; // esi
  char *v30; // rbx
  char *v31; // r14
  __int64 v32; // rdx
  unsigned int v33; // esi
  __int64 v34; // [rsp-10h] [rbp-250h]
  __int64 v35; // [rsp-8h] [rbp-248h]
  _QWORD *v36; // [rsp+0h] [rbp-240h]
  char v37; // [rsp+8h] [rbp-238h]
  __int16 v38; // [rsp+Ch] [rbp-234h]
  __int64 v39; // [rsp+10h] [rbp-230h]
  __int16 v40; // [rsp+18h] [rbp-228h]
  unsigned __int8 v41; // [rsp+1Ch] [rbp-224h]
  __int64 v42; // [rsp+20h] [rbp-220h]
  __int64 v43; // [rsp+20h] [rbp-220h]
  _BYTE v44[32]; // [rsp+30h] [rbp-210h] BYREF
  __int16 v45; // [rsp+50h] [rbp-1F0h]
  _QWORD v46[4]; // [rsp+60h] [rbp-1E0h] BYREF
  char v47; // [rsp+80h] [rbp-1C0h]
  char v48; // [rsp+81h] [rbp-1BFh]
  _BYTE v49[32]; // [rsp+90h] [rbp-1B0h] BYREF
  __int16 v50; // [rsp+B0h] [rbp-190h]
  _QWORD v51[4]; // [rsp+C0h] [rbp-180h] BYREF
  unsigned __int8 v52; // [rsp+E0h] [rbp-160h]
  __int64 v53; // [rsp+E8h] [rbp-158h]
  __int64 v54; // [rsp+F8h] [rbp-148h]
  char *v55; // [rsp+100h] [rbp-140h] BYREF
  unsigned int v56; // [rsp+108h] [rbp-138h]
  char v57; // [rsp+110h] [rbp-130h] BYREF
  __int64 v58; // [rsp+138h] [rbp-108h]
  __int64 v59; // [rsp+140h] [rbp-100h]
  __int64 v60; // [rsp+150h] [rbp-F0h]
  __int64 v61; // [rsp+158h] [rbp-E8h]
  void *v62; // [rsp+180h] [rbp-C0h]
  void *v63; // [rsp+188h] [rbp-B8h]
  _QWORD v64[12]; // [rsp+1E0h] [rbp-60h] BYREF

  sub_2D46B10((__int64)&v55, a2, a1[1]);
  LOWORD(v3) = *(_WORD *)(a2 + 2);
  v36 = v51;
  v40 = ((unsigned __int16)v3 >> 4) & 0x1F;
  _BitScanReverse64(&v3, 1LL << ((unsigned __int16)v3 >> 9));
  sub_2D44EF0(
    (__int64)v51,
    (__int64)&v55,
    a2,
    *(_QWORD *)(a2 + 8),
    *(_QWORD *)(a2 - 64),
    63 - (v3 ^ 0x3F),
    *(_DWORD *)(*a1 + 96) >> 3);
  v4 = *(_QWORD *)(a2 - 32);
  v46[0] = "ValOperand_Shifted";
  v5 = v51[0];
  v48 = 1;
  v47 = 3;
  v42 = v53;
  v45 = 257;
  if ( v51[0] == *(_QWORD *)(v4 + 8) )
  {
    v6 = v4;
  }
  else
  {
    v6 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64, __int64, _QWORD *))(*(_QWORD *)v60
                                                                                                  + 120LL))(
           v60,
           39,
           v4,
           v51[0],
           v34,
           v35,
           v51);
    if ( !v6 )
    {
      v50 = 257;
      v25 = sub_BD2C40(72, 1u);
      v6 = (__int64)v25;
      if ( v25 )
        sub_B515B0((__int64)v25, v4, v5, (__int64)v49, 0, 0);
      (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v61 + 16LL))(
        v61,
        v6,
        v44,
        v58,
        v59);
      v26 = &v55[16 * v56];
      if ( v55 != v26 )
      {
        v27 = v55;
        do
        {
          v28 = *((_QWORD *)v27 + 1);
          v29 = *(_DWORD *)v27;
          v27 += 16;
          sub_B99FD0(v6, v29, v28);
        }
        while ( v26 != v27 );
      }
    }
  }
  v7 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v60 + 32LL))(
         v60,
         25,
         v6,
         v42,
         0,
         0);
  if ( v7
    || (v50 = 257,
        v7 = sub_B504D0(25, v6, v42, (__int64)v49, 0, 0),
        (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)v61 + 16LL))(
          v61,
          v7,
          v46,
          v58,
          v59),
        v18 = 16LL * v56,
        v19 = &v55[v18],
        v55 == &v55[v18]) )
  {
    if ( v40 != 3 )
      goto LABEL_5;
  }
  else
  {
    v20 = v55;
    do
    {
      v21 = *((_QWORD *)v20 + 1);
      v22 = *(_DWORD *)v20;
      v20 += 16;
      sub_B99FD0(v7, v22, v21);
    }
    while ( v19 != v20 );
    if ( v40 != 3 )
      goto LABEL_5;
  }
  v48 = 1;
  v47 = 3;
  v23 = v54;
  v46[0] = "AndOperand";
  v24 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v60 + 16LL))(v60, 29, v7, v54);
  if ( !v24 )
  {
    v50 = 257;
    v43 = sub_B504D0(29, v7, v23, (__int64)v49, 0, 0);
    (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)v61 + 16LL))(
      v61,
      v43,
      v46,
      v58,
      v59);
    v24 = v43;
    v30 = &v55[16 * v56];
    if ( v55 != v30 )
    {
      v31 = v55;
      do
      {
        v32 = *((_QWORD *)v31 + 1);
        v33 = *(_DWORD *)v31;
        v31 += 16;
        sub_B99FD0(v43, v33, v32);
      }
      while ( v30 != v31 );
      v24 = v43;
    }
  }
  v7 = v24;
LABEL_5:
  v8 = *(_WORD *)(a2 + 2);
  v9 = *(_BYTE *)(a2 + 72);
  v50 = 257;
  v37 = v9;
  v38 = (v8 >> 1) & 7;
  v39 = v51[3];
  v41 = v52;
  v10 = sub_BD2C40(80, unk_3F148C0);
  v11 = (__int64)v10;
  if ( v10 )
    sub_B4D750((__int64)v10, v40, v39, v7, v41, v38, v37, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64))(*(_QWORD *)v61 + 16LL))(v61, v11, v49, v58, v59);
  v12 = &v55[16 * v56];
  if ( v55 != v12 )
  {
    v13 = v55;
    do
    {
      v14 = *((_QWORD *)v13 + 1);
      v15 = *(_DWORD *)v13;
      v13 += 16;
      sub_B99FD0(v11, v15, v14);
    }
    while ( v12 != v13 );
  }
  sub_2D42CA0(v11, a2);
  v16 = v11;
  if ( v51[0] != v51[1] )
    v16 = sub_2D44750((__int64 *)&v55, v11, v36);
  sub_BD84D0(a2, v16);
  sub_B43D60((_QWORD *)a2);
  sub_B32BF0(v64);
  v62 = &unk_49E5698;
  v63 = &unk_49D94D0;
  nullsub_63();
  nullsub_63();
  if ( v55 != &v57 )
    _libc_free((unsigned __int64)v55);
  return v11;
}
