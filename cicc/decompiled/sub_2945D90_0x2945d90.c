// Function: sub_2945D90
// Address: 0x2945d90
//
__int64 __fastcall sub_2945D90(__int64 a1, int *a2, __int64 a3, __int64 a4)
{
  unsigned int v6; // ecx
  __int64 v7; // r15
  void *v8; // r14
  char v9; // al
  __int64 v10; // rcx
  __int64 v11; // r9
  __int64 v12; // rsi
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  _QWORD *v16; // r14
  _QWORD *v17; // r13
  __int64 v18; // rax
  unsigned int v20; // [rsp+4h] [rbp-54Ch]
  unsigned int v21; // [rsp+8h] [rbp-548h]
  unsigned int v22; // [rsp+Ch] [rbp-544h]
  unsigned int v23; // [rsp+10h] [rbp-540h]
  __int16 v24; // [rsp+1Ah] [rbp-536h]
  int v25; // [rsp+1Ch] [rbp-534h]
  __int64 v26; // [rsp+20h] [rbp-530h]
  __int64 v27; // [rsp+28h] [rbp-528h]
  __int64 v28; // [rsp+30h] [rbp-520h] BYREF
  void **v29; // [rsp+38h] [rbp-518h]
  int v30; // [rsp+40h] [rbp-510h]
  int v31; // [rsp+44h] [rbp-50Ch]
  int v32; // [rsp+48h] [rbp-508h]
  char v33; // [rsp+4Ch] [rbp-504h]
  void *v34; // [rsp+50h] [rbp-500h] BYREF
  __int64 v35; // [rsp+60h] [rbp-4F0h] BYREF
  char *v36; // [rsp+68h] [rbp-4E8h]
  __int64 v37; // [rsp+70h] [rbp-4E0h]
  int v38; // [rsp+78h] [rbp-4D8h]
  char v39; // [rsp+7Ch] [rbp-4D4h]
  char v40; // [rsp+80h] [rbp-4D0h] BYREF
  _BYTE v41[8]; // [rsp+90h] [rbp-4C0h] BYREF
  int v42; // [rsp+98h] [rbp-4B8h] BYREF
  _QWORD *v43; // [rsp+A0h] [rbp-4B0h]
  int *v44; // [rsp+A8h] [rbp-4A8h]
  int *v45; // [rsp+B0h] [rbp-4A0h]
  __int64 v46; // [rsp+B8h] [rbp-498h]
  _BYTE *v47; // [rsp+C0h] [rbp-490h]
  __int64 v48; // [rsp+C8h] [rbp-488h]
  _BYTE v49[264]; // [rsp+D0h] [rbp-480h] BYREF
  _BYTE *v50; // [rsp+1D8h] [rbp-378h]
  __int64 v51; // [rsp+1E0h] [rbp-370h]
  _BYTE v52[768]; // [rsp+1E8h] [rbp-368h] BYREF
  __int64 v53; // [rsp+4E8h] [rbp-68h]
  __int64 v54; // [rsp+4F0h] [rbp-60h]
  __int16 v55; // [rsp+4F8h] [rbp-58h]
  int v56; // [rsp+4FCh] [rbp-54h]
  __int64 v57; // [rsp+500h] [rbp-50h]
  void *v58; // [rsp+508h] [rbp-48h]
  unsigned __int64 v59; // [rsp+510h] [rbp-40h]
  __int64 v60; // [rsp+518h] [rbp-38h]

  v26 = sub_BC1CD0(a4, &unk_4F81450, a3) + 8;
  v27 = sub_BC1CD0(a4, &unk_4F89C30, a3) + 8;
  v25 = *a2;
  v24 = *((_WORD *)a2 + 2);
  sub_C7D6A0(0, 0, 4);
  v6 = a2[8];
  if ( v6 )
  {
    v23 = a2[8];
    v7 = 4LL * v6;
    v8 = (void *)sub_C7D670(v7, 4);
    v21 = a2[6];
    v20 = a2[7];
    memcpy(v8, *((const void **)a2 + 2), v7);
    v6 = v23;
  }
  else
  {
    v20 = 0;
    v7 = 0;
    v8 = 0;
    v21 = 0;
  }
  v44 = &v42;
  v45 = &v42;
  v48 = 0x1000000000LL;
  v50 = v52;
  v51 = 0x2000000000LL;
  v22 = v6;
  v53 = v26;
  v42 = 0;
  v54 = v27;
  v43 = 0;
  v55 = v24;
  v46 = 0;
  v47 = v49;
  v56 = v25;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v60 = 0;
  sub_C7D6A0(0, 0, 4);
  LODWORD(v60) = v22;
  if ( v22 )
  {
    v58 = (void *)sub_C7D670(v7, 4);
    v59 = __PAIR64__(v20, v21);
    memcpy(v58, v8, 4LL * (unsigned int)v60);
  }
  else
  {
    v58 = 0;
    v59 = 0;
  }
  sub_C7D6A0((__int64)v8, v7, 4);
  v9 = sub_2942CE0((__int64)v41, a3);
  v30 = 2;
  v29 = &v34;
  v34 = &unk_4F81450;
  v12 = a1 + 32;
  v32 = 0;
  v33 = 1;
  v35 = 0;
  v36 = &v40;
  v37 = 2;
  v38 = 0;
  v39 = 1;
  v31 = 1;
  v28 = 1;
  if ( v9 )
  {
    sub_C8CD80(a1, v12, (__int64)&v28, v10, (__int64)&v28, v11);
    sub_C8CD80(a1 + 48, a1 + 80, (__int64)&v35, v13, v14, v15);
    if ( !v39 )
      _libc_free((unsigned __int64)v36);
    if ( !v33 )
      _libc_free((unsigned __int64)v29);
  }
  else
  {
    *(_QWORD *)(a1 + 8) = v12;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
  }
  sub_C7D6A0((__int64)v58, 4LL * (unsigned int)v60, 4);
  v16 = v50;
  v17 = &v50[24 * (unsigned int)v51];
  if ( v50 != (_BYTE *)v17 )
  {
    do
    {
      v18 = *(v17 - 1);
      v17 -= 3;
      if ( v18 != 0 && v18 != -4096 && v18 != -8192 )
        sub_BD60C0(v17);
    }
    while ( v16 != v17 );
    v17 = v50;
  }
  if ( v17 != (_QWORD *)v52 )
    _libc_free((unsigned __int64)v17);
  if ( v47 != v49 )
    _libc_free((unsigned __int64)v47);
  sub_293AA10(v43);
  return a1;
}
