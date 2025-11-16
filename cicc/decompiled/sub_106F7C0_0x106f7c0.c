// Function: sub_106F7C0
// Address: 0x106f7c0
//
__int64 __fastcall sub_106F7C0(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 *v3; // r14
  __int64 *v4; // r15
  unsigned __int64 v5; // r12
  __int64 v6; // rbx
  __int64 v7; // r8
  __int64 v8; // r9
  __int64 v9; // r13
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  int *v12; // r14
  __int64 v13; // r13
  unsigned __int8 i; // al
  unsigned __int8 *v15; // r14
  __int64 v16; // rdi
  __int16 v17; // ax
  __int64 v18; // rdi
  __int64 v19; // rdi
  unsigned __int32 v20; // eax
  __int64 v21; // rdi
  int *v22; // rsi
  _DWORD *v23; // r13
  _DWORD *v24; // r12
  __int64 v25; // rdi
  unsigned int v26; // eax
  __int64 *v27; // rax
  __int64 v28; // r13
  _QWORD *v29; // rbx
  __int64 v30; // rax
  __int64 v31; // r12
  __int64 v32; // rax
  char *v33; // r13
  char *v34; // rbx
  char v35; // al
  __int64 *v36; // r13
  int v37; // eax
  __int64 v38; // rdi
  unsigned __int32 v39; // edx
  _DWORD *v40; // rbx
  __int64 v41; // rax
  __int64 v43; // rbx
  __int64 *v44; // r12
  unsigned __int64 v45; // rax
  char v46; // dl
  unsigned __int64 v47; // rcx
  char v48; // al
  __int64 v49; // rdx
  unsigned __int64 v50; // rax
  __int64 v51; // r12
  unsigned __int64 v52; // rdx
  char v53; // al
  unsigned __int8 v54; // al
  __int64 v55; // [rsp+10h] [rbp-120h]
  unsigned __int8 *v56; // [rsp+10h] [rbp-120h]
  __int64 v57; // [rsp+18h] [rbp-118h]
  __int64 *v58; // [rsp+20h] [rbp-110h]
  int v59; // [rsp+28h] [rbp-108h]
  __int64 v60; // [rsp+28h] [rbp-108h]
  __int64 *v62; // [rsp+38h] [rbp-F8h]
  unsigned int v63; // [rsp+38h] [rbp-F8h]
  __int64 *v64; // [rsp+38h] [rbp-F8h]
  unsigned __int8 v65; // [rsp+4Fh] [rbp-E1h] BYREF
  int v66; // [rsp+50h] [rbp-E0h] BYREF
  _DWORD v67[2]; // [rsp+54h] [rbp-DCh] BYREF
  char v68; // [rsp+5Ch] [rbp-D4h]
  char v69; // [rsp+5Dh] [rbp-D3h]
  int v70; // [rsp+60h] [rbp-D0h]
  int v71; // [rsp+64h] [rbp-CCh]
  char v72; // [rsp+68h] [rbp-C8h] BYREF
  _BYTE *v73; // [rsp+70h] [rbp-C0h] BYREF
  __int64 v74; // [rsp+78h] [rbp-B8h]
  _BYTE v75[176]; // [rsp+80h] [rbp-B0h] BYREF

  v2 = a1;
  v3 = *(__int64 **)(a2 + 40);
  v73 = v75;
  v74 = 0x1000000000LL;
  if ( v3 == &v3[*(unsigned int *)(a2 + 48)] )
  {
    v59 = 32;
    v63 = 32;
  }
  else
  {
    v62 = &v3[*(unsigned int *)(a2 + 48)];
    v4 = v3;
    v5 = 0;
    do
    {
      v6 = *v4;
      v9 = sub_E5CAC0((__int64 *)a2, *v4);
      if ( v9 )
      {
        v10 = (unsigned int)v74;
        v11 = (unsigned int)v74 + 1LL;
        if ( v11 > HIDWORD(v74) )
        {
          sub_C8D5F0((__int64)&v73, v75, v11, 8u, v7, v8);
          v10 = (unsigned int)v74;
        }
        *(_QWORD *)&v73[8 * v10] = v5;
        LODWORD(v74) = v74 + 1;
        v5 = (v5 + v9 + 11) & 0xFFFFFFFFFFFFFFFCLL;
        if ( *(_QWORD *)(v6 + 136) == 4 && **(_DWORD **)(v6 + 128) == 1279875140 )
          v5 += 24LL;
      }
      ++v4;
    }
    while ( v62 != v4 );
    v2 = a1;
    v59 = 4 * v74 + 32;
    v63 = v5 + v59;
  }
  v12 = &v66;
  v13 = *(_QWORD *)(v2 + 104);
  v66 = 1128421444;
  for ( i = 68; ; i = *(_BYTE *)v12 )
  {
    v12 = (int *)((char *)v12 + 1);
    v65 = i;
    sub_CB6200(v13, &v65, 1u);
    if ( v12 == v67 )
      break;
  }
  v15 = (unsigned __int8 *)&v66;
  sub_CB6C70(*(_QWORD *)(v2 + 104), 0x10u);
  v16 = *(_QWORD *)(v2 + 104);
  v17 = 1;
  if ( *(_DWORD *)(v2 + 112) != 1 )
    v17 = 256;
  LOWORD(v66) = v17;
  sub_CB6200(v16, (unsigned __int8 *)&v66, 2u);
  v18 = *(_QWORD *)(v2 + 104);
  LOWORD(v66) = 0;
  sub_CB6200(v18, (unsigned __int8 *)&v66, 2u);
  v19 = *(_QWORD *)(v2 + 104);
  if ( *(_DWORD *)(v2 + 112) != 1 )
    v63 = _byteswap_ulong(v63);
  v66 = v63;
  sub_CB6200(v19, (unsigned __int8 *)&v66, 4u);
  v20 = v74;
  v21 = *(_QWORD *)(v2 + 104);
  if ( *(_DWORD *)(v2 + 112) != 1 )
    v20 = _byteswap_ulong(v74);
  v22 = &v66;
  v66 = v20;
  sub_CB6200(v21, (unsigned __int8 *)&v66, 4u);
  v23 = v73;
  v24 = &v73[8 * (unsigned int)v74];
  if ( v24 != (_DWORD *)v73 )
  {
    do
    {
      v25 = *(_QWORD *)(v2 + 104);
      v26 = v59 + *v23;
      if ( *(_DWORD *)(v2 + 112) != 1 )
        v26 = _byteswap_ulong(v26);
      v22 = &v66;
      v23 += 2;
      v66 = v26;
      sub_CB6200(v25, (unsigned __int8 *)&v66, 4u);
    }
    while ( v24 != v23 );
  }
  v27 = *(__int64 **)(a2 + 40);
  v64 = v27;
  v58 = &v27[*(unsigned int *)(a2 + 48)];
  if ( v58 != v27 )
  {
    do
    {
      v28 = *v64;
      v22 = (int *)*v64;
      v60 = sub_E5CAC0((__int64 *)a2, *v64);
      if ( v60 )
      {
        v29 = *(_QWORD **)(v2 + 104);
        v30 = (*(__int64 (__fastcall **)(_QWORD *))(*v29 + 80LL))(v29);
        v31 = *(_QWORD *)(v2 + 104);
        v55 = v28;
        v57 = v30 + v29[4] - v29[2];
        v32 = *(_QWORD *)(v28 + 128) + 4LL;
        v33 = *(char **)(v28 + 128);
        v34 = (char *)v32;
        do
        {
          v35 = *v33++;
          LOBYTE(v66) = v35;
          sub_CB6200(v31, v15, 1u);
        }
        while ( v34 != v33 );
        v36 = (__int64 *)v55;
        if ( *(_QWORD *)(v55 + 136) == 4 )
        {
          v37 = v60 + 24;
          if ( **(_DWORD **)(v55 + 128) != 1279875140 )
            v37 = v60;
        }
        else
        {
          v37 = v60;
        }
        v38 = *(_QWORD *)(v2 + 104);
        v39 = (v37 + 3) & 0xFFFFFFFC;
        if ( *(_DWORD *)(v2 + 112) != 1 )
          v39 = _byteswap_ulong(v39);
        v66 = v39;
        sub_CB6200(v38, v15, 4u);
        if ( *(_QWORD *)(v55 + 136) == 4 && **(_DWORD **)(v55 + 128) == 1279875140 )
        {
          *((_QWORD *)v15 + 2) = 0;
          *(_OWORD *)v15 = 0;
          v43 = *(_QWORD *)a2;
          v44 = (__int64 *)(*(_QWORD *)a2 + 24LL);
          v45 = sub_CC78E0((__int64)v44);
          v46 = v45;
          v47 = HIBYTE(v45);
          v48 = BYTE4(v45);
          if ( (v47 & 0x80u) == 0LL )
            v48 = 0;
          LOBYTE(v66) = (16 * v46) | v48;
          sub_CC7490(v44);
          if ( v49 )
            HIWORD(v66) = *(_DWORD *)(v43 + 72) - 33;
          v67[1] = 1279875140;
          v67[0] = (unsigned __int64)(v60 + 27) >> 2;
          v50 = sub_CC7DD0((__int64)v44);
          v56 = v15;
          v51 = *(_QWORD *)(v2 + 104);
          v69 = v50;
          v52 = HIBYTE(v50);
          v53 = BYTE4(v50);
          v70 = 16;
          v71 = v60;
          if ( (v52 & 0x80u) == 0LL )
            v53 = 0;
          v68 = v53;
          do
          {
            v54 = *v15++;
            v65 = v54;
            sub_CB6200(v51, &v65, 1u);
          }
          while ( &v72 != (char *)v15 );
          v15 = v56;
        }
        sub_E5CCC0((__int64 *)a2, *(_QWORD **)(v2 + 104), v36);
        v40 = *(_DWORD **)(v2 + 104);
        v41 = (*(unsigned int (__fastcall **)(_DWORD *))(*(_QWORD *)v40 + 80LL))(v40)
            + v40[8]
            - v40[4]
            - (unsigned int)v57;
        v22 = (int *)(((v41 + 3) & 0xFFFFFFFFFFFFFFFCLL) - v41);
        sub_CB6C70(*(_QWORD *)(v2 + 104), (unsigned int)v22);
      }
      ++v64;
    }
    while ( v58 != v64 );
  }
  if ( v73 != v75 )
    _libc_free(v73, v22);
  return 0;
}
