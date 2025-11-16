// Function: sub_26282B0
// Address: 0x26282b0
//
void __fastcall sub_26282B0(__int64 *a1)
{
  __int64 *v1; // rbx
  __int64 v2; // r15
  __int64 v3; // r12
  __int64 v4; // r14
  __int64 v5; // rax
  unsigned __int64 v6; // r13
  __int64 v7; // rcx
  __int64 v8; // r12
  __int64 v9; // rax
  int v10; // esi
  __int64 v11; // rax
  unsigned __int64 v12; // rax
  int v13; // edi
  __int64 v14; // rdx
  __int64 v15; // rdx
  __int64 v16; // rdx
  unsigned __int64 v17; // rcx
  unsigned __int64 v18; // rsi
  unsigned __int64 v19; // rbx
  unsigned __int64 v20; // r14
  unsigned __int64 v21; // rdi
  __int64 v22; // rax
  unsigned __int64 v23; // r14
  unsigned __int64 v24; // r15
  unsigned __int64 v25; // rdi
  unsigned __int64 v26; // r12
  char *v27; // r15
  __int64 v28; // rax
  __int64 v29; // rcx
  char **v30; // r8
  int v31; // r15d
  _QWORD *v32; // r14
  __int64 **v33; // r13
  __int64 v34; // r12
  unsigned __int64 v35; // rax
  __int64 v36; // rax
  _BYTE *v37; // rax
  char *v38; // r13
  size_t v39; // r12
  __int64 *v40; // rax
  __int64 **v41; // rax
  __int64 v42; // rax
  _QWORD *v43; // r12
  __int64 v44; // r14
  __int64 v45; // rdx
  int v46; // r14d
  __int64 v47; // r15
  __int64 v48; // r9
  _QWORD *v49; // rdi
  __int64 v50; // rax
  __int64 v51; // r12
  __int64 v52; // rax
  __int64 v53; // rdi
  __int64 v54; // rdi
  __int64 v55; // r8
  unsigned __int64 v56; // rdx
  __int64 v57; // [rsp+0h] [rbp-100h]
  __int64 v58; // [rsp+0h] [rbp-100h]
  __int64 *v59; // [rsp+8h] [rbp-F8h]
  __int64 v60; // [rsp+10h] [rbp-F0h]
  char *v61; // [rsp+18h] [rbp-E8h]
  __int64 v62; // [rsp+18h] [rbp-E8h]
  __int64 v63; // [rsp+20h] [rbp-E0h]
  unsigned __int64 *v64; // [rsp+20h] [rbp-E0h]
  unsigned __int8 *v65; // [rsp+20h] [rbp-E0h]
  char **v66; // [rsp+28h] [rbp-D8h]
  char *v67; // [rsp+28h] [rbp-D8h]
  __int64 v68[2]; // [rsp+30h] [rbp-D0h] BYREF
  char *v69; // [rsp+40h] [rbp-C0h] BYREF
  unsigned int v70; // [rsp+48h] [rbp-B8h]
  unsigned __int64 v71; // [rsp+50h] [rbp-B0h]
  unsigned int v72; // [rsp+58h] [rbp-A8h]
  __int16 v73; // [rsp+60h] [rbp-A0h]
  void *src; // [rsp+70h] [rbp-90h] BYREF
  __int64 v75; // [rsp+78h] [rbp-88h]
  __int64 v76; // [rsp+80h] [rbp-80h]
  __int128 v77; // [rsp+88h] [rbp-78h]
  __int128 v78; // [rsp+98h] [rbp-68h]
  __int128 v79; // [rsp+A8h] [rbp-58h]
  __int128 v80; // [rsp+B8h] [rbp-48h]

  v1 = a1;
  v2 = a1[20];
  v61 = (char *)a1[21];
  v3 = 0xCCCCCCCCCCCCCCCDLL * ((__int64)&v61[-v2] >> 4);
  if ( (__int64)&v61[-v2] <= 0 )
  {
LABEL_52:
    v6 = 0;
    sub_2620680(v2, v61);
  }
  else
  {
    while ( 1 )
    {
      v4 = 80 * v3;
      v5 = sub_2207800(80 * v3);
      v6 = v5;
      if ( v5 )
        break;
      v3 >>= 1;
      if ( !v3 )
        goto LABEL_52;
    }
    v60 = v3;
    v7 = v5 + 8;
    v8 = v5 + v4;
    v63 = v2 + 8;
    v9 = *(_QWORD *)(v2 + 16);
    if ( v9 )
    {
      v10 = *(_DWORD *)(v2 + 8);
      *(_QWORD *)(v6 + 16) = v9;
      *(_DWORD *)(v6 + 8) = v10;
      *(_QWORD *)(v6 + 24) = *(_QWORD *)(v2 + 24);
      *(_QWORD *)(v6 + 32) = *(_QWORD *)(v2 + 32);
      *(_QWORD *)(v9 + 8) = v7;
      v11 = *(_QWORD *)(v2 + 40);
      *(_QWORD *)(v2 + 16) = 0;
      *(_QWORD *)(v6 + 40) = v11;
      *(_QWORD *)(v2 + 40) = 0;
      *(_QWORD *)(v2 + 24) = v63;
      *(_QWORD *)(v2 + 32) = v63;
    }
    else
    {
      *(_DWORD *)(v6 + 8) = 0;
      *(_QWORD *)(v6 + 16) = 0;
      *(_QWORD *)(v6 + 24) = v7;
      *(_QWORD *)(v6 + 32) = v7;
      *(_QWORD *)(v6 + 40) = 0;
    }
    *(_QWORD *)(v6 + 48) = *(_QWORD *)(v2 + 48);
    *(_QWORD *)(v6 + 56) = *(_QWORD *)(v2 + 56);
    *(_QWORD *)(v6 + 64) = *(_QWORD *)(v2 + 64);
    *(_QWORD *)(v6 + 72) = *(_QWORD *)(v2 + 72);
    v12 = v6 + 80;
    if ( v8 == v6 + 80 )
    {
      v56 = v6;
    }
    else
    {
      do
      {
        v16 = *(_QWORD *)(v12 - 64);
        v17 = v12 - 72;
        v18 = v12 + 8;
        if ( v16 )
        {
          v13 = *(_DWORD *)(v12 - 72);
          *(_QWORD *)(v12 + 16) = v16;
          *(_DWORD *)(v12 + 8) = v13;
          *(_QWORD *)(v12 + 24) = *(_QWORD *)(v12 - 56);
          *(_QWORD *)(v12 + 32) = *(_QWORD *)(v12 - 48);
          *(_QWORD *)(v16 + 8) = v18;
          v14 = *(_QWORD *)(v12 - 40);
          *(_QWORD *)(v12 - 64) = 0;
          *(_QWORD *)(v12 + 40) = v14;
          *(_QWORD *)(v12 - 56) = v17;
          *(_QWORD *)(v12 - 48) = v17;
          *(_QWORD *)(v12 - 40) = 0;
        }
        else
        {
          *(_DWORD *)(v12 + 8) = 0;
          *(_QWORD *)(v12 + 16) = 0;
          *(_QWORD *)(v12 + 24) = v18;
          *(_QWORD *)(v12 + 32) = v18;
          *(_QWORD *)(v12 + 40) = 0;
        }
        v15 = *(_QWORD *)(v12 - 32);
        v12 += 80LL;
        *(_QWORD *)(v12 - 32) = v15;
        *(_QWORD *)(v12 - 24) = *(_QWORD *)(v12 - 104);
        *(_QWORD *)(v12 - 16) = *(_QWORD *)(v12 - 96);
        *(_QWORD *)(v12 - 8) = *(_QWORD *)(v12 - 88);
      }
      while ( v8 != v12 );
      v56 = v6 + 16 * (5 * ((0xCCCCCCCCCCCCCCDLL * ((unsigned __int64)(v4 - 160) >> 4)) & 0xFFFFFFFFFFFFFFFLL) + 5);
    }
    if ( *(_QWORD *)(v2 + 16) )
    {
      v59 = v1;
      v19 = *(_QWORD *)(v2 + 16);
      v20 = v56;
      do
      {
        sub_261DCB0(*(_QWORD *)(v19 + 24));
        v21 = v19;
        v19 = *(_QWORD *)(v19 + 16);
        j_j___libc_free_0(v21);
      }
      while ( v19 );
      v1 = v59;
      v56 = v20;
    }
    *(_QWORD *)(v2 + 16) = 0;
    *(_QWORD *)(v2 + 40) = 0;
    *(_QWORD *)(v2 + 24) = v63;
    *(_QWORD *)(v2 + 32) = v63;
    if ( *(_QWORD *)(v56 + 16) )
    {
      *(_DWORD *)(v2 + 8) = *(_DWORD *)(v56 + 8);
      v22 = *(_QWORD *)(v56 + 16);
      *(_QWORD *)(v2 + 16) = v22;
      *(_QWORD *)(v2 + 24) = *(_QWORD *)(v56 + 24);
      *(_QWORD *)(v2 + 32) = *(_QWORD *)(v56 + 32);
      *(_QWORD *)(v22 + 8) = v63;
      *(_QWORD *)(v2 + 40) = *(_QWORD *)(v56 + 40);
      *(_QWORD *)(v56 + 16) = 0;
      *(_QWORD *)(v56 + 24) = v56 + 8;
      *(_QWORD *)(v56 + 32) = v56 + 8;
      *(_QWORD *)(v56 + 40) = 0;
    }
    v23 = v6;
    *(_QWORD *)(v2 + 48) = *(_QWORD *)(v56 + 48);
    *(_QWORD *)(v2 + 56) = *(_QWORD *)(v56 + 56);
    *(_QWORD *)(v2 + 64) = *(_QWORD *)(v56 + 64);
    *(_QWORD *)(v2 + 72) = *(_QWORD *)(v56 + 72);
    sub_2624B30(v2, v61, v6, v60);
    do
    {
      v24 = *(_QWORD *)(v23 + 16);
      while ( v24 )
      {
        sub_261DCB0(*(_QWORD *)(v24 + 24));
        v25 = v24;
        v24 = *(_QWORD *)(v24 + 16);
        j_j___libc_free_0(v25);
      }
      v23 += 80LL;
    }
    while ( v8 != v23 );
  }
  j_j___libc_free_0(v6);
  if ( v1[21] - v1[20] < 0 )
    sub_4262D8((__int64)"cannot create std::vector larger than max_size()");
  if ( 0xCCCCCCCCCCCCCCCDLL * ((v1[21] - v1[20]) >> 4) )
  {
    v26 = 0x6666666666666668LL * ((v1[21] - v1[20]) >> 4);
    v27 = (char *)sub_22077B0(v26);
    if ( v27 != &v27[v26] )
      memset(v27, 0, v26);
  }
  else
  {
    v27 = 0;
  }
  v76 = 0;
  v28 = v1[20];
  src = 0;
  v75 = 0;
  v77 = 0;
  v78 = 0;
  v79 = 0;
  v80 = 0;
  if ( v1[21] == v28 )
  {
    v39 = 0;
    v38 = 0;
  }
  else
  {
    v29 = 0;
    v30 = &v69;
    v64 = (unsigned __int64 *)v27;
    v31 = 0;
    do
    {
      v66 = v30;
      v32 = (_QWORD *)(v28 + 80 * v29);
      sub_26281E0((__int64 *)&src, (__int64)v32, v32[6], &v64[v29], v30);
      v33 = (__int64 **)v1[9];
      v34 = v32[8];
      v35 = sub_ACD640(v1[8], (unsigned __int8)v69, 0);
      v36 = sub_AD4C70(v35, v33, 0);
      sub_BD84D0(v34, v36);
      sub_B30290(v32[8]);
      v37 = (_BYTE *)v32[9];
      v30 = v66;
      if ( v37 )
        *v37 = (_BYTE)v69;
      v28 = v1[20];
      v29 = (unsigned int)++v31;
    }
    while ( v31 != 0xCCCCCCCCCCCCCCCDLL * ((v1[21] - v28) >> 4) );
    v38 = (char *)src;
    v27 = (char *)v64;
    v39 = v75 - (_QWORD)src;
  }
  v40 = (__int64 *)sub_BCD140(*(_QWORD **)*v1, 8u);
  v41 = (__int64 **)sub_BCD420(v40, v39);
  v42 = sub_AC9630(v38, v39, v41);
  v43 = *(_QWORD **)(v42 + 8);
  v62 = v42;
  v44 = v42;
  v73 = 257;
  BYTE4(v68[0]) = 0;
  v65 = (unsigned __int8 *)sub_BD2C40(88, unk_3F0FAE8);
  if ( v65 )
    sub_B30000((__int64)v65, *v1, v43, 1, 8, v44, (__int64)&v69, 0, 0, v68[0], 0);
  v45 = v1[20];
  if ( v45 != v1[21] )
  {
    v46 = 0;
    v67 = v27;
    v47 = 0;
    do
    {
      v51 = v45 + 80 * v47;
      v52 = sub_ACD640(v1[14], 0, 0);
      v53 = v1[14];
      v68[0] = v52;
      v68[1] = sub_ACD640(v53, *(_QWORD *)&v67[8 * v47], 0);
      v54 = *(_QWORD *)(v62 + 8);
      LOBYTE(v73) = 0;
      v55 = sub_AD9FD0(v54, v65, v68, 2, 3u, (__int64)&v69, 0);
      if ( (_BYTE)v73 )
      {
        LOBYTE(v73) = 0;
        if ( v72 > 0x40 && v71 )
        {
          v57 = v55;
          j_j___libc_free_0_0(v71);
          v55 = v57;
        }
        if ( v70 > 0x40 && v69 )
        {
          v58 = v55;
          j_j___libc_free_0_0((unsigned __int64)v69);
          v55 = v58;
        }
      }
      v48 = *v1;
      v49 = (_QWORD *)v1[8];
      v69 = "bits";
      v47 = (unsigned int)++v46;
      v73 = 259;
      v50 = sub_B30500(v49, 0, 8, (__int64)&v69, v55, v48);
      sub_BD84D0(*(_QWORD *)(v51 + 56), v50);
      sub_B30290(*(_QWORD *)(v51 + 56));
      v45 = v1[20];
    }
    while ( v46 != 0xCCCCCCCCCCCCCCCDLL * ((v1[21] - v45) >> 4) );
    v27 = v67;
  }
  if ( src )
    j_j___libc_free_0((unsigned __int64)src);
  if ( v27 )
    j_j___libc_free_0((unsigned __int64)v27);
}
