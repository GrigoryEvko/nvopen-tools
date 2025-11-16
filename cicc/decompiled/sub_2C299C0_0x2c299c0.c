// Function: sub_2C299C0
// Address: 0x2c299c0
//
void __fastcall sub_2C299C0(_QWORD *a1, char a2, char a3)
{
  __int64 v5; // rax
  _QWORD *v6; // rcx
  __int64 v7; // rax
  _QWORD *v8; // rsi
  __int64 v9; // rdx
  __int64 v10; // rax
  _QWORD *v11; // rax
  _QWORD *v12; // rdx
  const char *v13; // rbx
  __int64 v14; // rax
  __int64 v15; // rax
  __int64 (__fastcall *v16)(__int64); // rax
  __int64 v17; // r13
  __int64 v18; // rax
  const char *v19; // rbx
  const char *v20; // rax
  __int64 v21; // rax
  __int64 v22; // r9
  __int64 v23; // rcx
  _QWORD *v24; // rbx
  __int64 v25; // rax
  const char *v26; // rdx
  const char *v27; // rax
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rax
  __int64 v32; // rdx
  const char *v33; // rax
  const char *v34; // rax
  __int64 *v35; // rax
  __int64 v36; // rcx
  const char *v37; // rax
  const char *v38; // rdi
  const char *v39; // r13
  __int64 v40; // r12
  bool v41; // zf
  __int64 *v42; // rbx
  __int64 v43; // rdi
  __int64 v44; // rax
  __int64 v45; // [rsp+8h] [rbp-F8h]
  const char *v46; // [rsp+18h] [rbp-E8h]
  __int64 v47; // [rsp+20h] [rbp-E0h]
  const char *v48; // [rsp+28h] [rbp-D8h]
  __int64 v49; // [rsp+28h] [rbp-D8h]
  _QWORD *v50; // [rsp+28h] [rbp-D8h]
  __int64 v51; // [rsp+28h] [rbp-D8h]
  const char *v52; // [rsp+30h] [rbp-D0h]
  __int64 v53; // [rsp+30h] [rbp-D0h]
  const char *v54; // [rsp+30h] [rbp-D0h]
  __int64 v55; // [rsp+30h] [rbp-D0h]
  const char *v56; // [rsp+30h] [rbp-D0h]
  __int64 v57; // [rsp+30h] [rbp-D0h]
  __int64 v58; // [rsp+38h] [rbp-C8h]
  __int64 *v59; // [rsp+38h] [rbp-C8h]
  const char *v60; // [rsp+50h] [rbp-B0h] BYREF
  const char *v61; // [rsp+58h] [rbp-A8h] BYREF
  const char *v62; // [rsp+60h] [rbp-A0h] BYREF
  const char *v63; // [rsp+68h] [rbp-98h] BYREF
  __int64 v64; // [rsp+70h] [rbp-90h] BYREF
  __int64 *v65; // [rsp+78h] [rbp-88h]
  const char *v66; // [rsp+80h] [rbp-80h] BYREF
  const char *v67; // [rsp+88h] [rbp-78h]
  const char *v68; // [rsp+90h] [rbp-70h] BYREF
  int v69; // [rsp+98h] [rbp-68h]
  char v70; // [rsp+A0h] [rbp-60h] BYREF
  __int16 v71; // [rsp+B0h] [rbp-50h]

  v5 = sub_2AAFF80((__int64)a1);
  v6 = *(_QWORD **)(v5 + 112);
  v7 = 8LL * *(unsigned int *)(v5 + 120);
  v8 = &v6[(unsigned __int64)v7 / 8];
  v9 = v7 >> 3;
  v10 = v7 >> 5;
  if ( v10 )
  {
    v11 = &v6[4 * v10];
    while ( 1 )
    {
      v12 = (_QWORD *)*v6;
      if ( *(_BYTE *)(*v6 - 32LL) == 15 )
        goto LABEL_13;
      v12 = (_QWORD *)v6[1];
      if ( *((_BYTE *)v12 - 32) == 15 )
        goto LABEL_13;
      v12 = (_QWORD *)v6[2];
      if ( *((_BYTE *)v12 - 32) == 15 )
        goto LABEL_13;
      v12 = (_QWORD *)v6[3];
      if ( *((_BYTE *)v12 - 32) == 15 )
        goto LABEL_13;
      v6 += 4;
      if ( v6 == v11 )
      {
        v9 = v8 - v6;
        break;
      }
    }
  }
  if ( v9 == 2 )
  {
LABEL_86:
    v12 = (_QWORD *)*v6;
    if ( *(_BYTE *)(*v6 - 32LL) == 15 )
      goto LABEL_13;
    ++v6;
    goto LABEL_88;
  }
  if ( v9 == 3 )
  {
    v12 = (_QWORD *)*v6;
    if ( *(_BYTE *)(*v6 - 32LL) == 15 )
      goto LABEL_13;
    ++v6;
    goto LABEL_86;
  }
  if ( v9 != 1 )
    goto LABEL_12;
LABEL_88:
  v12 = (_QWORD *)*v6;
  if ( *(_BYTE *)(*v6 - 32LL) != 15 )
  {
LABEL_12:
    v12 = (_QWORD *)*v8;
    if ( !*v8 )
      goto LABEL_14;
  }
LABEL_13:
  v12 -= 5;
LABEL_14:
  if ( !a2 )
  {
    v35 = (__int64 *)v12[4];
    v36 = v12[10];
    v66 = (const char *)(v12 + 12);
    v65 = v35;
    v68 = "active.lane.mask";
    v37 = (const char *)a1[25];
    v64 = v36;
    v71 = 259;
    v67 = v37;
    v62 = 0;
    v63 = 0;
    v24 = (_QWORD *)sub_2AAFFE0(&v64, 73, (__int64 *)&v66, 2, (__int64 *)&v63, (void **)&v68);
    sub_9C6650(&v63);
    v24[17] = 0;
    sub_9C6650(&v62);
    goto LABEL_60;
  }
  v13 = 0;
  v14 = sub_2BF3F10(a1);
  v47 = sub_2BF0520(v14);
  v15 = sub_2AAFF80((__int64)a1);
  v58 = v15;
  if ( *(_DWORD *)(v15 + 56) )
    v13 = **(const char ***)(v15 + 48);
  v16 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)v15 + 40LL);
  if ( v16 == sub_2AA7530 )
    v17 = *(_QWORD *)(*(_QWORD *)(v58 + 48) + 8LL);
  else
    v17 = v16(v58);
  if ( !v17 )
  {
    sub_2C26CD0(0);
    BUG();
  }
  sub_2C26CD0(v17 - 96);
  v60 = *(const char **)(v17 - 8);
  if ( v60 )
    sub_2C25AB0((__int64 *)&v60);
  v18 = sub_2BF3F10(a1);
  if ( v18 )
  {
    if ( *(_DWORD *)(v18 + 64) == 1 )
      v18 = **(_QWORD **)(v18 + 56);
    else
      v18 = 0;
  }
  v64 = v18;
  v65 = (__int64 *)(v18 + 112);
  v48 = (const char *)a1[25];
  if ( a3 )
  {
    v71 = 257;
    v63 = v60;
    v46 = (const char *)(v58 + 96);
    if ( v60 )
    {
      sub_2C25AB0((__int64 *)&v63);
      v62 = v48;
      v66 = v63;
      if ( v63 )
        sub_2C25AB0((__int64 *)&v66);
    }
    else
    {
      v66 = 0;
      v62 = v48;
    }
    v57 = sub_2AAFFE0(&v64, 76, (__int64 *)&v62, 1, (__int64 *)&v66, (void **)&v68);
    sub_9C6650(&v66);
    v44 = v57 + 96;
    if ( !v57 )
      v44 = 0;
    v52 = (const char *)v44;
    sub_9C6650(&v63);
  }
  else
  {
    v52 = (const char *)a1[25];
    v46 = (const char *)v17;
  }
  v68 = "index.part.next";
  v71 = 259;
  v66 = v60;
  if ( v60 )
    sub_2C25AB0((__int64 *)&v66);
  v63 = v13;
  v19 = (const char *)sub_2C28020(&v64, 77, (__int64 *)&v63, 1, 0, (__int64 *)&v66, (void **)&v68);
  sub_9C6650(&v66);
  v68 = "active.lane.mask.entry";
  v20 = v60;
  v71 = 259;
  v62 = v60;
  if ( v60 )
  {
    sub_2C25AB0((__int64 *)&v62);
    v20 = v62;
    if ( !v19 )
      goto LABEL_32;
  }
  else if ( !v19 )
  {
    v66 = 0;
    v63 = 0;
    v67 = v48;
    goto LABEL_34;
  }
  v19 += 96;
LABEL_32:
  v66 = v19;
  v63 = v20;
  v67 = v48;
  if ( v20 )
    sub_2C25AB0((__int64 *)&v63);
LABEL_34:
  v49 = sub_2AAFFE0(&v64, 73, (__int64 *)&v66, 2, (__int64 *)&v63, (void **)&v68);
  sub_9C6650(&v63);
  sub_9C6650(&v62);
  v66 = 0;
  v21 = sub_22077B0(0x98u);
  v23 = v49;
  v24 = (_QWORD *)v21;
  if ( v21 )
  {
    v50 = (_QWORD *)v21;
    if ( v23 )
      v23 += 96;
    v68 = v66;
    if ( v66 )
    {
      v45 = v23;
      sub_2C25AB0((__int64 *)&v68);
      v23 = v45;
    }
    sub_2AAFAC0((__int64)v24, 30, 0, v23, (__int64 *)&v68, v22);
    sub_9C6650(&v68);
    *v24 = &unk_4A24DB8;
    v24[5] = &unk_4A24E00;
    v24[12] = &unk_4A24E38;
  }
  else
  {
    v50 = 0;
  }
  sub_9C6650(&v66);
  sub_2C19DE0(v50, v58);
  v59 = (__int64 *)sub_2BF0A50(v47);
  v25 = v59[10];
  v71 = 257;
  v64 = v25;
  v65 = v59 + 3;
  v66 = v60;
  if ( v60 )
    sub_2C25AB0((__int64 *)&v66);
  v63 = v46;
  v51 = sub_2C28020(&v64, 77, (__int64 *)&v63, 1, 0, (__int64 *)&v66, (void **)&v68);
  sub_9C6650(&v66);
  v26 = (const char *)v51;
  v68 = "active.lane.mask.next";
  v27 = v60;
  v71 = 259;
  v62 = v60;
  if ( v60 )
  {
    sub_2C25AB0((__int64 *)&v62);
    v26 = (const char *)v51;
    v27 = v62;
    if ( !v51 )
      goto LABEL_45;
  }
  else if ( !v51 )
  {
    v66 = 0;
    v63 = 0;
    v67 = v52;
    goto LABEL_47;
  }
  v26 += 96;
LABEL_45:
  v66 = v26;
  v63 = v27;
  v67 = v52;
  if ( v27 )
    sub_2C25AB0((__int64 *)&v63);
LABEL_47:
  v53 = sub_2AAFFE0(&v64, 73, (__int64 *)&v66, 2, (__int64 *)&v63, (void **)&v68);
  sub_9C6650(&v63);
  sub_9C6650(&v62);
  v31 = v53;
  v32 = v53 + 96;
  if ( v53 )
    v31 = v53 + 96;
  v54 = (const char *)v31;
  sub_2AAECA0((__int64)(v24 + 5), v31, v32, v28, v29, v30);
  v33 = v54;
  v71 = 257;
  v61 = v60;
  if ( v60 )
  {
    sub_2C25AB0((__int64 *)&v61);
    v33 = v54;
    v63 = v61;
    if ( v61 )
    {
      sub_2C25AB0((__int64 *)&v63);
      v66 = v63;
      v62 = v54;
      if ( v63 )
        sub_2C25AB0((__int64 *)&v66);
      goto LABEL_53;
    }
  }
  else
  {
    v63 = 0;
  }
  v62 = v33;
  v66 = 0;
LABEL_53:
  v55 = sub_2AAFFE0(&v64, 70, (__int64 *)&v62, 1, (__int64 *)&v66, (void **)&v68);
  sub_9C6650(&v66);
  v34 = (const char *)v55;
  if ( v55 )
    v34 = (const char *)(v55 + 96);
  v56 = v34;
  sub_9C6650(&v63);
  sub_9C6650(&v61);
  v71 = 257;
  v63 = v60;
  if ( v60 )
  {
    sub_2C25AB0((__int64 *)&v63);
    v66 = v63;
    v62 = v56;
    if ( v63 )
      sub_2C25AB0((__int64 *)&v66);
  }
  else
  {
    v62 = v56;
    v66 = 0;
  }
  sub_2AAFFE0(&v64, 79, (__int64 *)&v62, 1, (__int64 *)&v66, (void **)&v68);
  sub_9C6650(&v66);
  sub_9C6650(&v63);
  sub_2C19E60(v59);
  sub_9C6650(&v60);
LABEL_60:
  sub_2C295B0((__int64)&v68, a1);
  v38 = v68;
  v39 = &v68[8 * v69];
  if ( v39 != v68 )
  {
    v40 = (__int64)(v24 + 12);
    v41 = v24 == 0;
    v42 = (__int64 *)v68;
    if ( v41 )
      v40 = 0;
    do
    {
      v43 = *v42++;
      sub_2BF1250(v43, v40);
    }
    while ( v39 != (const char *)v42 );
    v38 = v68;
  }
  if ( v38 != &v70 )
    _libc_free((unsigned __int64)v38);
}
