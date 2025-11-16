// Function: sub_2D0DB80
// Address: 0x2d0db80
//
__int64 __fastcall sub_2D0DB80(__int64 a1)
{
  unsigned int v2; // r12d
  __int64 v3; // r13
  __int64 v4; // r14
  __int64 v5; // r15
  _QWORD *v6; // rax
  _QWORD *v7; // rdi
  _QWORD *v8; // rax
  unsigned __int64 v9; // r15
  __int64 v10; // rax
  _QWORD *v11; // r13
  _QWORD *v12; // r14
  unsigned __int64 v13; // r12
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rdi
  unsigned __int64 v16; // rdi
  unsigned __int16 v17; // ax
  __int64 v18; // rsi
  int v19; // r13d
  _BYTE *v20; // r14
  _BYTE *v21; // r12
  __int64 v22; // r9
  int v23; // r11d
  unsigned __int64 v24; // rdi
  unsigned int v25; // ecx
  unsigned __int64 v26; // rax
  __int64 v27; // r8
  __int64 v28; // rdx
  unsigned int v29; // esi
  int v30; // ecx
  _BYTE *v31; // rdi
  int v33; // eax
  unsigned int v34; // eax
  __int64 v35; // rdx
  _QWORD *v36; // rax
  _QWORD *v37; // rsi
  __int64 v38; // rcx
  _QWORD *v39; // rdx
  __int64 v40; // r13
  __int64 v41; // rdi
  _QWORD *v42; // rax
  __int64 v43; // rax
  __int64 v44; // rax
  _QWORD *v45; // rax
  _BYTE *v46; // rsi
  _BYTE *v47; // r13
  __int64 *v48; // r14
  __int64 *v49; // rax
  __int64 v50; // rax
  __int64 v51; // r15
  size_t v52; // rdx
  __int64 v53; // r13
  char *v54; // r14
  _QWORD *v55; // r15
  unsigned __int64 v56; // rdi
  unsigned __int64 v57; // r13
  _QWORD *v58; // r15
  unsigned __int64 v59; // r13
  unsigned __int64 v60; // rdi
  void *v61; // rax
  __int64 v62; // r13
  unsigned __int8 *v63; // rax
  size_t v64; // rdx
  void *v65; // rdi
  __int64 v66; // rax
  unsigned int v67; // r13d
  void *v68; // rax
  __int64 v69; // rax
  void *v70; // rax
  __int64 v71; // r13
  unsigned int v72; // ecx
  _QWORD *v73; // rax
  unsigned __int64 v74; // r14
  void *v75; // rax
  unsigned __int64 v76; // r14
  __int64 v77; // rax
  __int64 v78; // rax
  __int64 v79; // rax
  void *v80; // rax
  unsigned __int8 v81; // [rsp+18h] [rbp-B8h]
  size_t v82; // [rsp+18h] [rbp-B8h]
  _QWORD *v83; // [rsp+28h] [rbp-A8h] BYREF
  __int64 *v84; // [rsp+30h] [rbp-A0h] BYREF
  _BYTE *v85; // [rsp+38h] [rbp-98h]
  _BYTE *v86; // [rsp+40h] [rbp-90h]
  _BYTE *v87; // [rsp+50h] [rbp-80h] BYREF
  __int64 v88; // [rsp+58h] [rbp-78h]
  _BYTE v89[112]; // [rsp+60h] [rbp-70h] BYREF

  sub_2D04FA0(a1);
  v2 = (unsigned __int8)byte_5015448;
  if ( byte_5015448 )
    v2 = sub_2D074C0((__int64 *)a1);
  v3 = *(_QWORD *)(a1 + 24);
  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(_QWORD *)a1;
  v6 = (_QWORD *)sub_22077B0(0x110u);
  v7 = v6;
  if ( v6 )
  {
    *v6 = v5;
    v8 = v6 + 26;
    *(v8 - 25) = v4;
    *(v8 - 24) = v3;
    *(v8 - 23) = 0;
    *(v8 - 22) = 0;
    *(v8 - 21) = 0;
    *((_BYTE *)v8 - 160) = 0;
    *(v8 - 19) = 0;
    *(v8 - 18) = 0;
    *(v8 - 17) = 0;
    *((_DWORD *)v8 - 32) = 0;
    *(v8 - 15) = 0;
    *(v8 - 14) = 0;
    *(v8 - 13) = 0;
    *(v8 - 12) = 0;
    *(v8 - 11) = 0;
    *(v8 - 10) = 0;
    *((_DWORD *)v8 - 18) = 0;
    *(v8 - 8) = 0;
    *(v8 - 7) = 0;
    *(v8 - 6) = 0;
    *((_DWORD *)v8 - 10) = 0;
    *(v8 - 4) = 0;
    v7[23] = v8;
    v7[24] = 8;
    *((_DWORD *)v7 + 50) = 0;
    *((_BYTE *)v7 + 204) = 1;
  }
  v9 = *(_QWORD *)(a1 + 48);
  *(_QWORD *)(a1 + 48) = v7;
  if ( v9 )
  {
    if ( !*(_BYTE *)(v9 + 204) )
      _libc_free(*(_QWORD *)(v9 + 184));
    sub_C7D6A0(*(_QWORD *)(v9 + 152), 16LL * *(unsigned int *)(v9 + 168), 8);
    v10 = *(unsigned int *)(v9 + 136);
    if ( (_DWORD)v10 )
    {
      v11 = *(_QWORD **)(v9 + 120);
      v81 = v2;
      v12 = &v11[2 * v10];
      do
      {
        if ( *v11 != -4096 && *v11 != -8192 )
        {
          v13 = v11[1];
          if ( v13 )
          {
            v14 = *(_QWORD *)(v13 + 96);
            if ( v14 != v13 + 112 )
              _libc_free(v14);
            v15 = *(_QWORD *)(v13 + 24);
            if ( v15 != v13 + 40 )
              _libc_free(v15);
            j_j___libc_free_0(v13);
          }
        }
        v11 += 2;
      }
      while ( v12 != v11 );
      v2 = v81;
      LODWORD(v10) = *(_DWORD *)(v9 + 136);
    }
    sub_C7D6A0(*(_QWORD *)(v9 + 120), 16LL * (unsigned int)v10, 8);
    v16 = *(_QWORD *)(v9 + 88);
    if ( v16 )
      j_j___libc_free_0(v16);
    sub_C7D6A0(*(_QWORD *)(v9 + 64), 16LL * *(unsigned int *)(v9 + 80), 8);
    j_j___libc_free_0(v9);
    v7 = *(_QWORD **)(a1 + 48);
  }
  sub_CE6510(v7);
  sub_FD3980(*(_QWORD *)(a1 + 48), *(_QWORD *)(a1 + 40));
  if ( (_BYTE)qword_5014FC8 )
  {
    v61 = sub_CB72A0();
    v62 = sub_904010((__int64)v61, "Function: ");
    v63 = (unsigned __int8 *)sub_BD5D20(*(_QWORD *)a1);
    v65 = *(void **)(v62 + 32);
    if ( *(_QWORD *)(v62 + 24) - (_QWORD)v65 < v64 )
    {
      v62 = sub_CB6200(v62, v63, v64);
    }
    else if ( v64 )
    {
      v82 = v64;
      memcpy(v65, v63, v64);
      *(_QWORD *)(v62 + 32) += v82;
    }
    sub_904010(v62, " ");
    v66 = sub_2C741A0(*(_QWORD *)a1);
    v67 = v66;
    if ( (_DWORD)v66 )
    {
      v74 = v66;
      v75 = sub_CB72A0();
      v76 = HIDWORD(v74);
      v77 = sub_904010((__int64)v75, "Launch bounds (");
      sub_CB59D0(v77, v67);
      v78 = (__int64)sub_CB72A0();
      if ( (_DWORD)v76 )
      {
        v79 = sub_904010(v78, ", ");
        v78 = sub_CB59D0(v79, (unsigned int)v76);
      }
      sub_904010(v78, ") ");
    }
    v68 = sub_CB72A0();
    v69 = sub_904010((__int64)v68, "Register Target: ");
    v87 = *(_BYTE **)(*(_QWORD *)(a1 + 48) + 40LL);
    sub_2D04E80((int *)&v87, v69);
    if ( *(_BYTE *)(*(_QWORD *)(a1 + 48) + 48LL) )
    {
      v80 = sub_CB72A0();
      sub_904010((__int64)v80, " (This fun has per block reg target)");
    }
    v70 = sub_CB72A0();
    v71 = sub_904010((__int64)v70, " Register Pressure: ");
    v87 = *(_BYTE **)(*(_QWORD *)(a1 + 48) + 24LL);
    sub_2D04E80((int *)&v87, v71);
    sub_904010(v71, "\n");
  }
  v17 = sub_2D06360(*(_QWORD *)(a1 + 48), 1.0);
  if ( !(_BYTE)v17 && !HIBYTE(v17) )
    return v2;
  v18 = *(_QWORD *)a1;
  v19 = 0;
  v87 = v89;
  v88 = 0x800000000LL;
  sub_2D06F50((__int64)&v87, v18);
  v20 = v87;
  v21 = &v87[8 * (unsigned int)v88];
  if ( v87 != v21 )
  {
    while ( 1 )
    {
      v28 = *((_QWORD *)v21 - 1);
      v29 = *(_DWORD *)(a1 + 272);
      ++v19;
      v83 = (_QWORD *)v28;
      if ( !v29 )
        break;
      v22 = *(_QWORD *)(a1 + 256);
      v23 = 1;
      v24 = 0;
      v25 = (v29 - 1) & (((unsigned int)v28 >> 9) ^ ((unsigned int)v28 >> 4));
      v26 = v22 + 16LL * v25;
      v27 = *(_QWORD *)v26;
      if ( v28 == *(_QWORD *)v26 )
      {
LABEL_29:
        v21 -= 8;
        *(_DWORD *)(v26 + 8) = v19;
        if ( v20 == v21 )
          goto LABEL_50;
      }
      else
      {
        while ( v27 != -4096 )
        {
          if ( v27 == -8192 && !v24 )
            v24 = v26;
          v25 = (v29 - 1) & (v23 + v25);
          v26 = v22 + 16LL * v25;
          v27 = *(_QWORD *)v26;
          if ( v28 == *(_QWORD *)v26 )
            goto LABEL_29;
          ++v23;
        }
        if ( !v24 )
          v24 = v26;
        v33 = *(_DWORD *)(a1 + 264);
        ++*(_QWORD *)(a1 + 248);
        v30 = v33 + 1;
        v84 = (__int64 *)v24;
        if ( 4 * (v33 + 1) < 3 * v29 )
        {
          if ( v29 - *(_DWORD *)(a1 + 268) - v30 > v29 >> 3 )
            goto LABEL_47;
          goto LABEL_33;
        }
LABEL_32:
        v29 *= 2;
LABEL_33:
        sub_B23080(a1 + 248, v29);
        sub_B1C700(a1 + 248, (__int64 *)&v83, &v84);
        v28 = (__int64)v83;
        v24 = (unsigned __int64)v84;
        v30 = *(_DWORD *)(a1 + 264) + 1;
LABEL_47:
        *(_DWORD *)(a1 + 264) = v30;
        if ( *(_QWORD *)v24 != -4096 )
          --*(_DWORD *)(a1 + 268);
        v21 -= 8;
        *(_QWORD *)v24 = v28;
        *(_DWORD *)(v24 + 8) = 0;
        *(_DWORD *)(v24 + 8) = v19;
        if ( v20 == v21 )
          goto LABEL_50;
      }
    }
    ++*(_QWORD *)(a1 + 248);
    v84 = 0;
    goto LABEL_32;
  }
LABEL_50:
  sub_2D0D040(a1);
  v34 = sub_2D0C690(a1);
  v35 = *(unsigned int *)(a1 + 172);
  v2 = v34;
  if ( (_DWORD)v35 != *(_DWORD *)(a1 + 176) )
  {
    v36 = *(_QWORD **)(a1 + 160);
    if ( !*(_BYTE *)(a1 + 180) )
      v35 = *(unsigned int *)(a1 + 168);
    v37 = &v36[v35];
    if ( v36 == v37 )
      goto LABEL_56;
    while ( 1 )
    {
      v38 = *v36;
      v39 = v36;
      if ( *v36 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v37 == ++v36 )
        goto LABEL_56;
    }
    if ( v37 == v36 )
    {
LABEL_56:
      v40 = 0;
    }
    else
    {
      v40 = 0;
      do
      {
        v72 = *(_DWORD *)(v38 + 32);
        if ( v72 <= 0x40 )
          v40 = (unsigned int)(1 << v72) | (unsigned __int64)v40;
        v73 = v39 + 1;
        if ( v39 + 1 == v37 )
          break;
        v38 = *v73;
        for ( ++v39; *v73 >= 0xFFFFFFFFFFFFFFFELL; v39 = v73 )
        {
          if ( v37 == ++v73 )
            goto LABEL_57;
          v38 = *v73;
        }
      }
      while ( v37 != v39 );
    }
LABEL_57:
    v84 = 0;
    v41 = *(_QWORD *)a1;
    v85 = 0;
    v86 = 0;
    v42 = (_QWORD *)sub_B2BE50(v41);
    v43 = sub_BCB2E0(v42);
    v44 = sub_ACD640(v43, v40, 0);
    v45 = sub_B98A20(v44, v40);
    v46 = v85;
    v83 = v45;
    if ( v85 == v86 )
    {
      sub_914280((__int64)&v84, v85, &v83);
      v47 = v85;
    }
    else
    {
      if ( v85 )
      {
        *(_QWORD *)v85 = v45;
        v46 = v85;
      }
      v47 = v46 + 8;
      v85 = v46 + 8;
    }
    v48 = v84;
    v49 = (__int64 *)sub_B2BE50(*(_QWORD *)a1);
    v50 = sub_B9C770(v49, v48, (__int64 *)((v47 - (_BYTE *)v48) >> 3), 0, 1);
    v51 = *(_QWORD *)a1;
    v52 = 0;
    v53 = v50;
    v54 = off_4C5D0B0[0];
    if ( off_4C5D0B0[0] )
      v52 = strlen(off_4C5D0B0[0]);
    sub_B99460(v51, v54, v52, v53);
    if ( v84 )
      j_j___libc_free_0((unsigned __int64)v84);
  }
  v55 = *(_QWORD **)(a1 + 104);
  while ( (_QWORD *)(a1 + 104) != v55 )
  {
    v57 = (unsigned __int64)v55;
    v55 = (_QWORD *)*v55;
    if ( !*(_BYTE *)(v57 + 172) )
      _libc_free(*(_QWORD *)(v57 + 152));
    v56 = *(_QWORD *)(v57 + 64);
    if ( v56 != v57 + 80 )
      _libc_free(v56);
    j_j___libc_free_0(v57);
  }
  *(_QWORD *)(a1 + 112) = v55;
  *(_QWORD *)(a1 + 104) = v55;
  v58 = *(_QWORD **)(a1 + 128);
  *(_QWORD *)(a1 + 120) = 0;
  while ( (_QWORD *)(a1 + 128) != v58 )
  {
    v59 = (unsigned __int64)v58;
    v58 = (_QWORD *)*v58;
    v60 = *(_QWORD *)(v59 + 152);
    if ( v60 != v59 + 168 )
      _libc_free(v60);
    if ( !*(_BYTE *)(v59 + 84) )
      _libc_free(*(_QWORD *)(v59 + 64));
    j_j___libc_free_0(v59);
  }
  *(_QWORD *)(a1 + 136) = v58;
  v31 = v87;
  *(_QWORD *)(a1 + 128) = v58;
  *(_QWORD *)(a1 + 144) = 0;
  if ( v31 != v89 )
    _libc_free((unsigned __int64)v31);
  return v2;
}
