// Function: sub_250D880
// Address: 0x250d880
//
__int64 __fastcall sub_250D880(__int64 a1)
{
  unsigned __int64 v2; // rdi
  _QWORD *v3; // r13
  _QWORD *v4; // r12
  __int64 v5; // rax
  __int64 v6; // rax
  unsigned __int64 v7; // rdi
  unsigned __int64 v8; // rdi
  _QWORD *v9; // r13
  _QWORD *v10; // r12
  __int64 v11; // rax
  __int64 v12; // rax
  _QWORD *v13; // r13
  _QWORD *v14; // r12
  __int64 v15; // rax
  __int64 v16; // rax
  unsigned __int64 v17; // rdi
  unsigned __int64 v18; // rdi
  unsigned __int64 v19; // rdi
  unsigned __int64 v20; // rdi
  unsigned __int64 v21; // rdi
  __int64 v22; // rax
  __int64 v23; // r13
  __int64 v24; // r14
  unsigned __int64 v25; // r15
  _QWORD *v26; // r12
  void (__fastcall *v27)(_QWORD *, _QWORD *, __int64); // rax
  void (__fastcall *v28)(_QWORD *, _QWORD *, __int64); // rax
  unsigned __int64 v29; // rdi
  __int64 v30; // rax
  __int64 v31; // r15
  __int64 v32; // r14
  __int64 v33; // r13
  unsigned __int64 v34; // r12
  void (__fastcall *v35)(unsigned __int64, unsigned __int64, __int64); // rax
  __int64 v36; // rax
  __int64 v37; // r15
  __int64 v38; // r14
  __int64 v39; // r13
  unsigned __int64 v40; // r12
  void (__fastcall *v41)(unsigned __int64, unsigned __int64, __int64); // rax
  __int64 v42; // rax
  __int64 *v43; // r13
  __int64 v44; // r15
  __int64 v45; // rax
  __int64 v46; // r12
  __int64 v47; // rdi
  unsigned __int64 v48; // r14
  void (__fastcall *v49)(unsigned __int64, unsigned __int64, __int64); // rax
  _QWORD *v51; // r12
  _QWORD *v52; // r13
  __int64 v53; // rax
  _QWORD *v54; // r12
  _QWORD *v55; // r13
  __int64 v56; // rax
  _QWORD *v57; // r12
  _QWORD *v58; // r13
  __int64 v59; // rax
  __int64 *v60; // r12
  __int64 *v61; // r13
  __int64 v62; // rcx
  __int64 v63; // rdx
  __int64 v64; // rsi
  __int64 v65; // rax
  void (__fastcall ***v66)(_QWORD, __int64, __int64, __int64); // rdi
  __int64 v67; // rax
  __int64 v68; // [rsp+0h] [rbp-90h]
  __int64 v69; // [rsp+8h] [rbp-88h]
  __int64 v70; // [rsp+10h] [rbp-80h]
  __int64 v71; // [rsp+18h] [rbp-78h]
  __int64 *v72; // [rsp+18h] [rbp-78h]
  __int64 v73; // [rsp+20h] [rbp-70h] BYREF
  __int64 v74; // [rsp+28h] [rbp-68h]
  __int64 v75; // [rsp+30h] [rbp-60h]
  __int64 v76; // [rsp+40h] [rbp-50h] BYREF
  __int64 v77; // [rsp+48h] [rbp-48h]
  __int64 v78; // [rsp+50h] [rbp-40h]

  if ( !*(_DWORD *)(a1 + 152) )
    goto LABEL_2;
  v60 = *(__int64 **)(a1 + 144);
  v61 = &v60[4 * *(unsigned int *)(a1 + 160)];
  v62 = unk_4FEE4D0;
  v63 = qword_4FEE4C0[0];
  v64 = qword_4FEE4C0[1];
  if ( v60 == v61 )
    goto LABEL_2;
  v65 = *v60;
  if ( *v60 == -4096 )
    goto LABEL_150;
LABEL_134:
  if ( v65 != -8192 || v60[1] != qword_4FEE4C0[0] || v60[2] != v64 )
  {
    while ( 1 )
    {
LABEL_135:
      if ( v61 == v60 )
        goto LABEL_2;
      v66 = (void (__fastcall ***)(_QWORD, __int64, __int64, __int64))v60[3];
      v60 += 4;
      (**v66)(v66, v64, v63, v62);
      v63 = unk_4FEE4D0;
      v64 = unk_4FEE4D8;
      v62 = qword_4FEE4C0[0];
      if ( v60 == v61 )
        goto LABEL_2;
      v67 = *v60;
      if ( *v60 == -4096 )
        goto LABEL_143;
      while ( v67 == -8192 && v60[1] == qword_4FEE4C0[0] && v60[2] == qword_4FEE4C0[1] )
      {
        while ( 1 )
        {
          v60 += 4;
          if ( v61 == v60 )
            goto LABEL_2;
          v67 = *v60;
          if ( *v60 != -4096 )
            break;
LABEL_143:
          if ( v60[1] != unk_4FEE4D0 || v60[2] != unk_4FEE4D8 )
            goto LABEL_135;
        }
      }
    }
  }
  while ( 1 )
  {
    v60 += 4;
    if ( v61 == v60 )
      break;
    v65 = *v60;
    if ( *v60 != -4096 )
      goto LABEL_134;
LABEL_150:
    if ( v60[1] != unk_4FEE4D0 || v60[2] != unk_4FEE4D8 )
      goto LABEL_135;
  }
LABEL_2:
  sub_A17130(a1 + 4416);
  sub_A17130(a1 + 4336);
  sub_A17130(a1 + 4304);
  v2 = *(_QWORD *)(a1 + 4152);
  if ( v2 != a1 + 4168 )
    _libc_free(v2);
  sub_C7D6A0(*(_QWORD *)(a1 + 4128), 8LL * *(unsigned int *)(a1 + 4144), 8);
  v3 = *(_QWORD **)(a1 + 3912);
  v4 = &v3[3 * *(unsigned int *)(a1 + 3920)];
  if ( v3 != v4 )
  {
    do
    {
      v5 = *(v4 - 1);
      v4 -= 3;
      if ( v5 != 0 && v5 != -4096 && v5 != -8192 )
        sub_BD60C0(v4);
    }
    while ( v3 != v4 );
    v4 = *(_QWORD **)(a1 + 3912);
  }
  if ( v4 != (_QWORD *)(a1 + 3928) )
    _libc_free((unsigned __int64)v4);
  v6 = *(unsigned int *)(a1 + 3904);
  if ( (_DWORD)v6 )
  {
    v57 = *(_QWORD **)(a1 + 3888);
    v73 = 4;
    v74 = 0;
    v75 = -4096;
    v58 = &v57[3 * v6];
    v76 = 4;
    v77 = 0;
    v78 = -8192;
    do
    {
      v59 = v57[2];
      if ( v59 != -4096 && v59 != 0 && v59 != -8192 )
        sub_BD60C0(v57);
      v57 += 3;
    }
    while ( v58 != v57 );
    sub_D68D70(&v76);
    sub_D68D70(&v73);
    v6 = *(unsigned int *)(a1 + 3904);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 3888), 24 * v6, 8);
  v7 = *(_QWORD *)(a1 + 3800);
  if ( v7 != a1 + 3816 )
    _libc_free(v7);
  sub_C7D6A0(*(_QWORD *)(a1 + 3776), 8LL * *(unsigned int *)(a1 + 3792), 8);
  v8 = *(_QWORD *)(a1 + 3688);
  if ( v8 != a1 + 3704 )
    _libc_free(v8);
  sub_C7D6A0(*(_QWORD *)(a1 + 3664), 8LL * *(unsigned int *)(a1 + 3680), 8);
  if ( !*(_BYTE *)(a1 + 3588) )
    _libc_free(*(_QWORD *)(a1 + 3568));
  v9 = *(_QWORD **)(a1 + 3152);
  v10 = &v9[3 * *(unsigned int *)(a1 + 3160)];
  if ( v9 != v10 )
  {
    do
    {
      v11 = *(v10 - 1);
      v10 -= 3;
      if ( v11 != 0 && v11 != -4096 && v11 != -8192 )
        sub_BD60C0(v10);
    }
    while ( v9 != v10 );
    v10 = *(_QWORD **)(a1 + 3152);
  }
  if ( v10 != (_QWORD *)(a1 + 3168) )
    _libc_free((unsigned __int64)v10);
  v12 = *(unsigned int *)(a1 + 3144);
  if ( (_DWORD)v12 )
  {
    v54 = *(_QWORD **)(a1 + 3128);
    v73 = 4;
    v74 = 0;
    v75 = -4096;
    v55 = &v54[3 * v12];
    v76 = 4;
    v77 = 0;
    v78 = -8192;
    do
    {
      v56 = v54[2];
      if ( v56 != -4096 && v56 != 0 && v56 != -8192 )
        sub_BD60C0(v54);
      v54 += 3;
    }
    while ( v55 != v54 );
    sub_D68D70(&v76);
    sub_D68D70(&v73);
    v12 = *(unsigned int *)(a1 + 3144);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 3128), 24 * v12, 8);
  v13 = *(_QWORD **)(a1 + 2720);
  v14 = &v13[3 * *(unsigned int *)(a1 + 2728)];
  if ( v13 != v14 )
  {
    do
    {
      v15 = *(v14 - 1);
      v14 -= 3;
      if ( v15 != 0 && v15 != -4096 && v15 != -8192 )
        sub_BD60C0(v14);
    }
    while ( v13 != v14 );
    v14 = *(_QWORD **)(a1 + 2720);
  }
  if ( v14 != (_QWORD *)(a1 + 2736) )
    _libc_free((unsigned __int64)v14);
  v16 = *(unsigned int *)(a1 + 2712);
  if ( (_DWORD)v16 )
  {
    v51 = *(_QWORD **)(a1 + 2696);
    v73 = 4;
    v74 = 0;
    v75 = -4096;
    v52 = &v51[3 * v16];
    v76 = 4;
    v77 = 0;
    v78 = -8192;
    do
    {
      v53 = v51[2];
      if ( v53 != -4096 && v53 != 0 && v53 != -8192 )
        sub_BD60C0(v51);
      v51 += 3;
    }
    while ( v52 != v51 );
    sub_D68D70(&v76);
    sub_D68D70(&v73);
    v16 = *(unsigned int *)(a1 + 2712);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 2696), 24 * v16, 8);
  v17 = *(_QWORD *)(a1 + 2160);
  if ( v17 != a1 + 2176 )
    _libc_free(v17);
  if ( (*(_BYTE *)(a1 + 1640) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(a1 + 1648), 16LL * *(unsigned int *)(a1 + 1656), 8);
  v18 = *(_QWORD *)(a1 + 1104);
  if ( v18 != a1 + 1120 )
    _libc_free(v18);
  if ( (*(_BYTE *)(a1 + 584) & 1) == 0 )
    sub_C7D6A0(*(_QWORD *)(a1 + 592), 16LL * *(unsigned int *)(a1 + 600), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 552), 8LL * *(unsigned int *)(a1 + 568), 8);
  v19 = *(_QWORD *)(a1 + 400);
  if ( v19 != a1 + 416 )
    _libc_free(v19);
  v20 = *(_QWORD *)(a1 + 320);
  if ( v20 != a1 + 336 )
    _libc_free(v20);
  sub_C7D6A0(*(_QWORD *)(a1 + 296), 8LL * *(unsigned int *)(a1 + 312), 8);
  v21 = *(_QWORD *)(a1 + 256);
  *(_QWORD *)(a1 + 216) = &unk_4A16C00;
  if ( v21 != a1 + 272 )
    _libc_free(v21);
  sub_C7D6A0(*(_QWORD *)(a1 + 232), 8LL * *(unsigned int *)(a1 + 248), 8);
  v22 = *(unsigned int *)(a1 + 192);
  if ( (_DWORD)v22 )
  {
    v23 = *(_QWORD *)(a1 + 176);
    v71 = v23 + 88 * v22;
    do
    {
      if ( *(_QWORD *)v23 != -8192 && *(_QWORD *)v23 != -4096 )
      {
        v24 = *(_QWORD *)(v23 + 8);
        v25 = v24 + 8LL * *(unsigned int *)(v23 + 16);
        if ( v24 != v25 )
        {
          do
          {
            v26 = *(_QWORD **)(v25 - 8);
            v25 -= 8LL;
            if ( v26 )
            {
              v27 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v26[19];
              if ( v27 )
                v27(v26 + 17, v26 + 17, 3);
              v28 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v26[15];
              if ( v28 )
                v28(v26 + 13, v26 + 13, 3);
              v29 = v26[3];
              if ( (_QWORD *)v29 != v26 + 5 )
                _libc_free(v29);
              j_j___libc_free_0((unsigned __int64)v26);
            }
          }
          while ( v24 != v25 );
          v25 = *(_QWORD *)(v23 + 8);
        }
        if ( v25 != v23 + 24 )
          _libc_free(v25);
      }
      v23 += 88;
    }
    while ( v71 != v23 );
    v22 = *(unsigned int *)(a1 + 192);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 176), 88 * v22, 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 144), 32LL * *(unsigned int *)(a1 + 160), 8);
  v30 = *(unsigned int *)(a1 + 120);
  if ( (_DWORD)v30 )
  {
    v31 = *(_QWORD *)(a1 + 104);
    v32 = v31 + 56 * v30;
    do
    {
      if ( *(_QWORD *)v31 != -8192 && *(_QWORD *)v31 != -4096 )
      {
        v33 = *(_QWORD *)(v31 + 8);
        v34 = v33 + 32LL * *(unsigned int *)(v31 + 16);
        if ( v33 != v34 )
        {
          do
          {
            v35 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(v34 - 16);
            v34 -= 32LL;
            if ( v35 )
              v35(v34, v34, 3);
          }
          while ( v33 != v34 );
          v34 = *(_QWORD *)(v31 + 8);
        }
        if ( v34 != v31 + 24 )
          _libc_free(v34);
      }
      v31 += 56;
    }
    while ( v32 != v31 );
    v30 = *(unsigned int *)(a1 + 120);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 104), 56 * v30, 8);
  v36 = *(unsigned int *)(a1 + 88);
  if ( (_DWORD)v36 )
  {
    v37 = *(_QWORD *)(a1 + 72);
    v38 = v37 + 56 * v36;
    do
    {
      if ( *(_QWORD *)v37 != -4096 && *(_QWORD *)v37 != -8192 )
      {
        v39 = *(_QWORD *)(v37 + 8);
        v40 = v39 + 32LL * *(unsigned int *)(v37 + 16);
        if ( v39 != v40 )
        {
          do
          {
            v41 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(v40 - 16);
            v40 -= 32LL;
            if ( v41 )
              v41(v40, v40, 3);
          }
          while ( v39 != v40 );
          v40 = *(_QWORD *)(v37 + 8);
        }
        if ( v40 != v37 + 24 )
          _libc_free(v40);
      }
      v37 += 56;
    }
    while ( v38 != v37 );
    v36 = *(unsigned int *)(a1 + 88);
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 72), 56 * v36, 8);
  v42 = *(unsigned int *)(a1 + 56);
  if ( (_DWORD)v42 )
  {
    v43 = *(__int64 **)(a1 + 40);
    v44 = unk_4FEE4D0;
    v72 = &v43[8 * v42];
    v69 = unk_4FEE4D8;
    v70 = qword_4FEE4C0[0];
    v68 = qword_4FEE4C0[1];
    while ( 1 )
    {
      v45 = *v43;
      if ( v44 == *v43 )
      {
        if ( v69 == v43[1] )
          goto LABEL_107;
        if ( v70 != v45 )
          goto LABEL_100;
      }
      else if ( v70 != v45 )
      {
        goto LABEL_100;
      }
      if ( v68 != v43[1] )
      {
LABEL_100:
        v46 = v43[2];
        v47 = 32LL * *((unsigned int *)v43 + 6);
        v48 = v46 + v47;
        if ( v46 != v46 + v47 )
        {
          do
          {
            v49 = *(void (__fastcall **)(unsigned __int64, unsigned __int64, __int64))(v48 - 16);
            v48 -= 32LL;
            if ( v49 )
              v49(v48, v48, 3);
          }
          while ( v46 != v48 );
          v48 = v43[2];
        }
        if ( (__int64 *)v48 != v43 + 4 )
          _libc_free(v48);
      }
LABEL_107:
      v43 += 8;
      if ( v72 == v43 )
      {
        LODWORD(v42) = *(_DWORD *)(a1 + 56);
        break;
      }
    }
  }
  sub_C7D6A0(*(_QWORD *)(a1 + 40), (unsigned __int64)(unsigned int)v42 << 6, 8);
  return sub_C7D6A0(*(_QWORD *)(a1 + 8), 16LL * *(unsigned int *)(a1 + 24), 8);
}
