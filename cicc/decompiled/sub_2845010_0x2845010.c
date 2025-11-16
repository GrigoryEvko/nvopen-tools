// Function: sub_2845010
// Address: 0x2845010
//
__int64 __fastcall sub_2845010(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  _QWORD *v9; // rbx
  __int64 v10; // rax
  _QWORD *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rcx
  __int64 v14; // r8
  _QWORD *v15; // rsi
  _BYTE *v16; // rdi
  _QWORD *v17; // rbx
  __int64 v18; // rax
  unsigned __int64 v19; // rsi
  __int64 **v20; // rdi
  __int64 **v21; // r12
  __int64 v22; // rax
  int v23; // edx
  __int64 v24; // rsi
  __int64 v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 **v31; // rdx
  __int64 v32; // rcx
  void **v33; // rax
  int v34; // eax
  void **v35; // rax
  _QWORD *v37; // rbx
  _QWORD *v38; // r13
  __int64 v39; // rax
  _QWORD *v40; // rbx
  _QWORD *v41; // r13
  __int64 v42; // rax
  __int64 *v43; // rax
  __int64 **v44; // rsi
  _QWORD *v45; // [rsp+0h] [rbp-6B0h]
  __int64 v46; // [rsp+18h] [rbp-698h]
  int v47; // [rsp+2Ch] [rbp-684h]
  _BYTE v48[80]; // [rsp+30h] [rbp-680h] BYREF
  _QWORD *v49; // [rsp+80h] [rbp-630h] BYREF
  __int64 **v50; // [rsp+88h] [rbp-628h]
  __int64 v51; // [rsp+90h] [rbp-620h]
  _BYTE v52[4]; // [rsp+98h] [rbp-618h] BYREF
  char v53; // [rsp+9Ch] [rbp-614h]
  char v54[16]; // [rsp+A0h] [rbp-610h] BYREF
  __int64 v55; // [rsp+B0h] [rbp-600h] BYREF
  void **v56; // [rsp+B8h] [rbp-5F8h]
  unsigned int v57; // [rsp+C4h] [rbp-5ECh]
  int v58; // [rsp+C8h] [rbp-5E8h]
  char v59; // [rsp+CCh] [rbp-5E4h]
  char v60[328]; // [rsp+D0h] [rbp-5E0h] BYREF
  __int64 v61; // [rsp+218h] [rbp-498h] BYREF
  _BYTE *v62; // [rsp+220h] [rbp-490h]
  __int64 v63; // [rsp+228h] [rbp-488h]
  int v64; // [rsp+230h] [rbp-480h]
  char v65; // [rsp+234h] [rbp-47Ch]
  _BYTE v66[64]; // [rsp+238h] [rbp-478h] BYREF
  _BYTE *v67; // [rsp+278h] [rbp-438h] BYREF
  __int64 v68; // [rsp+280h] [rbp-430h]
  _BYTE v69[200]; // [rsp+288h] [rbp-428h] BYREF
  int v70; // [rsp+350h] [rbp-360h] BYREF
  __int64 v71; // [rsp+358h] [rbp-358h]
  int *v72; // [rsp+360h] [rbp-350h]
  int *v73; // [rsp+368h] [rbp-348h]
  __int64 v74; // [rsp+370h] [rbp-340h]
  _QWORD v75[102]; // [rsp+380h] [rbp-330h] BYREF

  v6 = a2;
  v9 = (_QWORD *)a5;
  if ( *(_BYTE *)a2 && (a2 = 18, !(unsigned __int8)sub_B2D610(*(_QWORD *)(**(_QWORD **)(a3 + 32) + 72LL), 18))
    || (unsigned int)sub_F6E730(a3, a2, a3, a4, a5, a6) == 5 )
  {
    v47 = qword_50005E8;
  }
  else
  {
    v47 = 0;
  }
  v10 = sub_AA4E30(**(_QWORD **)(a3 + 32));
  sub_1002C90((__int64)v48, v9, v10);
  memset(v75, 0, 0x300u);
  v11 = (_QWORD *)v9[9];
  if ( v11 )
  {
    v50 = (__int64 **)v52;
    v49 = v11;
    v51 = 0x1000000000LL;
    v67 = v69;
    v75[0] = v11;
    v75[2] = 0x1000000000LL;
    v68 = 0x800000000LL;
    v72 = &v70;
    v73 = &v70;
    v75[1] = &v75[3];
    v61 = 0;
    v62 = v66;
    v63 = 8;
    v64 = 0;
    v65 = 1;
    v70 = 0;
    v71 = 0;
    v74 = 0;
    sub_C8CF70((__int64)&v75[51], &v75[55], 8, (__int64)v66, (__int64)&v61);
    v75[63] = &v75[65];
    v75[64] = 0x800000000LL;
    if ( (_DWORD)v68 )
      sub_2844B50((__int64)&v75[63], (__int64)&v67, v12, v13, v14, (__int64)&v70);
    if ( v71 )
    {
      v75[91] = v71;
      LODWORD(v75[90]) = v70;
      v75[92] = v72;
      v75[93] = v73;
      *(_QWORD *)(v71 + 8) = &v75[90];
      v71 = 0;
      v75[94] = v74;
      v72 = &v70;
      v73 = &v70;
      v74 = 0;
    }
    else
    {
      LODWORD(v75[90]) = 0;
      v75[91] = 0;
      v75[92] = &v75[90];
      v75[93] = &v75[90];
      v75[94] = 0;
    }
    LOBYTE(v75[95]) = 1;
    sub_2844850(0);
    v15 = v67;
    v16 = &v67[24 * (unsigned int)v68];
    if ( v67 != v16 )
    {
      v45 = v9;
      v17 = &v67[24 * (unsigned int)v68];
      do
      {
        v18 = *(v17 - 1);
        v17 -= 3;
        if ( v18 != 0 && v18 != -4096 && v18 != -8192 )
          sub_BD60C0(v17);
      }
      while ( v15 != v17 );
      v9 = v45;
      v16 = v67;
    }
    if ( v16 != v69 )
      _libc_free((unsigned __int64)v16);
    if ( !v65 )
      _libc_free((unsigned __int64)v62);
    v19 = (unsigned __int64)v50;
    v20 = &v50[3 * (unsigned int)v51];
    if ( v50 != v20 )
    {
      v46 = a1;
      v21 = &v50[3 * (unsigned int)v51];
      do
      {
        v22 = (__int64)*(v21 - 1);
        v21 -= 3;
        if ( v22 != 0 && v22 != -4096 && v22 != -8192 )
          sub_BD60C0(v21);
      }
      while ( (__int64 **)v19 != v21 );
      a1 = v46;
      v20 = v50;
    }
    if ( v20 != (__int64 **)v52 )
      _libc_free((unsigned __int64)v20);
    v23 = 1;
    if ( !*(_BYTE *)(v6 + 1) )
      v23 = (unsigned __int8)qword_5000508;
    v11 = 0;
    if ( LOBYTE(v75[95]) )
      v11 = v75;
  }
  else
  {
    v23 = 1;
    if ( !*(_BYTE *)(v6 + 1) )
      v23 = (unsigned __int8)qword_5000508;
  }
  v24 = v9[3];
  if ( !(unsigned __int8)sub_2A0FF30(a3, v24, v9[6], v9[1], v9[2], v9[4], (__int64)v11, (__int64)v48, 0, v47, 0, v23) )
  {
    *(_QWORD *)(a1 + 8) = a1 + 32;
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
    goto LABEL_57;
  }
  if ( v9[9] && byte_4F8F8E8[0] )
  {
    v24 = 0;
    nullsub_390();
  }
  sub_22D0390((__int64)&v49, v24, v25, v26, v27, v28);
  if ( v9[9] )
  {
    if ( v59 )
    {
      v31 = (__int64 **)&v56[v57];
      v32 = v57;
      if ( v56 == (void **)v31 )
      {
LABEL_83:
        v34 = v58;
      }
      else
      {
        v33 = v56;
        while ( *v33 != &unk_4F8F810 )
        {
          if ( v31 == (__int64 **)++v33 )
            goto LABEL_83;
        }
        v31 = (__int64 **)v56[--v57];
        *v33 = v31;
        v32 = v57;
        ++v55;
        v34 = v58;
      }
    }
    else
    {
      v43 = sub_C8CA60((__int64)&v55, (__int64)&unk_4F8F810);
      if ( v43 )
      {
        *v43 = -2;
        ++v55;
        v32 = v57;
        v34 = ++v58;
      }
      else
      {
        v32 = v57;
        v34 = v58;
      }
    }
    if ( (_DWORD)v32 == v34 )
    {
      if ( v53 )
      {
        v35 = (void **)v50;
        v44 = &v50[HIDWORD(v51)];
        v32 = HIDWORD(v51);
        v31 = v50;
        if ( v50 != v44 )
        {
          while ( *v31 != &qword_4F82400 )
          {
            if ( v44 == ++v31 )
            {
LABEL_52:
              while ( *v35 != &unk_4F8F810 )
              {
                if ( ++v35 == (void **)v31 )
                  goto LABEL_54;
              }
              goto LABEL_77;
            }
          }
          goto LABEL_77;
        }
        goto LABEL_54;
      }
      if ( sub_C8CA60((__int64)&v49, (__int64)&qword_4F82400) )
        goto LABEL_77;
    }
    if ( !v53 )
      goto LABEL_82;
    v35 = (void **)v50;
    v32 = HIDWORD(v51);
    v31 = &v50[HIDWORD(v51)];
    if ( v31 != v50 )
      goto LABEL_52;
LABEL_54:
    if ( (unsigned int)v32 < (unsigned int)v51 )
    {
      HIDWORD(v51) = v32 + 1;
      *v31 = (__int64 *)&unk_4F8F810;
      v49 = (_QWORD *)((char *)v49 + 1);
      goto LABEL_77;
    }
LABEL_82:
    sub_C8CC70((__int64)&v49, (__int64)&unk_4F8F810, (__int64)v31, v32, v29, v30);
  }
LABEL_77:
  sub_C8CF70(a1, (void *)(a1 + 32), 2, (__int64)v54, (__int64)&v49);
  sub_C8CF70(a1 + 48, (void *)(a1 + 80), 2, (__int64)v60, (__int64)&v55);
  if ( !v59 )
    _libc_free((unsigned __int64)v56);
  if ( !v53 )
    _libc_free((unsigned __int64)v50);
LABEL_57:
  if ( LOBYTE(v75[95]) )
  {
    LOBYTE(v75[95]) = 0;
    sub_2844850((_QWORD *)v75[91]);
    v37 = (_QWORD *)v75[63];
    v38 = (_QWORD *)(v75[63] + 24LL * LODWORD(v75[64]));
    if ( (_QWORD *)v75[63] != v38 )
    {
      do
      {
        v39 = *(v38 - 1);
        v38 -= 3;
        if ( v39 != 0 && v39 != -4096 && v39 != -8192 )
          sub_BD60C0(v38);
      }
      while ( v37 != v38 );
      v38 = (_QWORD *)v75[63];
    }
    if ( v38 != &v75[65] )
      _libc_free((unsigned __int64)v38);
    if ( !BYTE4(v75[54]) )
      _libc_free(v75[52]);
    v40 = (_QWORD *)v75[1];
    v41 = (_QWORD *)(v75[1] + 24LL * LODWORD(v75[2]));
    if ( (_QWORD *)v75[1] != v41 )
    {
      do
      {
        v42 = *(v41 - 1);
        v41 -= 3;
        if ( v42 != 0 && v42 != -4096 && v42 != -8192 )
          sub_BD60C0(v41);
      }
      while ( v40 != v41 );
      v41 = (_QWORD *)v75[1];
    }
    if ( v41 != &v75[3] )
      _libc_free((unsigned __int64)v41);
  }
  return a1;
}
