// Function: sub_19D7620
// Address: 0x19d7620
//
__int64 __fastcall sub_19D7620(__int64 a1, __int64 a2, __int64 a3, __int64 *a4)
{
  __int64 v4; // r13
  __int64 i; // rbx
  int v6; // eax
  void *v7; // r10
  __int64 v8; // rbx
  bool v9; // cc
  __int64 v10; // rdi
  __int64 v11; // rax
  __int64 v12; // rdi
  int v13; // eax
  __int64 v14; // r14
  __int64 v15; // rdi
  __int64 v16; // r15
  const char *v17; // rax
  __int64 v18; // rdi
  size_t v19; // rdx
  size_t v20; // r13
  const char *v21; // rax
  size_t v22; // rdx
  const char *v23; // r9
  const char *v24; // rdi
  __int64 v25; // rsi
  unsigned __int64 v26; // rdx
  __int64 v27; // r8
  unsigned int v28; // eax
  __int64 *v29; // rax
  __int64 v30; // r12
  __int64 v31; // rsi
  __int64 v32; // rsi
  __int64 v33; // rcx
  __int64 v34; // rsi
  __int64 v35; // rsi
  __int64 v36; // rbx
  int v37; // eax
  void *v38; // r10
  __int64 v39; // r13
  __int64 v40; // rdi
  __int64 v41; // rdx
  __int64 v42; // rdi
  int v43; // edx
  __int64 v44; // rdi
  const char *v45; // r8
  size_t v46; // rdx
  size_t v47; // r13
  __int64 v48; // rdi
  const char *v49; // rax
  size_t v50; // rdx
  const char *v51; // r8
  const char *v52; // rdi
  __int64 v53; // rdi
  __int64 v54; // rdi
  unsigned __int64 v56; // rdx
  __int64 v57; // rsi
  int v58; // eax
  __int64 v59; // rbx
  __int64 v60; // rdi
  __int64 v61; // rax
  __int64 v62; // rdi
  int v63; // eax
  unsigned int v64; // [rsp+4h] [rbp-10Ch]
  __int64 v65; // [rsp+8h] [rbp-108h]
  __int64 v66; // [rsp+10h] [rbp-100h]
  __int64 v67; // [rsp+18h] [rbp-F8h]
  __int64 v68; // [rsp+20h] [rbp-F0h]
  int v69; // [rsp+28h] [rbp-E8h]
  int v70; // [rsp+2Ch] [rbp-E4h]
  __int64 v71; // [rsp+30h] [rbp-E0h]
  char v73; // [rsp+38h] [rbp-D8h]
  __int64 v75; // [rsp+40h] [rbp-D0h]
  __int64 v76; // [rsp+48h] [rbp-C8h]
  __int64 v77; // [rsp+48h] [rbp-C8h]
  __int64 v79; // [rsp+58h] [rbp-B8h]
  __int64 v80; // [rsp+58h] [rbp-B8h]
  void *s1b; // [rsp+60h] [rbp-B0h]
  const char *s1; // [rsp+60h] [rbp-B0h]
  void *s1c; // [rsp+60h] [rbp-B0h]
  void *s1d; // [rsp+60h] [rbp-B0h]
  const char *s1a; // [rsp+60h] [rbp-B0h]
  void *s1e; // [rsp+60h] [rbp-B0h]
  __int64 v88; // [rsp+68h] [rbp-A8h]
  __int64 v89; // [rsp+A0h] [rbp-70h] BYREF
  int v90; // [rsp+A8h] [rbp-68h]
  __int64 v91; // [rsp+B0h] [rbp-60h]
  __int64 v92; // [rsp+B8h] [rbp-58h]
  __int64 v93; // [rsp+C0h] [rbp-50h]
  int v94; // [rsp+C8h] [rbp-48h]
  int v95; // [rsp+D0h] [rbp-40h]

  v4 = a1;
  v76 = (a3 - 1) / 2;
  if ( a2 >= v76 )
  {
    v14 = a2;
    v16 = a1 + 104 * a2;
    goto LABEL_33;
  }
  for ( i = a2; ; i = v14 )
  {
    v14 = 2 * (i + 1) - 1;
    v15 = *(_QWORD *)(a1 + 208 * (i + 1) + 32);
    v79 = a1 + 208 * (i + 1);
    v16 = a1 + 104 * v14;
    if ( v15 )
      v15 = *(_QWORD *)(v15 - 24LL * (*(_DWORD *)(v15 + 20) & 0xFFFFFFF));
    v17 = sub_1649960(v15);
    v18 = *(_QWORD *)(v16 + 32);
    v20 = v19;
    if ( v18 )
      v18 = *(_QWORD *)(v18 - 24LL * (*(_DWORD *)(v18 + 20) & 0xFFFFFFF));
    s1 = v17;
    v21 = sub_1649960(v18);
    v23 = s1;
    v7 = (void *)v22;
    if ( v20 <= v22 )
    {
      if ( v20 )
      {
        s1b = (void *)v22;
        v6 = memcmp(v23, v21, v20);
        v7 = s1b;
        if ( v6 )
          goto LABEL_22;
      }
      if ( (void *)v20 != v7 )
      {
LABEL_6:
        if ( v20 >= (unsigned __int64)v7 )
          goto LABEL_23;
        goto LABEL_7;
      }
      v25 = *(_QWORD *)(v16 + 32);
      v26 = *(_QWORD *)(v79 + 32);
      if ( v26 )
      {
        v27 = *(_QWORD *)(v26 - 24LL * (*(_DWORD *)(v26 + 20) & 0xFFFFFFF));
        if ( v25 )
        {
          if ( v27 != *(_QWORD *)(v25 - 24LL * (*(_DWORD *)(v25 + 20) & 0xFFFFFFF)) )
          {
LABEL_27:
            v26 = *(_QWORD *)(v26 - 24LL * (*(_DWORD *)(v26 + 20) & 0xFFFFFFF));
            if ( !v25 )
            {
LABEL_23:
              v16 = a1 + 208 * (i + 1);
              v14 = 2 * (i + 1);
              goto LABEL_7;
            }
            v28 = *(_DWORD *)(v25 + 20);
            goto LABEL_29;
          }
        }
        else if ( v27 )
        {
          goto LABEL_27;
        }
      }
      else if ( v25 )
      {
        v28 = *(_DWORD *)(v25 + 20);
        if ( *(_QWORD *)(v25 - 24LL * (v28 & 0xFFFFFFF)) )
        {
LABEL_29:
          LOBYTE(v28) = *(_QWORD *)(v25 - 24LL * (v28 & 0xFFFFFFF)) > v26;
          goto LABEL_30;
        }
      }
      v28 = (unsigned int)sub_16AEA10(v79 + 48, v16 + 48) >> 31;
LABEL_30:
      if ( (_BYTE)v28 )
        goto LABEL_7;
      goto LABEL_23;
    }
    if ( !v22 )
      goto LABEL_23;
    v24 = s1;
    s1c = (void *)v22;
    v6 = memcmp(v24, v21, v22);
    v7 = s1c;
    if ( !v6 )
      goto LABEL_6;
LABEL_22:
    if ( v6 >= 0 )
      goto LABEL_23;
LABEL_7:
    v8 = a1 + 104 * i;
    v9 = *(_DWORD *)(v8 + 56) <= 0x40u;
    *(_QWORD *)v8 = *(_QWORD *)v16;
    *(_QWORD *)(v8 + 8) = *(_QWORD *)(v16 + 8);
    *(_QWORD *)(v8 + 16) = *(_QWORD *)(v16 + 16);
    *(_BYTE *)(v8 + 24) = *(_BYTE *)(v16 + 24);
    *(_QWORD *)(v8 + 32) = *(_QWORD *)(v16 + 32);
    *(_QWORD *)(v8 + 40) = *(_QWORD *)(v16 + 40);
    if ( !v9 )
    {
      v10 = *(_QWORD *)(v8 + 48);
      if ( v10 )
        j_j___libc_free_0_0(v10);
    }
    *(_QWORD *)(v8 + 48) = *(_QWORD *)(v16 + 48);
    *(_DWORD *)(v8 + 56) = *(_DWORD *)(v16 + 56);
    v11 = *(_QWORD *)(v16 + 64);
    *(_DWORD *)(v16 + 56) = 0;
    v9 = *(_DWORD *)(v8 + 88) <= 0x40u;
    *(_QWORD *)(v8 + 64) = v11;
    *(_QWORD *)(v8 + 72) = *(_QWORD *)(v16 + 72);
    if ( !v9 )
    {
      v12 = *(_QWORD *)(v8 + 80);
      if ( v12 )
        j_j___libc_free_0_0(v12);
    }
    *(_QWORD *)(v8 + 80) = *(_QWORD *)(v16 + 80);
    *(_DWORD *)(v8 + 88) = *(_DWORD *)(v16 + 88);
    v13 = *(_DWORD *)(v16 + 96);
    *(_DWORD *)(v16 + 88) = 0;
    *(_DWORD *)(v8 + 96) = v13;
    if ( v14 >= v76 )
      break;
  }
  v4 = a1;
LABEL_33:
  if ( (a3 & 1) == 0 && (a3 - 2) / 2 == v14 )
  {
    v14 = 2 * v14 + 1;
    v9 = *(_DWORD *)(v16 + 56) <= 0x40u;
    v59 = v4 + 104 * v14;
    *(_QWORD *)v16 = *(_QWORD *)v59;
    *(_QWORD *)(v16 + 8) = *(_QWORD *)(v59 + 8);
    *(_QWORD *)(v16 + 16) = *(_QWORD *)(v59 + 16);
    *(_BYTE *)(v16 + 24) = *(_BYTE *)(v59 + 24);
    *(_QWORD *)(v16 + 32) = *(_QWORD *)(v59 + 32);
    *(_QWORD *)(v16 + 40) = *(_QWORD *)(v59 + 40);
    if ( !v9 )
    {
      v60 = *(_QWORD *)(v16 + 48);
      if ( v60 )
        j_j___libc_free_0_0(v60);
    }
    *(_QWORD *)(v16 + 48) = *(_QWORD *)(v59 + 48);
    *(_DWORD *)(v16 + 56) = *(_DWORD *)(v59 + 56);
    v61 = *(_QWORD *)(v59 + 64);
    *(_DWORD *)(v59 + 56) = 0;
    v9 = *(_DWORD *)(v16 + 88) <= 0x40u;
    *(_QWORD *)(v16 + 64) = v61;
    *(_QWORD *)(v16 + 72) = *(_QWORD *)(v59 + 72);
    if ( !v9 )
    {
      v62 = *(_QWORD *)(v16 + 80);
      if ( v62 )
        j_j___libc_free_0_0(v62);
    }
    *(_QWORD *)(v16 + 80) = *(_QWORD *)(v59 + 80);
    *(_DWORD *)(v16 + 88) = *(_DWORD *)(v59 + 88);
    v63 = *(_DWORD *)(v59 + 96);
    *(_DWORD *)(v59 + 88) = 0;
    *(_DWORD *)(v16 + 96) = v63;
    v16 = v4 + 104 * v14;
  }
  v29 = a4;
  v30 = a4[4];
  v80 = *a4;
  v77 = a4[1];
  v75 = a4[2];
  v31 = a4[5];
  v73 = *((_BYTE *)a4 + 24);
  v71 = v31;
  v32 = v29[6];
  v70 = *((_DWORD *)v29 + 14);
  v90 = v70;
  v33 = v29[8];
  v68 = v32;
  v89 = v32;
  v34 = v29[9];
  v67 = v33;
  v91 = v33;
  LODWORD(v33) = *((_DWORD *)v29 + 22);
  *((_DWORD *)v29 + 14) = 0;
  v66 = v34;
  v92 = v34;
  v69 = v33;
  v35 = v29[10];
  *((_DWORD *)v29 + 22) = 0;
  LODWORD(v29) = *((_DWORD *)v29 + 24);
  v94 = v33;
  v64 = (unsigned int)v29;
  v95 = (int)v29;
  v65 = v35;
  v93 = v35;
  v36 = (v14 - 1) / 2;
  if ( v14 > a2 )
  {
    v88 = v4;
    while ( 1 )
    {
      v16 = v88 + 104 * v36;
      v44 = *(_QWORD *)(v16 + 32);
      if ( v44 )
        v44 = *(_QWORD *)(v44 - 24LL * (*(_DWORD *)(v44 + 20) & 0xFFFFFFF));
      v45 = sub_1649960(v44);
      v47 = v46;
      if ( v30 )
        v48 = *(_QWORD *)(v30 - 24LL * (*(_DWORD *)(v30 + 20) & 0xFFFFFFF));
      else
        v48 = 0;
      s1a = v45;
      v49 = sub_1649960(v48);
      v51 = s1a;
      v38 = (void *)v50;
      if ( v47 > v50 )
      {
        if ( !v50 )
          goto LABEL_57;
        v52 = s1a;
        s1e = (void *)v50;
        v37 = memcmp(v52, v49, v50);
        v38 = s1e;
        if ( !v37 )
        {
LABEL_40:
          if ( v47 >= (unsigned __int64)v38 )
            goto LABEL_57;
          goto LABEL_41;
        }
LABEL_56:
        if ( v37 >= 0 )
          goto LABEL_57;
        goto LABEL_41;
      }
      if ( v47 )
      {
        s1d = (void *)v50;
        v37 = memcmp(v51, v49, v47);
        v38 = s1d;
        if ( v37 )
          goto LABEL_56;
      }
      if ( (void *)v47 != v38 )
        goto LABEL_40;
      v56 = *(_QWORD *)(v16 + 32);
      if ( v56 )
      {
        v57 = *(_QWORD *)(v56 - 24LL * (*(_DWORD *)(v56 + 20) & 0xFFFFFFF));
        if ( v30 )
        {
          if ( *(_QWORD *)(v30 - 24LL * (*(_DWORD *)(v30 + 20) & 0xFFFFFFF)) != v57 )
            goto LABEL_68;
        }
        else if ( v57 )
        {
LABEL_68:
          v56 = *(_QWORD *)(v56 - 24LL * (*(_DWORD *)(v56 + 20) & 0xFFFFFFF));
          if ( !v30 )
            goto LABEL_57;
          v58 = *(_DWORD *)(v30 + 20);
LABEL_70:
          if ( *(_QWORD *)(v30 - 24LL * (v58 & 0xFFFFFFF)) <= v56 )
            goto LABEL_57;
          goto LABEL_41;
        }
      }
      else if ( v30 )
      {
        v58 = *(_DWORD *)(v30 + 20);
        if ( *(_QWORD *)(v30 - 24LL * (v58 & 0xFFFFFFF)) )
          goto LABEL_70;
      }
      if ( (int)sub_16AEA10(v16 + 48, (__int64)&v89) >= 0 )
      {
LABEL_57:
        v16 = v88 + 104 * v14;
        break;
      }
LABEL_41:
      v39 = v88 + 104 * v14;
      *(_QWORD *)v39 = *(_QWORD *)v16;
      v9 = *(_DWORD *)(v39 + 56) <= 0x40u;
      *(_QWORD *)(v39 + 8) = *(_QWORD *)(v16 + 8);
      *(_QWORD *)(v39 + 16) = *(_QWORD *)(v16 + 16);
      *(_BYTE *)(v39 + 24) = *(_BYTE *)(v16 + 24);
      *(_QWORD *)(v39 + 32) = *(_QWORD *)(v16 + 32);
      *(_QWORD *)(v39 + 40) = *(_QWORD *)(v16 + 40);
      if ( !v9 )
      {
        v40 = *(_QWORD *)(v39 + 48);
        if ( v40 )
          j_j___libc_free_0_0(v40);
      }
      *(_QWORD *)(v39 + 48) = *(_QWORD *)(v16 + 48);
      *(_DWORD *)(v39 + 56) = *(_DWORD *)(v16 + 56);
      v41 = *(_QWORD *)(v16 + 64);
      *(_DWORD *)(v16 + 56) = 0;
      v9 = *(_DWORD *)(v39 + 88) <= 0x40u;
      *(_QWORD *)(v39 + 64) = v41;
      *(_QWORD *)(v39 + 72) = *(_QWORD *)(v16 + 72);
      if ( !v9 )
      {
        v42 = *(_QWORD *)(v39 + 80);
        if ( v42 )
          j_j___libc_free_0_0(v42);
      }
      v14 = v36;
      *(_QWORD *)(v39 + 80) = *(_QWORD *)(v16 + 80);
      *(_DWORD *)(v39 + 88) = *(_DWORD *)(v16 + 88);
      v43 = *(_DWORD *)(v16 + 96);
      *(_DWORD *)(v16 + 88) = 0;
      *(_DWORD *)(v39 + 96) = v43;
      if ( a2 >= v36 )
        break;
      v36 = (v36 - 1) / 2;
    }
  }
  v9 = *(_DWORD *)(v16 + 56) <= 0x40u;
  *(_QWORD *)(v16 + 32) = v30;
  *(_QWORD *)v16 = v80;
  *(_QWORD *)(v16 + 8) = v77;
  *(_QWORD *)(v16 + 16) = v75;
  *(_BYTE *)(v16 + 24) = v73;
  *(_QWORD *)(v16 + 40) = v71;
  if ( !v9 )
  {
    v53 = *(_QWORD *)(v16 + 48);
    if ( v53 )
      j_j___libc_free_0_0(v53);
  }
  v9 = *(_DWORD *)(v16 + 88) <= 0x40u;
  *(_QWORD *)(v16 + 48) = v68;
  *(_DWORD *)(v16 + 56) = v70;
  *(_QWORD *)(v16 + 64) = v67;
  *(_QWORD *)(v16 + 72) = v66;
  if ( !v9 )
  {
    v54 = *(_QWORD *)(v16 + 80);
    if ( v54 )
      j_j___libc_free_0_0(v54);
  }
  *(_QWORD *)(v16 + 80) = v65;
  *(_DWORD *)(v16 + 88) = v69;
  *(_DWORD *)(v16 + 96) = v64;
  return v64;
}
