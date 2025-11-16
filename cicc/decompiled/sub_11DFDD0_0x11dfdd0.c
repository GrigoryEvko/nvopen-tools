// Function: sub_11DFDD0
// Address: 0x11dfdd0
//
__int64 __fastcall sub_11DFDD0(__int64 a1, __int64 a2, char a3, __int64 a4)
{
  int v7; // edx
  __int64 v8; // rdx
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // r15
  __int64 v12; // r10
  __int64 v14; // rax
  __int64 v15; // rdi
  __int64 v16; // r15
  __int64 v17; // rax
  char v18; // bl
  _QWORD *v19; // rax
  __int64 v20; // r13
  unsigned int *v21; // r14
  __int64 v22; // rbx
  __int64 v23; // rdx
  unsigned int v24; // esi
  __int64 v25; // rax
  char v26; // bl
  _QWORD *v27; // rax
  __int64 v28; // r9
  __int64 v29; // r14
  unsigned int *v30; // r15
  __int64 v31; // rbx
  __int64 v32; // rdx
  unsigned int v33; // esi
  _BYTE *v34; // rax
  __int64 v35; // r13
  __int64 v36; // rax
  __int64 v37; // rax
  unsigned __int64 v38; // rax
  unsigned __int64 v39; // rax
  unsigned __int64 v40; // r10
  const void *v41; // r8
  unsigned __int64 v42; // r10
  __int64 v43; // r14
  char *v44; // rax
  __int64 v45; // rax
  __int64 *v46; // r15
  _QWORD **v47; // rbx
  unsigned int v48; // eax
  __int64 v49; // rax
  __int64 v50; // rax
  unsigned __int8 *v51; // rax
  unsigned __int64 v52; // rax
  __int64 v53; // r13
  __int64 v54; // rax
  char *v55; // rax
  _QWORD *v56; // rdi
  __int64 v57; // rax
  unsigned __int16 v58; // ax
  unsigned int v59; // ebx
  __int64 v60; // rax
  __int64 v61; // rax
  __int64 v62; // rbx
  __int64 v63; // r12
  __int64 v64; // rax
  __int64 *v65; // rsi
  unsigned __int64 v66; // rax
  bool v67; // zf
  char *v68; // rax
  char *v69; // rdi
  __int64 v70; // [rsp-10h] [rbp-120h]
  void *src; // [rsp+8h] [rbp-108h]
  unsigned __int64 v72; // [rsp+10h] [rbp-100h]
  unsigned __int64 v73; // [rsp+10h] [rbp-100h]
  __int64 v74; // [rsp+18h] [rbp-F8h]
  unsigned __int64 v75; // [rsp+18h] [rbp-F8h]
  unsigned __int64 v76; // [rsp+18h] [rbp-F8h]
  unsigned __int64 v77; // [rsp+20h] [rbp-F0h]
  unsigned __int64 v78; // [rsp+20h] [rbp-F0h]
  __int64 v80; // [rsp+38h] [rbp-D8h]
  _BYTE *v81; // [rsp+40h] [rbp-D0h] BYREF
  size_t n; // [rsp+48h] [rbp-C8h]
  char *v83; // [rsp+50h] [rbp-C0h] BYREF
  signed __int64 v84; // [rsp+58h] [rbp-B8h]
  _QWORD v85[2]; // [rsp+60h] [rbp-B0h] BYREF
  char v86; // [rsp+70h] [rbp-A0h]
  char v87; // [rsp+71h] [rbp-9Fh]
  __m128i v88; // [rsp+80h] [rbp-90h] BYREF
  __int64 v89; // [rsp+90h] [rbp-80h]
  __int64 v90; // [rsp+98h] [rbp-78h] BYREF
  __int64 v91; // [rsp+A0h] [rbp-70h]
  __int64 v92; // [rsp+A8h] [rbp-68h]
  __int64 v93; // [rsp+B0h] [rbp-60h]
  __int64 v94; // [rsp+B8h] [rbp-58h]
  __int16 v95; // [rsp+C0h] [rbp-50h]

  v7 = *(_DWORD *)(a2 + 4);
  v88.m128i_i64[1] = 0;
  v8 = v7 & 0x7FFFFFF;
  v89 = 0;
  v90 = 0;
  v91 = 0;
  v92 = 0;
  v9 = *(_QWORD *)(a2 - 32 * v8);
  v93 = 0;
  v94 = 0;
  v80 = v9;
  v10 = *(_QWORD *)(a2 + 32 * (1 - v8));
  v11 = *(_QWORD *)(a2 + 32 * (2 - v8));
  v88.m128i_i64[0] = *(_QWORD *)(a1 + 16);
  v95 = 257;
  if ( (unsigned __int8)sub_9B6260(v11, &v88, 0) )
  {
    v88.m128i_i32[0] = 0;
    sub_11DA4B0(a2, v88.m128i_i32, 1);
    v88.m128i_i32[0] = 1;
    sub_11DA4B0(a2, v88.m128i_i32, 1);
  }
  if ( *(_BYTE *)v11 != 17 )
  {
    v12 = -1;
LABEL_20:
    v77 = v12;
    v38 = sub_98B430(v10, 8u);
    if ( !v38 )
      return 0;
    v75 = v38;
    v88.m128i_i32[0] = 1;
    sub_11DA2E0(a2, (unsigned int *)&v88, 1, v38);
    v39 = v75;
    v40 = v77;
    v76 = v75 - 1;
    if ( !v76 )
    {
      v83 = *(char **)(a2 + 72);
      v88.m128i_i64[0] = sub_A744E0(&v83, 0);
      v58 = sub_A73630(v88.m128i_i64);
      if ( !HIBYTE(v58) )
        LOBYTE(v58) = 0;
      v59 = (unsigned __int8)v58;
      v60 = sub_BCB2B0(*(_QWORD **)(a4 + 72));
      BYTE1(v59) = 1;
      v61 = sub_ACD640(v60, 0, 0);
      v62 = sub_B34240(a4, v80, v61, v11, v59, 0, 0, 0, 0);
      v83 = *(char **)(a2 + 72);
      v63 = sub_A744E0(&v83, 0);
      v64 = sub_BD5C60(a2);
      sub_A74940((__int64)&v88, v64, v63);
      v83 = *(char **)(v62 + 72);
      v65 = (__int64 *)sub_BD5C60(a2);
      v66 = sub_A7B2C0((__int64 *)&v83, v65, 1, (__int64)&v88);
      v67 = *(_BYTE *)v62 == 85;
      *(_QWORD *)(v62 + 72) = v66;
      if ( v67 )
        *(_WORD *)(v62 + 2) = *(_WORD *)(v62 + 2) & 0xFFFC | *(_WORD *)(a2 + 2) & 3;
      if ( (__int64 *)v88.m128i_i64[1] != &v90 )
        _libc_free(v88.m128i_i64[1], v65);
      return v80;
    }
    if ( v39 >= v77 )
    {
LABEL_32:
      v46 = *(__int64 **)(a1 + 24);
      v78 = v40;
      v47 = (_QWORD **)sub_B43CA0(a2);
      v48 = sub_97FA80(*v46, (__int64)v47);
      v49 = sub_BCCE00(*v47, v48);
      v50 = sub_ACD640(v49, v78, 0);
      v51 = (unsigned __int8 *)sub_B343C0(a4, 0xEEu, v80, 0x100u, v10, 0x100u, v50, 0, 0, 0, 0, 0);
      sub_11DAF00(v51, a2);
      if ( a3 )
      {
        v52 = v76;
        if ( v76 > v78 )
          v52 = v78;
        v53 = v52;
        v54 = sub_BCB2E0(*(_QWORD **)(a4 + 72));
        v55 = (char *)sub_ACD640(v54, v53, 0);
        v56 = *(_QWORD **)(a4 + 72);
        v83 = v55;
        v88.m128i_i64[0] = (__int64)"endptr";
        LOWORD(v91) = 259;
        v57 = sub_BCB2B0(v56);
        return sub_921130((unsigned int **)a4, v57, v80, &v83, 1, (__int64)&v88, 3u);
      }
      return v80;
    }
    if ( v77 > 0x80 )
      return 0;
    v81 = 0;
    n = 0;
    if ( !(unsigned __int8)sub_98B0F0(v10, &v81, 1u) )
      return 0;
    v41 = v81;
    v42 = v77;
    if ( !v81 )
    {
      v84 = 0;
      v83 = (char *)v85;
      LOBYTE(v85[0]) = 0;
LABEL_30:
      v72 = v42;
      sub_22410F0(&v83, v42, 0);
      v88.m128i_i64[0] = (__int64)"str";
      LOWORD(v91) = 259;
      v45 = sub_B33830(a4, v83, v84, (__int64)&v88, 0, 0, 0);
      v40 = v72;
      v10 = v45;
      if ( v83 != (char *)v85 )
      {
        j_j___libc_free_0(v83, v85[0] + 1LL);
        v40 = v72;
      }
      goto LABEL_32;
    }
    v43 = n;
    v83 = (char *)v85;
    v88.m128i_i64[0] = n;
    if ( n > 0xF )
    {
      src = v81;
      v68 = (char *)sub_22409D0(&v83, &v88, 0);
      v42 = v77;
      v41 = src;
      v83 = v68;
      v69 = v68;
      v85[0] = v88.m128i_i64[0];
    }
    else
    {
      if ( n == 1 )
      {
        LOBYTE(v85[0]) = *v81;
        v44 = (char *)v85;
LABEL_29:
        v84 = v43;
        v44[v43] = 0;
        goto LABEL_30;
      }
      if ( !n )
      {
        v44 = (char *)v85;
        goto LABEL_29;
      }
      v69 = (char *)v85;
    }
    v73 = v42;
    memcpy(v69, v41, v43);
    v43 = v88.m128i_i64[0];
    v44 = v83;
    v42 = v73;
    goto LABEL_29;
  }
  v12 = *(_QWORD *)(v11 + 24);
  if ( *(_DWORD *)(v11 + 32) > 0x40u )
    v12 = *(_QWORD *)v12;
  if ( v12 )
  {
    if ( v12 == 1 )
    {
      v14 = sub_BCB2B0(*(_QWORD **)(a4 + 72));
      v15 = *(_QWORD *)(a4 + 48);
      v87 = 1;
      v74 = v14;
      v16 = v14;
      v83 = "stxncpy.char0";
      v86 = 3;
      v17 = sub_AA4E30(v15);
      v18 = sub_AE5020(v17, v16);
      LOWORD(v91) = 257;
      v19 = sub_BD2C40(80, unk_3F10A14);
      v20 = (__int64)v19;
      if ( v19 )
        sub_B4D190((__int64)v19, v16, v10, (__int64)&v88, 0, v18, 0, 0);
      (*(void (__fastcall **)(_QWORD, __int64, char **, _QWORD, _QWORD))(**(_QWORD **)(a4 + 88) + 16LL))(
        *(_QWORD *)(a4 + 88),
        v20,
        &v83,
        *(_QWORD *)(a4 + 56),
        *(_QWORD *)(a4 + 64));
      v21 = *(unsigned int **)a4;
      v22 = *(_QWORD *)a4 + 16LL * *(unsigned int *)(a4 + 8);
      if ( *(_QWORD *)a4 != v22 )
      {
        do
        {
          v23 = *((_QWORD *)v21 + 1);
          v24 = *v21;
          v21 += 4;
          sub_B99FD0(v20, v24, v23);
        }
        while ( (unsigned int *)v22 != v21 );
      }
      v25 = sub_AA4E30(*(_QWORD *)(a4 + 48));
      v26 = sub_AE5020(v25, *(_QWORD *)(v20 + 8));
      LOWORD(v91) = 257;
      v27 = sub_BD2C40(80, unk_3F10A10);
      v29 = (__int64)v27;
      if ( v27 )
      {
        sub_B4D3C0((__int64)v27, v20, v80, 0, v26, v28, 0, 0);
        v28 = v70;
      }
      (*(void (__fastcall **)(_QWORD, __int64, __m128i *, _QWORD, _QWORD, __int64))(**(_QWORD **)(a4 + 88) + 16LL))(
        *(_QWORD *)(a4 + 88),
        v29,
        &v88,
        *(_QWORD *)(a4 + 56),
        *(_QWORD *)(a4 + 64),
        v28);
      v30 = *(unsigned int **)a4;
      v31 = *(_QWORD *)a4 + 16LL * *(unsigned int *)(a4 + 8);
      if ( *(_QWORD *)a4 != v31 )
      {
        do
        {
          v32 = *((_QWORD *)v30 + 1);
          v33 = *v30;
          v30 += 4;
          sub_B99FD0(v29, v33, v32);
        }
        while ( (unsigned int *)v31 != v30 );
      }
      if ( a3 )
      {
        v34 = (_BYTE *)sub_AD64C0(v74, 0, 0);
        v88.m128i_i64[0] = (__int64)"stpncpy.char0cmp";
        LOWORD(v91) = 259;
        v35 = sub_92B530((unsigned int **)a4, 0x20u, v20, v34, (__int64)&v88);
        v36 = sub_BCB2D0(*(_QWORD **)(a4 + 72));
        v83 = (char *)sub_ACD640(v36, 1, 0);
        v88.m128i_i64[0] = (__int64)"stpncpy.end";
        LOWORD(v91) = 259;
        v37 = sub_921130((unsigned int **)a4, v74, v80, &v83, 1, (__int64)&v88, 3u);
        v88.m128i_i64[0] = (__int64)"stpncpy.sel";
        LOWORD(v91) = 259;
        return sub_B36550((unsigned int **)a4, v35, v80, v37, (__int64)&v88, 0);
      }
      return v80;
    }
    goto LABEL_20;
  }
  return v80;
}
