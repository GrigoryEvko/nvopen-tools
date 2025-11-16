// Function: sub_397D680
// Address: 0x397d680
//
void (*__fastcall sub_397D680(
        _QWORD *a1,
        char *a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned int a7))()
{
  _QWORD *v7; // r12
  unsigned __int64 v8; // rax
  __int64 (*v9)(); // rax
  void (*v10)(); // rax
  __int64 *v11; // rdi
  void (*result)(); // rax
  __int64 *v13; // r13
  unsigned __int64 *v14; // rcx
  unsigned __int64 *v15; // r15
  unsigned __int64 *v16; // r14
  unsigned __int64 v17; // rsi
  __int64 *v18; // r8
  __int64 v19; // rsi
  unsigned __int64 *v20; // rbx
  unsigned __int64 *v21; // rsi
  char *v22; // r15
  _QWORD *v23; // rdi
  _QWORD *v24; // rdi
  unsigned __int64 v25; // rbx
  __int64 v26; // rdi
  __int64 v27; // rdx
  unsigned __int64 v28; // rax
  __int64 v29; // rax
  _BYTE *v30; // rbx
  __int64 v31; // rax
  __int64 (*v32)(void); // rdx
  unsigned __int64 v33; // r15
  __int64 (__fastcall *v34)(__int64, _BYTE *, unsigned __int64, __int64); // rax
  __int64 v35; // r13
  void (*v36)(); // rax
  __int64 v37; // rax
  __int64 (*v38)(); // rax
  __int64 v39; // rax
  void (*v40)(); // r14
  __int64 v41; // rsi
  void (*v42)(); // rax
  __int64 v43; // rsi
  void (*v44)(); // r14
  __int64 v45; // rax
  __int64 v46; // rbx
  unsigned __int64 *v47; // rsi
  unsigned __int64 *v48; // rdi
  unsigned __int64 *v49; // rbx
  unsigned __int64 v50; // rdx
  __int64 v51; // rax
  __int64 *v52; // r15
  __int64 *i; // rbx
  unsigned __int64 *v54; // rbx
  unsigned __int64 *v55; // r14
  void *v56; // rax
  void *v57; // rdx
  unsigned __int64 *v58; // r13
  unsigned __int64 v59; // rdi
  unsigned __int64 *v60; // rbx
  unsigned __int64 *v61; // r15
  unsigned __int64 v62; // rbx
  unsigned __int64 v63; // r15
  __int64 *v64; // rdi
  __int64 v65; // r13
  __int64 v66; // rbx
  __int64 v67; // rbx
  __int64 v68; // rax
  __int64 v69; // [rsp+0h] [rbp-A0h]
  unsigned __int64 *v70; // [rsp+0h] [rbp-A0h]
  signed __int64 v72; // [rsp+10h] [rbp-90h]
  __int64 *v75; // [rsp+28h] [rbp-78h]
  __int64 *v76; // [rsp+28h] [rbp-78h]
  char v77; // [rsp+28h] [rbp-78h]
  __int64 *v78; // [rsp+28h] [rbp-78h]
  unsigned __int64 *v79; // [rsp+28h] [rbp-78h]
  unsigned __int64 *v80; // [rsp+28h] [rbp-78h]
  const void *v81; // [rsp+30h] [rbp-70h] BYREF
  unsigned __int64 v82; // [rsp+38h] [rbp-68h]
  __int64 v83; // [rsp+48h] [rbp-58h] BYREF
  __int64 v84[2]; // [rsp+50h] [rbp-50h] BYREF
  __int64 v85; // [rsp+60h] [rbp-40h]

  v7 = a1;
  v81 = a2;
  v82 = a3;
  if ( !a2[a3 - 1] )
  {
    v8 = a3 - 1;
    if ( !a3 )
      v8 = 0;
    v82 = v8;
  }
  if ( *(_BYTE *)(*(_QWORD *)(a1[29] + 608LL) + 392LL)
    || (a1 = (_QWORD *)a1[32], v9 = *(__int64 (**)())(*a1 + 96LL), v9 != sub_168DB70) && (unsigned __int8)v9() )
  {
    v13 = (__int64 *)v7[62];
    if ( !v13 )
    {
      v56 = (void *)sub_22077B0(0x68u);
      v57 = v56;
      if ( v56 )
        memset(v56, 0, 0x68u);
      v58 = (unsigned __int64 *)v7[62];
      v7[62] = v56;
      if ( v58 )
      {
        v59 = v58[8];
        if ( v59 )
          j_j___libc_free_0(v59);
        v60 = (unsigned __int64 *)v58[4];
        v61 = (unsigned __int64 *)v58[3];
        if ( v60 != v61 )
        {
          do
          {
            if ( (unsigned __int64 *)*v61 != v61 + 2 )
              j_j___libc_free_0(*v61);
            v61 += 4;
          }
          while ( v60 != v61 );
          v61 = (unsigned __int64 *)v58[3];
        }
        if ( v61 )
          j_j___libc_free_0((unsigned __int64)v61);
        v62 = v58[1];
        v63 = *v58;
        if ( v62 != *v58 )
        {
          do
          {
            v64 = (__int64 *)v63;
            v63 += 24LL;
            sub_16CE300(v64);
          }
          while ( v62 != v63 );
          v63 = *v58;
        }
        if ( v63 )
          j_j___libc_free_0(v63);
        j_j___libc_free_0((unsigned __int64)v58);
        v57 = (void *)v7[62];
      }
      *(_QWORD *)(v7[34] + 176LL) = v57;
      v65 = **(_QWORD **)(v7[34] + 1688LL);
      a1 = (_QWORD *)v65;
      if ( sub_1602750(v65) )
      {
        v66 = v7[62];
        a1 = (_QWORD *)v65;
        *(_QWORD *)(v66 + 88) = sub_1602750(v65);
        v67 = v7[62];
        *(_QWORD *)(v67 + 96) = sub_1602760(v65);
        v68 = v7[62];
        *(_QWORD *)(v68 + 48) = sub_397CF80;
        *(_QWORD *)(v68 + 56) = v68;
      }
      v13 = (__int64 *)v7[62];
    }
    if ( (__int64 *)(a5 + 72) == v13 + 3 )
      goto LABEL_25;
    v14 = *(unsigned __int64 **)(a5 + 80);
    v15 = (unsigned __int64 *)v13[3];
    v16 = *(unsigned __int64 **)(a5 + 72);
    v17 = v13[5] - (_QWORD)v15;
    v72 = (char *)v14 - (char *)v16;
    if ( v17 < (char *)v14 - (char *)v16 )
    {
      v50 = (char *)v14 - (char *)v16;
      if ( v14 == v16 )
      {
        v52 = 0;
      }
      else
      {
        if ( v50 > 0x7FFFFFFFFFFFFFE0LL )
          sub_4261EA(a1, v17, v50);
        v79 = *(unsigned __int64 **)(a5 + 80);
        v51 = sub_22077B0(v72);
        v14 = v79;
        v52 = (__int64 *)v51;
      }
      for ( i = v52; v14 != v16; i += 4 )
      {
        if ( i )
        {
          v80 = v14;
          *i = (__int64)(i + 2);
          sub_397D180(i, (_BYTE *)*v16, *v16 + v16[1]);
          v14 = v80;
        }
        v16 += 4;
      }
      v54 = (unsigned __int64 *)v13[4];
      v55 = (unsigned __int64 *)v13[3];
      if ( v54 != v55 )
      {
        do
        {
          if ( (unsigned __int64 *)*v55 != v55 + 2 )
            j_j___libc_free_0(*v55);
          v55 += 4;
        }
        while ( v54 != v55 );
        v55 = (unsigned __int64 *)v13[3];
      }
      if ( v55 )
        j_j___libc_free_0((unsigned __int64)v55);
      v13[3] = (__int64)v52;
      v22 = (char *)v52 + v72;
      v13[5] = (__int64)v22;
      goto LABEL_24;
    }
    v18 = (__int64 *)v13[4];
    v19 = (char *)v18 - (char *)v15;
    if ( v72 > (unsigned __int64)((char *)v18 - (char *)v15) )
    {
      v46 = ((char *)v18 - (char *)v15) >> 5;
      if ( v19 > 0 )
      {
        do
        {
          v47 = v16;
          v48 = v15;
          v16 += 4;
          v15 += 4;
          sub_2240AE0(v48, v47);
          --v46;
        }
        while ( v46 );
        v18 = (__int64 *)v13[4];
        v15 = (unsigned __int64 *)v13[3];
        v14 = *(unsigned __int64 **)(a5 + 80);
        v16 = *(unsigned __int64 **)(a5 + 72);
        v19 = (char *)v18 - (char *)v15;
      }
      v49 = (unsigned __int64 *)((char *)v16 + v19);
      v22 = (char *)v15 + v72;
      if ( v14 == (unsigned __int64 *)((char *)v16 + v19) )
        goto LABEL_24;
      do
      {
        if ( v18 )
        {
          v70 = v14;
          *v18 = (__int64)(v18 + 2);
          v78 = v18;
          sub_397D180(v18, (_BYTE *)*v49, *v49 + v49[1]);
          v14 = v70;
          v18 = v78;
        }
        v49 += 4;
        v18 += 4;
      }
      while ( v14 != v49 );
    }
    else
    {
      v20 = (unsigned __int64 *)v13[3];
      v69 = v72 >> 5;
      if ( v72 <= 0 )
        goto LABEL_22;
      do
      {
        v21 = v16;
        v75 = v18;
        v16 += 4;
        sub_2240AE0(v20, v21);
        v20 += 4;
        v18 = v75;
        --v69;
      }
      while ( v69 );
      v15 = (unsigned __int64 *)((char *)v15 + v72);
      if ( v75 != (__int64 *)v15 )
      {
        do
        {
          if ( (unsigned __int64 *)*v15 != v15 + 2 )
          {
            v76 = v18;
            j_j___libc_free_0(*v15);
            v18 = v76;
          }
          v15 += 4;
LABEL_22:
          ;
        }
        while ( v18 != (__int64 *)v15 );
      }
    }
    v22 = (char *)(v13[3] + v72);
LABEL_24:
    v13[4] = (__int64)v22;
LABEL_25:
    v84[0] = (__int64)"<inline asm>";
    LOWORD(v85) = 259;
    sub_16C28C0(&v83, v81, v82, (__int64)v84);
    v84[1] = 0;
    v85 = 0;
    v84[0] = v83;
    v23 = (_QWORD *)v13[1];
    if ( v23 == (_QWORD *)v13[2] )
    {
      sub_168C7C0(v13, v13[1], (__int64)v84);
      v24 = (_QWORD *)v13[1];
    }
    else
    {
      if ( v23 )
      {
        sub_16CE2D0(v23, v84);
        v23 = (_QWORD *)v13[1];
      }
      v24 = v23 + 3;
      v13[1] = (__int64)v24;
    }
    v25 = 0xAAAAAAAAAAAAAAABLL * (((__int64)v24 - *v13) >> 3);
    sub_16CE300(v84);
    if ( a6 )
    {
      v26 = v7[62];
      v27 = *(_QWORD *)(v26 + 64);
      v28 = (*(_QWORD *)(v26 + 72) - v27) >> 3;
      if ( (unsigned int)v25 > v28 )
      {
        sub_397D4D0(v26 + 64, (unsigned int)v25 - v28);
        *(_QWORD *)(*(_QWORD *)(v7[62] + 64LL) + 8LL * (unsigned int)(v25 - 1)) = a6;
      }
      else
      {
        if ( (unsigned int)v25 < v28 )
        {
          v29 = v27 + 8LL * (unsigned int)v25;
          if ( *(_QWORD *)(v26 + 72) != v29 )
          {
            *(_QWORD *)(v26 + 72) = v29;
            v27 = *(_QWORD *)(v7[62] + 64LL);
          }
        }
        *(_QWORD *)(v27 + 8LL * (unsigned int)(v25 - 1)) = a6;
      }
    }
    v30 = (_BYTE *)sub_38E8880((__int64)v13, v7[31], v7[32], v7[30], v25);
    *(_BYTE *)(v7[32] + 260LL) = 0;
    v31 = *(_QWORD *)(v7[29] + 8LL);
    v32 = *(__int64 (**)(void))(v31 + 56);
    if ( v32 )
    {
      v33 = v32();
      v31 = *(_QWORD *)(v7[29] + 8LL);
    }
    else
    {
      v33 = 0;
    }
    v34 = *(__int64 (__fastcall **)(__int64, _BYTE *, unsigned __int64, __int64))(v31 + 104);
    if ( !v34 || (v35 = v34(a4, v30, v33, a5)) == 0 )
      sub_16BD130("Inline asm not supported by this streamer because we don't have an asm parser for this target\n", 1u);
    v36 = *(void (**)())(*(_QWORD *)v30 + 72LL);
    if ( v36 != nullsub_1962 )
      ((void (__fastcall *)(_BYTE *, _QWORD))v36)(v30, a7);
    sub_3909440((__int64)v30, v35);
    v30[18] = *((_BYTE *)v7 + 376);
    if ( a7 == 1 )
      (*(void (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v30 + 88LL))(v30, 1);
    v37 = v7[33];
    if ( v37 )
    {
      v38 = *(__int64 (**)())(**(_QWORD **)(v37 + 16) + 112LL);
      if ( v38 == sub_1D00B10 )
        BUG();
      v39 = v38();
      v40 = *(void (**)())(*(_QWORD *)v35 + 40LL);
      v41 = (*(unsigned int (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v39 + 416LL))(v39, v7[33]);
      if ( v40 != nullsub_1964 )
        ((void (__fastcall *)(__int64, __int64))v40)(v35, v41);
    }
    v42 = *(void (**)())(*v7 + 376LL);
    if ( v42 != nullsub_1976 )
      ((void (__fastcall *)(_QWORD *))v42)(v7);
    v43 = 1;
    v77 = (*(__int64 (__fastcall **)(_BYTE *, __int64, __int64))(*(_QWORD *)v30 + 80LL))(v30, 1, 1);
    v44 = *(void (**)())(*v7 + 384LL);
    v45 = sub_390A040(v35);
    if ( v44 != nullsub_1977 )
    {
      v43 = a4;
      ((void (__fastcall *)(_QWORD *, __int64, __int64))v44)(v7, a4, v45);
    }
    if ( v77 && !*(_QWORD *)(v7[62] + 88LL) )
      sub_16BD130("Error parsing inline asm\n", 1u);
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v35 + 8LL))(v35);
    if ( v33 )
    {
      v43 = 32;
      j_j___libc_free_0(v33);
    }
    return (void (*)())(*(__int64 (__fastcall **)(_BYTE *, __int64))(*(_QWORD *)v30 + 8LL))(v30, v43);
  }
  v10 = *(void (**)())(*v7 + 376LL);
  if ( v10 != nullsub_1976 )
    ((void (__fastcall *)(_QWORD *))v10)(v7);
  v11 = (__int64 *)v7[32];
  LOWORD(v85) = 261;
  v84[0] = (__int64)&v81;
  sub_38DD5A0(v11, (__int64)v84);
  result = *(void (**)())(*v7 + 384LL);
  if ( result != nullsub_1977 )
    return (void (*)())((__int64 (__fastcall *)(_QWORD *, __int64, _QWORD))result)(v7, a4, 0);
  return result;
}
