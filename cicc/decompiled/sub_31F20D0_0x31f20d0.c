// Function: sub_31F20D0
// Address: 0x31f20d0
//
void (*__fastcall sub_31F20D0(
        __int64 a1,
        char *a2,
        unsigned __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        unsigned int a7))()
{
  __int64 v8; // rcx
  unsigned __int64 v9; // rax
  bool v10; // cf
  __int64 v11; // rax
  __int64 (*v12)(); // rax
  void (*v13)(); // rax
  __int64 *v14; // rdi
  void (*result)(); // rax
  char v16; // al
  int v17; // eax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rdx
  _QWORD *v21; // r13
  unsigned __int64 *v22; // rcx
  unsigned __int64 *v23; // r15
  unsigned __int64 *v24; // r14
  unsigned __int64 v25; // rsi
  __int64 *v26; // r8
  __int64 v27; // rsi
  unsigned __int64 *v28; // rbx
  unsigned __int64 *v29; // rsi
  char *v30; // r15
  __int64 v31; // r15
  __int64 v32; // rax
  __int64 (*v33)(void); // rdx
  unsigned __int64 v34; // r13
  __int64 (__fastcall *v35)(__int64, __int64, unsigned __int64, __int64); // rax
  __int64 v36; // r14
  void (*v37)(); // rax
  void (*v38)(); // rax
  __int64 v39; // rsi
  void (*v40)(); // rbx
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 *v43; // r15
  __int64 *i; // rbx
  unsigned __int64 *v45; // rbx
  unsigned __int64 *v46; // r14
  __int64 v47; // rbx
  unsigned __int64 *v48; // rsi
  unsigned __int64 *v49; // rdi
  unsigned __int64 *v50; // rbx
  int v51; // [rsp+4h] [rbp-8Ch]
  __int64 v53; // [rsp+8h] [rbp-88h]
  unsigned __int64 *v54; // [rsp+8h] [rbp-88h]
  signed __int64 v56; // [rsp+18h] [rbp-78h]
  unsigned __int64 v58; // [rsp+28h] [rbp-68h]
  __int64 *v59; // [rsp+28h] [rbp-68h]
  __int64 *v60; // [rsp+28h] [rbp-68h]
  unsigned __int64 *v61; // [rsp+28h] [rbp-68h]
  unsigned __int64 *v62; // [rsp+28h] [rbp-68h]
  __int64 *v63; // [rsp+28h] [rbp-68h]
  unsigned __int64 v64; // [rsp+28h] [rbp-68h]
  _QWORD v65[4]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v66; // [rsp+50h] [rbp-40h]

  v8 = a6;
  if ( !a2[a3 - 1] )
  {
    v9 = a3 - 1;
    v10 = a3 != 0;
    a3 = 0;
    if ( v10 )
      a3 = v9;
  }
  v11 = *(_QWORD *)(*(_QWORD *)(a1 + 200) + 656LL);
  if ( *(_BYTE *)(v11 + 392)
    || *(_BYTE *)(v11 + 393)
    || (v12 = *(__int64 (**)())(**(_QWORD **)(a1 + 224) + 112LL), v12 != sub_C13F00)
    && (v58 = a3, v16 = v12(), a3 = v58, v8 = a6, v16) )
  {
    v17 = sub_31F1F80(a1, a2, a3, v8);
    v18 = *(_QWORD *)(a1 + 240);
    v51 = v17;
    v19 = *(_QWORD *)(v18 + 2480);
    v20 = v18 + 8;
    if ( !v19 )
      v19 = v20;
    v21 = *(_QWORD **)(v19 + 88);
    if ( (_QWORD *)(a5 + 224) == v21 + 3 )
      goto LABEL_26;
    v22 = *(unsigned __int64 **)(a5 + 232);
    v23 = (unsigned __int64 *)v21[3];
    v24 = *(unsigned __int64 **)(a5 + 224);
    v25 = v21[5] - (_QWORD)v23;
    v56 = (char *)v22 - (char *)v24;
    if ( v25 < (char *)v22 - (char *)v24 )
    {
      if ( v22 == v24 )
      {
        v43 = 0;
      }
      else
      {
        if ( (unsigned __int64)((char *)v22 - (char *)v24) > 0x7FFFFFFFFFFFFFE0LL )
          sub_4261EA(a1, v25, a5 + 224);
        v61 = *(unsigned __int64 **)(a5 + 232);
        v42 = sub_22077B0(v56);
        v22 = v61;
        v43 = (__int64 *)v42;
      }
      for ( i = v43; v22 != v24; i += 4 )
      {
        if ( i )
        {
          v62 = v22;
          *i = (__int64)(i + 2);
          sub_31F1830(i, (_BYTE *)*v24, *v24 + v24[1]);
          v22 = v62;
        }
        v24 += 4;
      }
      v45 = (unsigned __int64 *)v21[4];
      v46 = (unsigned __int64 *)v21[3];
      if ( v45 != v46 )
      {
        do
        {
          if ( (unsigned __int64 *)*v46 != v46 + 2 )
            j_j___libc_free_0(*v46);
          v46 += 4;
        }
        while ( v45 != v46 );
        v46 = (unsigned __int64 *)v21[3];
      }
      if ( v46 )
        j_j___libc_free_0((unsigned __int64)v46);
      v21[3] = v43;
      v30 = (char *)v43 + v56;
      v21[5] = v30;
      goto LABEL_25;
    }
    v26 = (__int64 *)v21[4];
    v27 = (char *)v26 - (char *)v23;
    if ( v56 > (unsigned __int64)((char *)v26 - (char *)v23) )
    {
      v47 = ((char *)v26 - (char *)v23) >> 5;
      if ( v27 > 0 )
      {
        do
        {
          v48 = v24;
          v49 = v23;
          v24 += 4;
          v23 += 4;
          sub_2240AE0(v49, v48);
          --v47;
        }
        while ( v47 );
        v26 = (__int64 *)v21[4];
        v23 = (unsigned __int64 *)v21[3];
        v22 = *(unsigned __int64 **)(a5 + 232);
        v24 = *(unsigned __int64 **)(a5 + 224);
        v27 = (char *)v26 - (char *)v23;
      }
      v50 = (unsigned __int64 *)((char *)v24 + v27);
      v30 = (char *)v23 + v56;
      if ( (unsigned __int64 *)((char *)v24 + v27) == v22 )
        goto LABEL_25;
      do
      {
        if ( v26 )
        {
          v54 = v22;
          *v26 = (__int64)(v26 + 2);
          v63 = v26;
          sub_31F1830(v26, (_BYTE *)*v50, *v50 + v50[1]);
          v22 = v54;
          v26 = v63;
        }
        v50 += 4;
        v26 += 4;
      }
      while ( v50 != v22 );
    }
    else
    {
      v53 = v56 >> 5;
      v28 = (unsigned __int64 *)v21[3];
      if ( v56 > 0 )
      {
        do
        {
          v29 = v24;
          v59 = v26;
          v24 += 4;
          sub_2240AE0(v28, v29);
          v28 += 4;
          v26 = v59;
          --v53;
        }
        while ( v53 );
        v23 = (unsigned __int64 *)((char *)v23 + v56);
      }
      for ( ; v26 != (__int64 *)v23; v23 += 4 )
      {
        if ( (unsigned __int64 *)*v23 != v23 + 2 )
        {
          v60 = v26;
          j_j___libc_free_0(*v23);
          v26 = v60;
        }
      }
    }
    v30 = (char *)(v21[3] + v56);
LABEL_25:
    v21[4] = v30;
LABEL_26:
    v31 = sub_EA87E0((__int64)v21, *(_DWORD **)(a1 + 216), *(_QWORD *)(a1 + 224), *(_QWORD *)(a1 + 208), v51);
    v32 = *(_QWORD *)(*(_QWORD *)(a1 + 200) + 8LL);
    v33 = *(__int64 (**)(void))(v32 + 64);
    if ( v33 )
    {
      v34 = v33();
      v32 = *(_QWORD *)(*(_QWORD *)(a1 + 200) + 8LL);
    }
    else
    {
      v34 = 0;
    }
    v35 = *(__int64 (__fastcall **)(__int64, __int64, unsigned __int64, __int64))(v32 + 112);
    if ( !v35 || (v36 = v35(a4, v31, v34, a5)) == 0 )
      sub_C64ED0("Inline asm not supported by this streamer because we don't have an asm parser for this target\n", 1u);
    if ( (unsigned int)(*(_DWORD *)(*(_QWORD *)(a1 + 200) + 544LL) - 38) <= 1 )
    {
      v37 = *(void (**)())(*(_QWORD *)v31 + 72LL);
      if ( v37 != nullsub_97 )
        ((void (__fastcall *)(__int64, _QWORD))v37)(v31, a7);
      if ( a7 == 1 )
        *(_BYTE *)((*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v31 + 40LL))(v31) + 117) = 1;
    }
    sub_ECD790(v31, v36);
    v38 = *(void (**)())(*(_QWORD *)a1 + 544LL);
    if ( v38 != nullsub_1850 )
      ((void (__fastcall *)(__int64))v38)(a1);
    v39 = 1;
    (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v31 + 80LL))(v31, 1, 1);
    v40 = *(void (**)())(*(_QWORD *)a1 + 552LL);
    v41 = sub_ECE6C0(v36);
    if ( v40 != nullsub_1851 )
    {
      v39 = a4;
      ((void (__fastcall *)(__int64, __int64, __int64))v40)(a1, a4, v41);
    }
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v36 + 8LL))(v36);
    if ( v34 )
    {
      v39 = 48;
      j_j___libc_free_0(v34);
    }
    return (void (*)())(*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v31 + 8LL))(v31, v39);
  }
  v13 = *(void (**)())(*(_QWORD *)a1 + 544LL);
  if ( v13 != nullsub_1850 )
  {
    v64 = a3;
    ((void (__fastcall *)(__int64))v13)(a1);
    a3 = v64;
  }
  v14 = *(__int64 **)(a1 + 224);
  v65[0] = a2;
  v66 = 261;
  v65[1] = a3;
  sub_E99A90(v14, (__int64)v65);
  result = *(void (**)())(*(_QWORD *)a1 + 552LL);
  if ( result != nullsub_1851 )
    return (void (*)())((__int64 (__fastcall *)(__int64, __int64, _QWORD))result)(a1, a4, 0);
  return result;
}
