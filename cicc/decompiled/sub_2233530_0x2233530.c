// Function: sub_2233530
// Address: 0x2233530
//
__int64 __fastcall sub_2233530(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, char a6, double a7)
{
  __int64 v10; // rax
  __int64 v11; // r13
  int v12; // eax
  __int64 *v13; // rdi
  __int64 v14; // rax
  signed __int64 v15; // r15
  __int64 v16; // rdi
  int v17; // r10d
  char *v18; // rdx
  size_t v19; // r8
  void *v20; // rsp
  char v21; // al
  char *v22; // rcx
  char *v23; // r13
  void (__fastcall *v24)(__int64, char *, char *, char *, size_t); // rax
  size_t v25; // rdx
  char *v26; // rax
  char *v27; // rax
  char *v28; // r8
  __int64 v29; // rsi
  __int64 v30; // rdx
  void *v32; // rsp
  void *v33; // rsp
  char v34; // al
  char *v35; // r9
  int v36; // ebx
  int v37; // ebx
  void *v38; // rsp
  int v39; // eax
  __int64 *v40; // rdi
  void *v41; // rsp
  char *v42; // [rsp-1Eh] [rbp-E0h]
  char v43; // [rsp-Eh] [rbp-D0h] BYREF
  _BYTE v44[14]; // [rsp-Dh] [rbp-CFh] BYREF
  char *v45; // [rsp+32h] [rbp-90h]
  __int64 *v46; // [rsp+3Ah] [rbp-88h]
  int v47; // [rsp+46h] [rbp-7Ch]
  __int64 v48; // [rsp+4Ah] [rbp-78h]
  __int64 v49; // [rsp+52h] [rbp-70h]
  char *v50; // [rsp+5Ah] [rbp-68h]
  __int64 v51; // [rsp+62h] [rbp-60h]
  double v52; // [rsp+6Ah] [rbp-58h]
  char v53; // [rsp+75h] [rbp-4Dh] BYREF
  int v54; // [rsp+76h] [rbp-4Ch] BYREF
  __int64 v55; // [rsp+7Ah] [rbp-48h] BYREF
  char v56[64]; // [rsp+82h] [rbp-40h] BYREF

  v48 = a1;
  v49 = a3;
  v47 = a5;
  v52 = a7;
  v50 = (char *)(a4 + 208);
  v10 = sub_2232A70((__int64)&v53, (__int64 *)(a4 + 208));
  v11 = *(_QWORD *)(a4 + 8);
  v51 = v10;
  if ( v11 < 0 )
    v11 = 6;
  sub_2255110(a4, v56, (unsigned int)a6);
  if ( (*(_DWORD *)(a4 + 24) & 0x104) == 0x104 )
  {
    v55 = sub_2208E60(a4, v56);
    v46 = &v55;
    v39 = sub_2218500((__int64)&v55, &v43, 45, v56, v52);
    v40 = v46;
    v54 = v39;
    if ( v39 > 44 )
    {
      v45 = (char *)v46;
      LODWORD(v46) = v39 + 1;
      v41 = alloca(v39 + 1 + 8LL);
      v55 = sub_2208E60(v40, &v43);
      v54 = sub_2218500((__int64)v45, &v43, (int)v46, v56, v52);
    }
  }
  else
  {
    v55 = sub_2208E60(a4, v56);
    v46 = &v55;
    v12 = sub_2218500((__int64)&v55, &v43, 45, v56, v11, v52);
    v13 = v46;
    v54 = v12;
    if ( v12 > 44 )
    {
      v45 = (char *)v46;
      LODWORD(v46) = v12 + 1;
      v32 = alloca(v12 + 1 + 8LL);
      v55 = sub_2208E60(v13, &v43);
      v54 = sub_2218500((__int64)v45, &v43, (int)v46, v56, v11, v52);
    }
  }
  v14 = sub_222F790(v50, (__int64)&v43);
  v15 = v54;
  v16 = v14;
  v17 = v54;
  v18 = &v44[v54 - 1];
  v19 = v54;
  v20 = alloca(v54 + 8LL);
  v21 = *(_BYTE *)(v14 + 56);
  v22 = &v43;
  v23 = &v43;
  if ( v21 == 1 )
  {
    if ( v18 != &v43 )
    {
      LODWORD(v52) = v54;
      v25 = v54;
LABEL_11:
      v26 = (char *)memcpy(v22, &v43, v25);
      v17 = LODWORD(v52);
      v22 = v26;
    }
  }
  else
  {
    if ( v21 )
    {
      v24 = *(void (__fastcall **)(__int64, char *, char *, char *, size_t))(*(_QWORD *)v16 + 56LL);
      if ( (char *)v24 == (char *)sub_2216D40 )
      {
LABEL_9:
        v15 = v54;
        v17 = v54;
        if ( v18 == &v43 )
          goto LABEL_12;
        LODWORD(v52) = v54;
        v25 = v19;
        goto LABEL_11;
      }
    }
    else
    {
      v45 = &v43;
      v46 = (__int64 *)&v44[v54 - 1];
      v52 = *(double *)&v16;
      v50 = (char *)v54;
      sub_2216D60(v16);
      v16 = *(_QWORD *)&v52;
      v22 = v45;
      v18 = (char *)v46;
      v19 = (size_t)v50;
      v24 = *(void (__fastcall **)(__int64, char *, char *, char *, size_t))(**(_QWORD **)&v52 + 56LL);
      if ( (char *)v24 == (char *)sub_2216D40 )
        goto LABEL_9;
    }
    v52 = *(double *)&v22;
    v24(v16, &v43, v18, v22, v19);
    v15 = v54;
    v22 = *(char **)&v52;
    v17 = v54;
  }
LABEL_12:
  if ( !v15 )
  {
    if ( *(_BYTE *)(v51 + 32) )
    {
      v28 = 0;
      goto LABEL_21;
    }
LABEL_15:
    v30 = *(_QWORD *)(a4 + 16);
    if ( v15 >= v30 )
      goto LABEL_16;
    goto LABEL_24;
  }
  v50 = v22;
  LODWORD(v52) = v17;
  v27 = (char *)memchr(&v43, 46, v15);
  v17 = LODWORD(v52);
  v22 = v50;
  v28 = v27;
  if ( !v27 )
  {
    if ( *(_BYTE *)(v51 + 32)
      && (SLODWORD(v52) <= 2 || v44[0] <= 57 && (unsigned __int8)(v44[1] - 48) <= 9u && v44[0] > 47) )
    {
      goto LABEL_21;
    }
    goto LABEL_15;
  }
  v29 = v51;
  v28 = &v50[v27 - &v43];
  *v28 = *(_BYTE *)(v51 + 72);
  if ( !*(_BYTE *)(v29 + 32) )
    goto LABEL_15;
LABEL_21:
  v33 = alloca(2 * v15 + 8);
  if ( ((v43 - 43) & 0xFD) != 0 )
  {
    v35 = &v43;
    v36 = 0;
  }
  else
  {
    v34 = *v22;
    v23 = v22 + 1;
    v54 = v17 - 1;
    v35 = v44;
    v36 = 1;
    v43 = v34;
  }
  sub_2231600(v48, *(char **)(v51 + 16), *(_QWORD *)(v51 + 24), *(_BYTE *)(v51 + 73), v28, v35, (__int64)v23, &v54);
  v37 = v54 + v36;
  v30 = *(_QWORD *)(a4 + 16);
  v15 = v37;
  v22 = v42;
  v54 = v37;
  if ( v37 >= v30 )
    goto LABEL_16;
LABEL_24:
  v38 = alloca(v30 + 8);
  sub_22328D0(v48, v47, v30, a4, &v43, &v43, &v54);
  v15 = v54;
LABEL_16:
  *(_QWORD *)(a4 + 16) = 0;
  if ( !(_BYTE)v49 )
    (*(__int64 (__fastcall **)(__int64, char *, signed __int64, char *))(*(_QWORD *)a2 + 96LL))(a2, &v43, v15, v22);
  return a2;
}
