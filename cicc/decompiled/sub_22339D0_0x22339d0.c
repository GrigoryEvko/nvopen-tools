// Function: sub_22339D0
// Address: 0x22339d0
//
__int64 __fastcall sub_22339D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, char a6, __int64 a7, char *a8)
{
  __int64 v11; // rax
  __int64 v12; // r13
  int v13; // eax
  _QWORD *v14; // rdi
  char *v15; // rsi
  __int64 v16; // rax
  signed __int64 v17; // r15
  char *v18; // rdi
  int v19; // r10d
  char *v20; // rdx
  size_t v21; // r8
  void *v22; // rsp
  char v23; // al
  char *v24; // rcx
  char *v25; // r13
  void (__fastcall *v26)(char *, char *, char *, char *, size_t); // rax
  size_t v27; // rdx
  char *v28; // rax
  _BYTE *v29; // rax
  _BYTE *v30; // r8
  __int64 v31; // rsi
  __int64 v32; // rdx
  void *v34; // rsp
  void *v35; // rsp
  char v36; // al
  char *v37; // r9
  int v38; // ebx
  int v39; // ebx
  void *v40; // rsp
  int v41; // eax
  _QWORD *v42; // rdi
  void *v43; // rsp
  char v44; // [rsp-Eh] [rbp-D0h] BYREF
  _BYTE v45[14]; // [rsp-Dh] [rbp-CFh] BYREF
  char *v46; // [rsp+32h] [rbp-90h]
  char *v47; // [rsp+3Ah] [rbp-88h]
  int v48; // [rsp+46h] [rbp-7Ch]
  size_t v49; // [rsp+4Ah] [rbp-78h]
  __int64 v50; // [rsp+52h] [rbp-70h]
  __int64 v51; // [rsp+5Ah] [rbp-68h]
  char *v52; // [rsp+62h] [rbp-60h]
  __int64 v53; // [rsp+6Ah] [rbp-58h]
  char v54; // [rsp+75h] [rbp-4Dh] BYREF
  int v55; // [rsp+76h] [rbp-4Ch] BYREF
  __int64 v56; // [rsp+7Ah] [rbp-48h] BYREF
  char v57[64]; // [rsp+82h] [rbp-40h] BYREF

  v50 = a1;
  v51 = a3;
  v48 = a5;
  v52 = (char *)(a4 + 208);
  v11 = sub_2232A70((__int64)&v54, (__int64 *)(a4 + 208));
  v12 = *(_QWORD *)(a4 + 8);
  v53 = v11;
  if ( v12 < 0 )
    v12 = 6;
  sub_2255110(a4, v57, (unsigned int)a6);
  if ( (*(_DWORD *)(a4 + 24) & 0x104) == 0x104 )
  {
    v15 = &v44;
    v56 = sub_2208E60(a4, v57);
    v49 = (size_t)&v56;
    v41 = sub_2218500((__int64)&v56, &v44, 54, v57);
    v42 = (_QWORD *)v49;
    v55 = v41;
    if ( v41 > 53 )
    {
      v47 = (char *)v49;
      LODWORD(v49) = v41 + 1;
      v43 = alloca(v41 + 1 + 8LL);
      v56 = sub_2208E60(v42, &v44);
      v15 = &v44;
      v55 = sub_2218500((__int64)v47, &v44, v49, v57);
    }
  }
  else
  {
    v56 = sub_2208E60(a4, v57);
    v49 = (size_t)&v56;
    v13 = sub_2218500((__int64)&v56, &v44, 54, v57, v12);
    v14 = (_QWORD *)v49;
    v55 = v13;
    v15 = a8;
    if ( v13 > 53 )
    {
      v47 = (char *)v49;
      LODWORD(v49) = v13 + 1;
      v34 = alloca(v13 + 1 + 8LL);
      v56 = sub_2208E60(v14, a8);
      v15 = a8;
      v55 = sub_2218500((__int64)v47, &v44, v49, v57, v12);
    }
  }
  v16 = sub_222F790(v52, (__int64)v15);
  v17 = v55;
  v18 = (char *)v16;
  v19 = v55;
  v20 = &v45[v55 - 1];
  v21 = v55;
  v22 = alloca(v55 + 8LL);
  v23 = *(_BYTE *)(v16 + 56);
  v24 = &v44;
  v25 = &v44;
  if ( v23 == 1 )
  {
    if ( v20 != &v44 )
    {
      LODWORD(v52) = v55;
      v27 = v55;
LABEL_11:
      v28 = (char *)memcpy(v24, &v44, v27);
      v19 = (int)v52;
      v24 = v28;
    }
  }
  else
  {
    if ( v23 )
    {
      v26 = *(void (__fastcall **)(char *, char *, char *, char *, size_t))(*(_QWORD *)v18 + 56LL);
      if ( (char *)v26 == (char *)sub_2216D40 )
      {
LABEL_9:
        v17 = v55;
        v19 = v55;
        if ( v20 == &v44 )
          goto LABEL_12;
        LODWORD(v52) = v55;
        v27 = v21;
        goto LABEL_11;
      }
    }
    else
    {
      v46 = &v44;
      v47 = &v45[v55 - 1];
      v52 = v18;
      v49 = v55;
      sub_2216D60((__int64)v18);
      v18 = v52;
      v24 = v46;
      v20 = v47;
      v21 = v49;
      v26 = *(void (__fastcall **)(char *, char *, char *, char *, size_t))(*(_QWORD *)v52 + 56LL);
      if ( (char *)v26 == (char *)sub_2216D40 )
        goto LABEL_9;
    }
    v52 = v24;
    v26(v18, &v44, v20, v24, v21);
    v17 = v55;
    v24 = v52;
    v19 = v55;
  }
LABEL_12:
  if ( !v17 )
  {
    if ( *(_BYTE *)(v53 + 32) )
    {
      v30 = 0;
      goto LABEL_21;
    }
LABEL_15:
    v32 = *(_QWORD *)(a4 + 16);
    if ( v17 >= v32 )
      goto LABEL_16;
    goto LABEL_24;
  }
  v49 = (size_t)v24;
  LODWORD(v52) = v19;
  v29 = memchr(&v44, 46, v17);
  v19 = (int)v52;
  v24 = (char *)v49;
  v30 = v29;
  if ( !v29 )
  {
    if ( *(_BYTE *)(v53 + 32) && ((int)v52 <= 2 || v45[0] <= 57 && (unsigned __int8)(v45[1] - 48) <= 9u && v45[0] > 47) )
      goto LABEL_21;
    goto LABEL_15;
  }
  v31 = v53;
  v30 = (_BYTE *)(v49 + v29 - &v44);
  *v30 = *(_BYTE *)(v53 + 72);
  if ( !*(_BYTE *)(v31 + 32) )
    goto LABEL_15;
LABEL_21:
  v35 = alloca(2 * v17 + 8);
  if ( ((v44 - 43) & 0xFD) != 0 )
  {
    v37 = &v44;
    v38 = 0;
  }
  else
  {
    v36 = *v24;
    v25 = v24 + 1;
    v55 = v19 - 1;
    v37 = v45;
    v38 = 1;
    v44 = v36;
  }
  sub_2231600(v50, *(char **)(v53 + 16), *(_QWORD *)(v53 + 24), *(_BYTE *)(v53 + 73), v30, v37, (__int64)v25, &v55);
  v39 = v55 + v38;
  v32 = *(_QWORD *)(a4 + 16);
  v17 = v39;
  v55 = v39;
  if ( v39 >= v32 )
    goto LABEL_16;
LABEL_24:
  v40 = alloca(v32 + 8);
  sub_22328D0(v50, v48, v32, a4, &v44, &v44, &v55);
  v17 = v55;
LABEL_16:
  *(_QWORD *)(a4 + 16) = 0;
  if ( !(_BYTE)v51 )
    (*(__int64 (__fastcall **)(__int64, char *, signed __int64, char *))(*(_QWORD *)a2 + 96LL))(a2, &v44, v17, v24);
  return a2;
}
