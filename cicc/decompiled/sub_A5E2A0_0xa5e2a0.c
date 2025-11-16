// Function: sub_A5E2A0
// Address: 0xa5e2a0
//
__int64 __fastcall sub_A5E2A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rdx
  _BYTE *v9; // rax
  __int64 v10; // rcx
  _BYTE *v11; // rax
  unsigned int v12; // eax
  __int64 v13; // rcx
  unsigned int v14; // ecx
  int v15; // ebx
  __int64 v16; // rdi
  __int64 v17; // rax
  __int64 v18; // rax
  unsigned int v19; // r15d
  __int64 v20; // rdi
  __int64 v21; // rdx
  unsigned __int64 v22; // rsi
  unsigned int v23; // eax
  unsigned int *v24; // rdi
  unsigned int *v25; // r15
  char v26; // bl
  void *v27; // rdi
  __int64 v28; // rax
  __int64 v29; // r9
  const void *v30; // r10
  size_t v31; // rdx
  size_t v32; // r8
  __int64 v33; // rdx
  __int64 v34; // rax
  __int64 v35; // rdi
  _BYTE *v36; // rax
  unsigned int v37; // eax
  __int64 v38; // rcx
  _BYTE *v39; // rax
  _BYTE *v40; // rax
  unsigned int v41; // eax
  __int64 v42; // rcx
  unsigned int v43; // eax
  __int64 v44; // rcx
  unsigned int v45; // eax
  __int64 v46; // rcx
  __int64 v47; // rdx
  __int64 v48; // r8
  __int64 v50; // rax
  unsigned int v51; // [rsp+Ch] [rbp-A4h]
  __int64 v53; // [rsp+18h] [rbp-98h]
  size_t v54; // [rsp+18h] [rbp-98h]
  size_t v55; // [rsp+20h] [rbp-90h]
  const void *v56; // [rsp+20h] [rbp-90h]
  unsigned int *v57; // [rsp+28h] [rbp-88h]
  __int64 v58; // [rsp+30h] [rbp-80h] BYREF
  char v59; // [rsp+38h] [rbp-78h]
  const char *v60; // [rsp+40h] [rbp-70h]
  __int64 v61; // [rsp+48h] [rbp-68h]
  unsigned int *v62; // [rsp+50h] [rbp-60h] BYREF
  __int64 v63; // [rsp+58h] [rbp-58h]
  _BYTE v64[80]; // [rsp+60h] [rbp-50h] BYREF

  sub_904010(a1, "!DISubprogram(");
  v58 = a1;
  v59 = 1;
  v60 = ", ";
  v61 = a3;
  v5 = sub_A547D0(a2, 2);
  sub_A53660(&v58, "name", 4u, v5, v6, 1);
  v7 = sub_A547D0(a2, 3);
  sub_A53660(&v58, "linkageName", 0xBu, v7, v8, 1);
  v9 = sub_A17150((_BYTE *)(a2 - 16));
  sub_A5CC00((__int64)&v58, "scope", 5u, *((_QWORD *)v9 + 1), 0);
  v10 = a2;
  if ( *(_BYTE *)a2 != 16 )
    v10 = *(_QWORD *)sub_A17150((_BYTE *)(a2 - 16));
  sub_A5CC00((__int64)&v58, "file", 4u, v10, 1);
  sub_A537C0((__int64)&v58, "line", 4u, *(_DWORD *)(a2 + 16), 1);
  v11 = sub_A17150((_BYTE *)(a2 - 16));
  sub_A5CC00((__int64)&v58, "type", 4u, *((_QWORD *)v11 + 4), 1);
  sub_A537C0((__int64)&v58, "scopeLine", 9u, *(_DWORD *)(a2 + 20), 1);
  if ( (*(_BYTE *)(a2 - 16) & 2) != 0 )
    v12 = *(_DWORD *)(a2 - 24);
  else
    v12 = (*(_WORD *)(a2 - 16) >> 6) & 0xF;
  v13 = 0;
  if ( v12 > 8 )
    v13 = *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 8);
  sub_A5CC00((__int64)&v58, "containingType", 0xEu, v13, 1);
  v14 = *(_DWORD *)(a2 + 24);
  if ( (*(_BYTE *)(a2 + 36) & 3) != 0 || v14 )
    sub_A537C0((__int64)&v58, "virtualIndex", 0xCu, v14, 0);
  v15 = *(_DWORD *)(a2 + 28);
  if ( v15 )
  {
    v16 = v58;
    if ( v59 )
      v59 = 0;
    else
      v16 = sub_904010(v58, v60);
    v17 = sub_A51340(v16, "thisAdjustment", 0xEu);
    v18 = sub_904010(v17, ": ");
    sub_CB59F0(v18, v15);
  }
  sub_A53C60(&v58, "flags", 5u, *(_DWORD *)(a2 + 32));
  v19 = *(_DWORD *)(a2 + 36);
  v20 = v58;
  if ( v59 )
    v59 = 0;
  else
    v20 = sub_904010(v58, v60);
  v21 = *(_QWORD *)(v20 + 32);
  if ( (unsigned __int64)(*(_QWORD *)(v20 + 24) - v21) <= 6 )
  {
    v20 = sub_CB6200(v20, "spFlags", 7);
  }
  else
  {
    *(_DWORD *)v21 = 1816555635;
    *(_WORD *)(v21 + 4) = 26465;
    *(_BYTE *)(v21 + 6) = 115;
    *(_QWORD *)(v20 + 32) += 7LL;
  }
  sub_904010(v20, ": ");
  if ( !v19 )
  {
    sub_CB59F0(v58, 0);
    goto LABEL_35;
  }
  v22 = (unsigned __int64)&v62;
  v62 = (unsigned int *)v64;
  v63 = 0x800000000LL;
  v23 = sub_AF3990(v19, &v62);
  v24 = v62;
  v51 = v23;
  v57 = &v62[(unsigned int)v63];
  if ( v62 == v57 )
  {
    if ( !v23 && (_DWORD)v63 )
      goto LABEL_33;
    v35 = v58;
LABEL_31:
    v22 = v51;
    sub_CB59D0(v35, v51);
    goto LABEL_32;
  }
  v25 = v62;
  v26 = 1;
  do
  {
    while ( 1 )
    {
      v28 = sub_AF3870(*v25);
      v29 = v58;
      v30 = (const void *)v28;
      v32 = v31;
      if ( v26 )
      {
        v27 = *(void **)(v58 + 32);
        v26 = 0;
        goto LABEL_21;
      }
      v33 = *(_QWORD *)(v58 + 32);
      if ( (unsigned __int64)(*(_QWORD *)(v58 + 24) - v33) > 2 )
        break;
      v54 = v32;
      v56 = (const void *)v28;
      v50 = sub_CB6200(v58, " | ", 3);
      v30 = v56;
      v32 = v54;
      v27 = *(void **)(v50 + 32);
      v29 = v50;
LABEL_21:
      if ( *(_QWORD *)(v29 + 24) - (_QWORD)v27 >= v32 )
        goto LABEL_22;
LABEL_28:
      ++v25;
      sub_CB6200(v29, v30, v32);
      if ( v57 == v25 )
        goto LABEL_29;
    }
    *(_BYTE *)(v33 + 2) = 32;
    *(_WORD *)v33 = 31776;
    v27 = (void *)(*(_QWORD *)(v29 + 32) + 3LL);
    v34 = *(_QWORD *)(v29 + 24);
    *(_QWORD *)(v29 + 32) = v27;
    if ( v34 - (__int64)v27 < v32 )
      goto LABEL_28;
LABEL_22:
    if ( v32 )
    {
      v53 = v29;
      v55 = v32;
      memcpy(v27, v30, v32);
      *(_QWORD *)(v53 + 32) += v55;
    }
    ++v25;
  }
  while ( v57 != v25 );
LABEL_29:
  v22 = v51;
  if ( v51 || !(_DWORD)v63 )
  {
    v35 = sub_904010(v58, " | ");
    goto LABEL_31;
  }
LABEL_32:
  v24 = v62;
LABEL_33:
  if ( v24 != (unsigned int *)v64 )
    _libc_free(v24, v22);
LABEL_35:
  v36 = sub_A17150((_BYTE *)(a2 - 16));
  sub_A5CC00((__int64)&v58, "unit", 4u, *((_QWORD *)v36 + 5), 1);
  if ( (*(_BYTE *)(a2 - 16) & 2) != 0 )
    v37 = *(_DWORD *)(a2 - 24);
  else
    v37 = (*(_WORD *)(a2 - 16) >> 6) & 0xF;
  v38 = 0;
  if ( v37 > 9 )
    v38 = *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 9);
  sub_A5CC00((__int64)&v58, "templateParams", 0xEu, v38, 1);
  v39 = sub_A17150((_BYTE *)(a2 - 16));
  sub_A5CC00((__int64)&v58, "declaration", 0xBu, *((_QWORD *)v39 + 6), 1);
  v40 = sub_A17150((_BYTE *)(a2 - 16));
  sub_A5CC00((__int64)&v58, "retainedNodes", 0xDu, *((_QWORD *)v40 + 7), 1);
  if ( (*(_BYTE *)(a2 - 16) & 2) != 0 )
    v41 = *(_DWORD *)(a2 - 24);
  else
    v41 = (*(_WORD *)(a2 - 16) >> 6) & 0xF;
  v42 = 0;
  if ( v41 > 0xA )
    v42 = *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 10);
  sub_A5CC00((__int64)&v58, "thrownTypes", 0xBu, v42, 1);
  if ( (*(_BYTE *)(a2 - 16) & 2) != 0 )
    v43 = *(_DWORD *)(a2 - 24);
  else
    v43 = (*(_WORD *)(a2 - 16) >> 6) & 0xF;
  v44 = 0;
  if ( v43 > 0xB )
    v44 = *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 11);
  sub_A5CC00((__int64)&v58, "annotations", 0xBu, v44, 1);
  if ( (*(_BYTE *)(a2 - 16) & 2) != 0 )
    v45 = *(_DWORD *)(a2 - 24);
  else
    v45 = (*(_WORD *)(a2 - 16) >> 6) & 0xF;
  if ( v45 > 0xC && *((_QWORD *)sub_A17150((_BYTE *)(a2 - 16)) + 12) )
  {
    v46 = sub_A547D0(a2, 12);
    v48 = v47;
  }
  else
  {
    v48 = 0;
    v46 = 0;
  }
  sub_A53660(&v58, "targetFuncName", 0xEu, v46, v48, 1);
  return sub_904010(a1, ")");
}
