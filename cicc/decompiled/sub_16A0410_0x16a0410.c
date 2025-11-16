// Function: sub_16A0410
// Address: 0x16a0410
//
__int64 __fastcall sub_16A0410(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        unsigned int a6,
        double a7,
        double a8,
        double a9)
{
  void *v10; // rbx
  unsigned int v11; // r12d
  char v12; // al
  int v13; // r15d
  int v14; // r15d
  int v15; // r15d
  int v16; // r15d
  int v17; // r15d
  void **v18; // rax
  unsigned int v19; // r15d
  __int64 v20; // rdx
  char v21; // al
  void **v22; // rdi
  __int64 v23; // rdi
  __int64 *v25; // r12
  int v26; // r12d
  char v27; // al
  __int64 v28; // rdi
  __int64 *v29; // rsi
  int v30; // r12d
  int v31; // eax
  __int64 v32; // rsi
  int v33; // r12d
  int v34; // ebx
  int v35; // ebx
  int v36; // r12d
  int v37; // [rsp+Ch] [rbp-D4h]
  int v38; // [rsp+10h] [rbp-D0h]
  int v39; // [rsp+18h] [rbp-C8h]
  __int64 *v40; // [rsp+20h] [rbp-C0h]
  int v45; // [rsp+40h] [rbp-A0h]
  _BYTE v47[8]; // [rsp+50h] [rbp-90h] BYREF
  void *v48; // [rsp+58h] [rbp-88h] BYREF
  __int64 v49; // [rsp+60h] [rbp-80h]
  char v50; // [rsp+6Ah] [rbp-76h]
  _BYTE v51[8]; // [rsp+70h] [rbp-70h] BYREF
  __int64 v52[3]; // [rsp+78h] [rbp-68h] BYREF
  _BYTE v53[8]; // [rsp+90h] [rbp-50h] BYREF
  void *v54; // [rsp+98h] [rbp-48h] BYREF
  __int64 v55; // [rsp+A0h] [rbp-40h]
  char v56; // [rsp+AAh] [rbp-36h]

  v40 = (__int64 *)(a2 + 8);
  v10 = sub_16982C0();
  if ( *(void **)(a2 + 8) == v10 )
    sub_169C6E0(&v48, (__int64)v40);
  else
    sub_16986C0(&v48, (__int64 *)(a2 + 8));
  v11 = sub_16A0E20(v47, a4, a6);
  if ( v10 != v48 )
  {
    v12 = v50 & 7;
    if ( (v50 & 7) != 1 )
      goto LABEL_5;
LABEL_23:
    sub_169ED90((void **)(*(_QWORD *)(a1 + 8) + 8LL), &v48);
    v23 = *(_QWORD *)(a1 + 8);
    if ( v10 == *(void **)(v23 + 40) )
    {
LABEL_38:
      sub_169C980((void **)(v23 + 40), 0);
      goto LABEL_25;
    }
LABEL_24:
    sub_169B620(v23 + 40, 0);
    goto LABEL_25;
  }
  v12 = *(_BYTE *)(v49 + 26) & 7;
  if ( v12 == 1 )
    goto LABEL_23;
LABEL_5:
  if ( v12 )
  {
    if ( v10 == *(void **)(a2 + 8) )
      sub_169C6E0(v52, (__int64)v40);
    else
      sub_16986C0(v52, v40);
    v37 = sub_16A1420(v51, v47, a6) | v11;
    if ( v10 == (void *)v52[0] )
      sub_169C6E0(&v54, (__int64)v52);
    else
      sub_16986C0(&v54, v52);
    v45 = v37 | sub_16A0E20(v53, a4, a6);
    v13 = sub_16A0E20(v51, v47, a6) | v45;
    v14 = sub_16A1420(v51, a2, a6) | v13;
    if ( v10 == (void *)v52[0] )
      sub_169C8D0((__int64)v52, a7, a8, a9);
    else
      sub_1699490((__int64)v52);
    v15 = sub_16A0E20(v53, v51, a6) | v14;
    v16 = sub_16A0E20(v53, a3, a6) | v15;
    v17 = sub_16A0E20(v53, a5, a6) | v16;
    if ( v10 == v54 )
    {
      if ( (*(_BYTE *)(v55 + 26) & 7) != 3 )
        goto LABEL_15;
      v18 = (void **)(v55 + 8);
    }
    else
    {
      v18 = &v54;
      if ( (v56 & 7) != 3 )
        goto LABEL_15;
    }
    if ( (*((_BYTE *)v18 + 18) & 8) == 0 )
    {
      v11 = 0;
      sub_169ED90((void **)(*(_QWORD *)(a1 + 8) + 8LL), &v48);
      sub_169C9F0(*(_QWORD *)(a1 + 8) + 32LL, 0);
LABEL_21:
      sub_127D120(&v54);
      sub_127D120(v52);
      goto LABEL_25;
    }
LABEL_15:
    sub_16A0360((__int64 *)(*(_QWORD *)(a1 + 8) + 8LL), (__int64 *)&v48);
    v19 = sub_16A0E20(*(_QWORD *)(a1 + 8), v53, a6) | v17;
    v20 = *(_QWORD *)(a1 + 8);
    if ( v10 == *(void **)(v20 + 8) )
    {
      v21 = *(_BYTE *)(*(_QWORD *)(v20 + 16) + 26LL) & 7;
      if ( v21 != 1 )
      {
LABEL_17:
        v22 = (void **)(v20 + 40);
        if ( v21 )
        {
          sub_169ED90(v22, &v48);
          v34 = sub_16A1420(*(_QWORD *)(a1 + 8) + 32LL, *(_QWORD *)(a1 + 8), a6);
          v35 = sub_16A0E20(*(_QWORD *)(a1 + 8) + 32LL, v53, a6) | v34;
          sub_127D120(&v54);
          v11 = v19 | v35;
          sub_127D120(v52);
          goto LABEL_25;
        }
        goto LABEL_18;
      }
    }
    else
    {
      v21 = *(_BYTE *)(v20 + 26) & 7;
      if ( v21 != 1 )
        goto LABEL_17;
    }
    v22 = (void **)(v20 + 40);
LABEL_18:
    if ( v10 == *(void **)(v20 + 40) )
      sub_169C980(v22, 0);
    else
      sub_169B620((__int64)v22, 0);
    v11 = v19;
    goto LABEL_21;
  }
  v38 = sub_16A1030(a2, a4);
  v25 = (__int64 *)(a5 + 8);
  if ( v10 != v48 )
  {
    if ( v10 != *(void **)(a5 + 8) )
    {
      sub_1698680((__int64 *)&v48, (__int64 *)(a5 + 8));
      goto LABEL_32;
    }
LABEL_30:
    sub_127D120(&v48);
    if ( v10 == *(void **)(a5 + 8) )
      sub_169C6E0(&v48, (__int64)v25);
    else
      sub_16986C0(&v48, v25);
    goto LABEL_32;
  }
  if ( v10 != *(void **)(a5 + 8) )
    goto LABEL_30;
  sub_16A0170((__int64 *)&v48, (__int64 *)(a5 + 8));
LABEL_32:
  v39 = sub_16A0E20(v47, a3, a6);
  if ( v38 == 2 )
  {
    v36 = sub_16A0E20(v47, a4, a6);
    v11 = v39 | sub_16A0E20(v47, a2, a6) | v36;
  }
  else
  {
    v26 = sub_16A0E20(v47, a2, a6);
    v11 = v39 | sub_16A0E20(v47, a4, a6) | v26;
  }
  if ( v10 == v48 )
  {
    v27 = *(_BYTE *)(v49 + 26) & 7;
    if ( v27 == 1 )
      goto LABEL_36;
  }
  else
  {
    v27 = v50 & 7;
    if ( (v50 & 7) == 1 )
    {
LABEL_36:
      v28 = *(_QWORD *)(a1 + 8);
      goto LABEL_37;
    }
  }
  v28 = *(_QWORD *)(a1 + 8);
  if ( !v27 )
  {
LABEL_37:
    sub_169ED90((void **)(v28 + 8), &v48);
    v23 = *(_QWORD *)(a1 + 8);
    if ( v10 == *(void **)(v23 + 40) )
      goto LABEL_38;
    goto LABEL_24;
  }
  sub_16A0360((__int64 *)(v28 + 8), (__int64 *)&v48);
  v29 = (__int64 *)(a3 + 8);
  if ( v10 == *(void **)(a3 + 8) )
    sub_169C6E0(&v54, (__int64)v29);
  else
    sub_16986C0(&v54, v29);
  v30 = sub_16A0E20(v53, a5, a6) | v11;
  if ( v38 == 2 )
  {
    sub_16A0360((__int64 *)(*(_QWORD *)(a1 + 8) + 40LL), v40);
    v31 = sub_16A1420(*(_QWORD *)(a1 + 8) + 32LL, v47, a6);
    v32 = a4;
  }
  else
  {
    sub_16A0360((__int64 *)(*(_QWORD *)(a1 + 8) + 40LL), (__int64 *)(a4 + 8));
    v31 = sub_16A1420(*(_QWORD *)(a1 + 8) + 32LL, v47, a6);
    v32 = a2;
  }
  v33 = sub_16A0E20(*(_QWORD *)(a1 + 8) + 32LL, v32, a6) | v31 | v30;
  v11 = sub_16A0E20(*(_QWORD *)(a1 + 8) + 32LL, v53, a6) | v33;
  sub_127D120(&v54);
LABEL_25:
  sub_127D120(&v48);
  return v11;
}
