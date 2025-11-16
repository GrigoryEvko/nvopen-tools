// Function: sub_3138420
// Address: 0x3138420
//
unsigned __int64 *__fastcall sub_3138420(unsigned __int64 *a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // r13
  __int64 *v8; // rsi
  __int64 v9; // r14
  unsigned __int64 v10; // rdi
  int v11; // eax
  _QWORD *v12; // rdi
  char *v13; // rdx
  __int64 v14; // r13
  __int64 v15; // rax
  const char *v16; // r14
  _BYTE *v17; // rax
  __int64 v18; // rax
  __int64 v19; // r12
  _QWORD *v20; // rax
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // r12
  __int64 v24; // rdx
  unsigned int v25; // esi
  char *v26; // rax
  __int64 v27; // rsi
  __int64 v28; // rdi
  unsigned __int64 v29; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // rdx
  __int64 v34; // rcx
  char *v35; // rdx
  __int64 v36; // r14
  __int64 v37; // rax
  __int64 v38; // [rsp+8h] [rbp-98h]
  __int64 v41; // [rsp+20h] [rbp-80h]
  __int64 v42; // [rsp+20h] [rbp-80h]
  __int64 v43; // [rsp+20h] [rbp-80h]
  __int64 v44; // [rsp+38h] [rbp-68h] BYREF
  const char *v45; // [rsp+40h] [rbp-60h] BYREF
  char *v46; // [rsp+48h] [rbp-58h]
  const char *v47; // [rsp+50h] [rbp-50h]
  __int16 v48; // [rsp+60h] [rbp-40h]

  v7 = a2[70];
  v8 = (__int64 *)a2[71];
  v9 = v7 + 48;
  if ( (__int64 *)(v7 + 48) == v8 )
  {
    v43 = *(_QWORD *)(v7 + 72);
    v45 = sub_BD5D20(v7);
    v48 = 773;
    v46 = v35;
    v47 = ".cont";
    v36 = sub_AA48A0(v7);
    v37 = sub_22077B0(0x50u);
    v38 = v37;
    if ( v37 )
    {
      v8 = (__int64 *)v36;
      sub_AA4D50(v37, v36, (__int64)&v45, v43, 0);
    }
  }
  else
  {
    if ( v8 )
      v8 -= 3;
    v48 = 257;
    v8 += 3;
    v38 = sub_F36990(v7, v8, 0, 0, 0, 0, (void **)&v45, 0);
    v10 = *(_QWORD *)(v7 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v9 == v10 )
    {
      v12 = 0;
    }
    else
    {
      if ( !v10 )
        BUG();
      v11 = *(unsigned __int8 *)(v10 - 24);
      v12 = (_QWORD *)(v10 - 24);
      if ( (unsigned int)(v11 - 30) >= 0xB )
        v12 = 0;
    }
    sub_B43D60(v12);
    a2[70] = v7;
    a2[71] = v9;
    *((_WORD *)a2 + 288) = 0;
  }
  v41 = *(_QWORD *)(v7 + 72);
  v45 = sub_BD5D20(v7);
  v48 = 773;
  v46 = v13;
  v47 = ".cncl";
  v14 = sub_AA48A0(v7);
  v15 = sub_22077B0(0x50u);
  v16 = (const char *)v15;
  if ( v15 )
  {
    v8 = (__int64 *)v14;
    sub_AA4D50(v15, v14, (__int64)&v45, v41, 0);
  }
  v48 = 257;
  v17 = (_BYTE *)sub_AD6530(*(_QWORD *)(a3 + 8), (__int64)v8);
  v18 = sub_92B530((unsigned int **)a2 + 64, 0x20u, a3, v17, (__int64)&v45);
  v48 = 257;
  v19 = v18;
  v20 = sub_BD2C40(72, 3u);
  v21 = (__int64)v20;
  if ( v20 )
    sub_B4C9A0((__int64)v20, v38, (__int64)v16, v19, 3u, 0, 0, 0);
  (*(void (__fastcall **)(__int64, __int64, const char **, __int64, __int64))(*(_QWORD *)a2[75] + 16LL))(
    a2[75],
    v21,
    &v45,
    a2[71],
    a2[72]);
  v22 = a2[64];
  v23 = v22;
  v42 = v22 + 16LL * *((unsigned int *)a2 + 130);
  if ( v22 != v42 )
  {
    do
    {
      v24 = *(_QWORD *)(v23 + 8);
      v25 = *(_DWORD *)v23;
      v23 += 16;
      sub_B99FD0(v21, v25, v24);
    }
    while ( v42 != v23 );
  }
  v26 = (char *)(v16 + 48);
  v27 = 0;
  a2[70] = (__int64)v16;
  v28 = 0;
  a2[71] = (__int64)(v16 + 48);
  *((_WORD *)a2 + 288) = 0;
  if ( *(_QWORD *)(a5 + 16) )
  {
    v46 = (char *)(v16 + 48);
    LOWORD(v47) = 0;
    v45 = v16;
    (*(void (__fastcall **)(__int64 *, __int64, const char **))(a5 + 24))(&v44, a5, &v45);
    v29 = v44 & 0xFFFFFFFFFFFFFFFELL;
    if ( (v44 & 0xFFFFFFFFFFFFFFFELL) != 0 )
      goto LABEL_17;
    v26 = (char *)a2[71];
    v28 = *((unsigned __int8 *)a2 + 576);
    v27 = *((unsigned __int8 *)a2 + 577);
    v16 = (const char *)a2[70];
  }
  v31 = *((unsigned int *)a2 + 2);
  v45 = v16;
  v46 = v26;
  v32 = 5 * v31;
  v33 = *a2;
  LOBYTE(v47) = v28;
  BYTE1(v47) = v27;
  v34 = v33 + 8 * v32 - 40;
  if ( !*(_QWORD *)(v34 + 16) )
    sub_4263D6(v28, v27, v33);
  (*(void (__fastcall **)(__int64 *, __int64, const char **))(v34 + 24))(&v44, v34, &v45);
  v29 = v44 & 0xFFFFFFFFFFFFFFFELL;
  if ( (v44 & 0xFFFFFFFFFFFFFFFELL) == 0 )
  {
    sub_A88F30((__int64)(a2 + 64), v38, *(_QWORD *)(v38 + 56), 1);
    *a1 = 1;
    return a1;
  }
LABEL_17:
  *a1 = v29 | 1;
  return a1;
}
