// Function: sub_3892870
// Address: 0x3892870
//
__int64 __fastcall sub_3892870(__int64 a1)
{
  _BYTE *v2; // rsi
  __int64 v3; // rdx
  unsigned __int64 v4; // r13
  unsigned int v5; // r15d
  size_t v7; // r15
  unsigned int v8; // r8d
  _QWORD *v9; // r9
  __int64 v10; // rax
  unsigned int v11; // eax
  char *v12; // rdx
  unsigned int v13; // r9d
  _QWORD *v14; // r11
  __int64 v15; // rax
  __int64 v16; // rax
  unsigned int v17; // r8d
  _QWORD *v18; // r9
  _QWORD *v19; // rcx
  void *v20; // rdi
  __int64 *v21; // rdx
  __int64 v22; // rax
  void *v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  size_t v26; // rdx
  unsigned int v27; // r9d
  _QWORD *v28; // r11
  _QWORD *v29; // rcx
  _BYTE *v30; // rdi
  __int64 *v31; // rdx
  __int64 v32; // rax
  _BYTE *v33; // rax
  _QWORD *v34; // [rsp+8h] [rbp-B8h]
  _QWORD *v35; // [rsp+8h] [rbp-B8h]
  _QWORD *v36; // [rsp+10h] [rbp-B0h]
  unsigned int v37; // [rsp+10h] [rbp-B0h]
  _QWORD *v38; // [rsp+10h] [rbp-B0h]
  _QWORD *v39; // [rsp+10h] [rbp-B0h]
  unsigned int v40; // [rsp+10h] [rbp-B0h]
  unsigned int v41; // [rsp+18h] [rbp-A8h]
  _QWORD *v42; // [rsp+18h] [rbp-A8h]
  size_t v43; // [rsp+18h] [rbp-A8h]
  unsigned int v44; // [rsp+18h] [rbp-A8h]
  _QWORD *v45; // [rsp+18h] [rbp-A8h]
  _QWORD *v46; // [rsp+18h] [rbp-A8h]
  size_t v47; // [rsp+20h] [rbp-A0h]
  size_t v48; // [rsp+20h] [rbp-A0h]
  size_t v49; // [rsp+20h] [rbp-A0h]
  unsigned int v50; // [rsp+20h] [rbp-A0h]
  size_t v51; // [rsp+20h] [rbp-A0h]
  unsigned __int8 *v52; // [rsp+28h] [rbp-98h]
  unsigned int v53; // [rsp+28h] [rbp-98h]
  unsigned __int8 *src; // [rsp+30h] [rbp-90h]
  char *srca; // [rsp+30h] [rbp-90h]
  void *srcb; // [rsp+30h] [rbp-90h]
  void *srcc; // [rsp+30h] [rbp-90h]
  __int64 v58; // [rsp+38h] [rbp-88h]
  __int64 v59; // [rsp+48h] [rbp-78h] BYREF
  const char *v60; // [rsp+50h] [rbp-70h] BYREF
  char v61; // [rsp+60h] [rbp-60h]
  char v62; // [rsp+61h] [rbp-5Fh]
  unsigned __int8 *v63; // [rsp+70h] [rbp-50h] BYREF
  size_t n; // [rsp+78h] [rbp-48h]
  _QWORD v65[8]; // [rsp+80h] [rbp-40h] BYREF

  v2 = *(_BYTE **)(a1 + 72);
  v3 = *(_QWORD *)(a1 + 80);
  v63 = (unsigned __int8 *)v65;
  sub_3887850((__int64 *)&v63, v2, (__int64)&v2[v3]);
  v4 = *(_QWORD *)(a1 + 56);
  *(_DWORD *)(a1 + 64) = sub_3887100(a1 + 8);
  if ( (unsigned __int8)sub_388AF10(a1, 3, "expected '=' after name")
    || (unsigned __int8)sub_388AF10(a1, 199, "expected 'type' after name") )
  {
    goto LABEL_2;
  }
  v7 = n;
  v59 = 0;
  v58 = a1 + 728;
  src = v63;
  v8 = sub_16D19C0(a1 + 728, v63, n);
  v9 = (_QWORD *)(*(_QWORD *)(a1 + 728) + 8LL * v8);
  v10 = *v9;
  if ( *v9 )
  {
    if ( v10 != -8 )
      goto LABEL_9;
    --*(_DWORD *)(a1 + 744);
  }
  v36 = v9;
  v41 = v8;
  v16 = malloc(v7 + 25);
  v17 = v41;
  v18 = v36;
  v19 = (_QWORD *)v16;
  if ( !v16 )
  {
    if ( v7 == -25 )
    {
      v22 = malloc(1u);
      v17 = v41;
      v18 = v36;
      v19 = 0;
      if ( v22 )
      {
        v20 = (void *)(v22 + 24);
        v19 = (_QWORD *)v22;
        goto LABEL_24;
      }
    }
    v39 = v19;
    v45 = v18;
    v50 = v17;
    sub_16BD1C0("Allocation failed", 1u);
    v17 = v50;
    v18 = v45;
    v19 = v39;
  }
  v20 = v19 + 3;
  if ( v7 + 1 > 1 )
  {
LABEL_24:
    v42 = v19;
    v47 = (size_t)v18;
    v53 = v17;
    v23 = memcpy(v20, src, v7);
    v19 = v42;
    v18 = (_QWORD *)v47;
    v17 = v53;
    v20 = v23;
  }
  *((_BYTE *)v20 + v7) = 0;
  *v19 = v7;
  v19[1] = 0;
  v19[2] = 0;
  *v18 = v19;
  ++*(_DWORD *)(a1 + 740);
  v21 = (__int64 *)(*(_QWORD *)(a1 + 728) + 8LL * (unsigned int)sub_16D1CD0(v58, v17));
  v10 = *v21;
  if ( *v21 )
    goto LABEL_20;
  do
  {
    do
    {
      v10 = v21[1];
      ++v21;
    }
    while ( !v10 );
LABEL_20:
    ;
  }
  while ( v10 == -8 );
LABEL_9:
  v5 = sub_38925C0(a1, v4, v63, n, (__int64 *)(v10 + 8), &v59);
  if ( (_BYTE)v5 )
  {
LABEL_2:
    v5 = 1;
    goto LABEL_3;
  }
  if ( *(_BYTE *)(v59 + 8) == 13 )
    goto LABEL_3;
  srca = (char *)n;
  v52 = v63;
  v11 = sub_16D19C0(v58, v63, n);
  v12 = srca;
  v13 = v11;
  v14 = (_QWORD *)(*(_QWORD *)(a1 + 728) + 8LL * v11);
  v15 = *v14;
  if ( !*v14 )
  {
LABEL_27:
    v34 = v14;
    v37 = v13;
    v43 = (size_t)srca;
    v48 = (size_t)(srca + 25);
    srcb = srca + 1;
    v25 = malloc((unsigned __int64)(v12 + 25));
    v26 = v43;
    v27 = v37;
    v28 = v34;
    v29 = (_QWORD *)v25;
    if ( !v25 )
    {
      if ( !v48 )
      {
        v32 = malloc(1u);
        v26 = v43;
        v29 = 0;
        v27 = v37;
        v28 = v34;
        if ( v32 )
        {
          v30 = (_BYTE *)(v32 + 24);
          v29 = (_QWORD *)v32;
          goto LABEL_36;
        }
      }
      v35 = v28;
      v40 = v27;
      v46 = v29;
      v51 = v26;
      sub_16BD1C0("Allocation failed", 1u);
      v26 = v51;
      v29 = v46;
      v27 = v40;
      v28 = v35;
    }
    v30 = v29 + 3;
    if ( (unsigned __int64)srcb <= 1 )
    {
LABEL_29:
      v30[v26] = 0;
      *v29 = v26;
      v29[1] = 0;
      v29[2] = 0;
      *v28 = v29;
      ++*(_DWORD *)(a1 + 740);
      v31 = (__int64 *)(*(_QWORD *)(a1 + 728) + 8LL * (unsigned int)sub_16D1CD0(v58, v27));
      v15 = *v31;
      if ( *v31 == -8 || !v15 )
      {
        do
        {
          do
          {
            v15 = v31[1];
            ++v31;
          }
          while ( !v15 );
        }
        while ( v15 == -8 );
      }
      goto LABEL_13;
    }
LABEL_36:
    v38 = v28;
    v44 = v27;
    v49 = (size_t)v29;
    srcc = (void *)v26;
    v33 = memcpy(v30, v52, v26);
    v28 = v38;
    v27 = v44;
    v29 = (_QWORD *)v49;
    v26 = (size_t)srcc;
    v30 = v33;
    goto LABEL_29;
  }
  if ( v15 == -8 )
  {
    --*(_DWORD *)(a1 + 744);
    goto LABEL_27;
  }
LABEL_13:
  if ( *(_QWORD *)(v15 + 8) )
  {
    v62 = 1;
    v61 = 3;
    v60 = "non-struct types may not be recursive";
    v5 = sub_38814C0(a1 + 8, v4, (__int64)&v60);
  }
  else
  {
    v24 = v59;
    *(_QWORD *)(v15 + 16) = 0;
    *(_QWORD *)(v15 + 8) = v24;
  }
LABEL_3:
  if ( v63 != (unsigned __int8 *)v65 )
    j_j___libc_free_0((unsigned __int64)v63);
  return v5;
}
