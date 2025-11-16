// Function: sub_163A800
// Address: 0x163a800
//
__int64 __fastcall sub_163A800(__int64 a1, _QWORD *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // r12
  unsigned int v8; // esi
  __int64 v9; // r11
  __int64 v10; // r13
  __int64 v11; // r9
  unsigned int v12; // edi
  _QWORD *v13; // rax
  __int64 v14; // rcx
  const void *v15; // r14
  size_t v16; // r15
  _QWORD *v17; // rdi
  __int64 v18; // rsi
  __int64 *v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  _QWORD *v23; // r10
  __int64 v24; // rax
  __int64 v25; // r13
  __int64 v26; // r14
  void (*v27)(); // rax
  __int64 result; // rax
  __int64 v29; // rax
  unsigned int v30; // r9d
  _QWORD *v31; // r10
  _QWORD *v32; // rcx
  void *v33; // rdi
  __int64 v34; // rax
  void *v35; // rax
  int v36; // eax
  int v37; // r9d
  __int64 v38; // rdi
  unsigned int v39; // eax
  int v40; // ecx
  _QWORD *v41; // rdx
  __int64 v42; // rsi
  int v43; // r11d
  _QWORD *v44; // r10
  __int64 v45; // rdi
  int v46; // r15d
  int v47; // eax
  int v48; // eax
  int v49; // esi
  __int64 v50; // rdi
  int v51; // r10d
  unsigned int v52; // r14d
  _QWORD *v53; // r9
  __int64 v54; // rax
  _QWORD *v55; // [rsp+8h] [rbp-68h]
  _QWORD *v56; // [rsp+8h] [rbp-68h]
  unsigned int v57; // [rsp+10h] [rbp-60h]
  _QWORD *v58; // [rsp+10h] [rbp-60h]
  _QWORD *v59; // [rsp+10h] [rbp-60h]
  _QWORD *v60; // [rsp+18h] [rbp-58h]
  unsigned int v61; // [rsp+18h] [rbp-58h]
  unsigned int v62; // [rsp+20h] [rbp-50h]
  char v63; // [rsp+2Ch] [rbp-44h]
  __int64 v64[7]; // [rsp+38h] [rbp-38h] BYREF

  v6 = a2;
  v63 = a3;
  if ( !(unsigned __int8)sub_16D5D40(a1, a2, a3, a4, a5, a6) )
  {
    v8 = *(_DWORD *)(a1 + 40);
    ++*(_DWORD *)(a1 + 12);
    v9 = a1 + 16;
    v10 = v6[4];
    if ( v8 )
      goto LABEL_3;
LABEL_25:
    ++*(_QWORD *)(a1 + 16);
    goto LABEL_26;
  }
  sub_16C9080(a1);
  v8 = *(_DWORD *)(a1 + 40);
  v10 = v6[4];
  v9 = a1 + 16;
  if ( !v8 )
    goto LABEL_25;
LABEL_3:
  v11 = *(_QWORD *)(a1 + 24);
  v12 = (v8 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
  v13 = (_QWORD *)(v11 + 16LL * v12);
  v14 = *v13;
  if ( v10 == *v13 )
    goto LABEL_4;
  v46 = 1;
  v41 = 0;
  while ( v14 != -4 )
  {
    if ( v14 != -8 || v41 )
      v13 = v41;
    v12 = (v8 - 1) & (v46 + v12);
    v14 = *(_QWORD *)(v11 + 16LL * v12);
    if ( v10 == v14 )
      goto LABEL_4;
    ++v46;
    v41 = v13;
    v13 = (_QWORD *)(v11 + 16LL * v12);
  }
  if ( !v41 )
    v41 = v13;
  v47 = *(_DWORD *)(a1 + 32);
  ++*(_QWORD *)(a1 + 16);
  v40 = v47 + 1;
  if ( 4 * (v47 + 1) >= 3 * v8 )
  {
LABEL_26:
    sub_1614D60(v9, 2 * v8);
    v36 = *(_DWORD *)(a1 + 40);
    if ( v36 )
    {
      v37 = v36 - 1;
      v38 = *(_QWORD *)(a1 + 24);
      v39 = (v36 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v40 = *(_DWORD *)(a1 + 32) + 1;
      v41 = (_QWORD *)(v38 + 16LL * v39);
      v42 = *v41;
      if ( v10 != *v41 )
      {
        v43 = 1;
        v44 = 0;
        while ( v42 != -4 )
        {
          if ( v42 == -8 && !v44 )
            v44 = v41;
          v39 = v37 & (v43 + v39);
          v41 = (_QWORD *)(v38 + 16LL * v39);
          v42 = *v41;
          if ( v10 == *v41 )
            goto LABEL_47;
          ++v43;
        }
        if ( v44 )
          v41 = v44;
      }
      goto LABEL_47;
    }
    goto LABEL_75;
  }
  if ( v8 - *(_DWORD *)(a1 + 36) - v40 <= v8 >> 3 )
  {
    sub_1614D60(v9, v8);
    v48 = *(_DWORD *)(a1 + 40);
    if ( v48 )
    {
      v49 = v48 - 1;
      v50 = *(_QWORD *)(a1 + 24);
      v51 = 1;
      v52 = (v48 - 1) & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
      v53 = 0;
      v40 = *(_DWORD *)(a1 + 32) + 1;
      v41 = (_QWORD *)(v50 + 16LL * v52);
      v54 = *v41;
      if ( v10 != *v41 )
      {
        while ( v54 != -4 )
        {
          if ( !v53 && v54 == -8 )
            v53 = v41;
          v52 = v49 & (v51 + v52);
          v41 = (_QWORD *)(v50 + 16LL * v52);
          v54 = *v41;
          if ( v10 == *v41 )
            goto LABEL_47;
          ++v51;
        }
        if ( v53 )
          v41 = v53;
      }
      goto LABEL_47;
    }
LABEL_75:
    ++*(_DWORD *)(a1 + 32);
    BUG();
  }
LABEL_47:
  *(_DWORD *)(a1 + 32) = v40;
  if ( *v41 != -4 )
    --*(_DWORD *)(a1 + 36);
  *v41 = v10;
  v41[1] = v6;
LABEL_4:
  v15 = (const void *)v6[2];
  v16 = v6[3];
  v17 = (_QWORD *)(a1 + 48);
  v18 = (__int64)v15;
  v20 = (unsigned int)sub_16D19C0(a1 + 48, v15, v16);
  v22 = v20;
  v23 = (_QWORD *)(*(_QWORD *)(a1 + 48) + 8 * v20);
  v24 = *v23;
  if ( *v23 )
  {
    if ( v24 != -8 )
      goto LABEL_6;
    --*(_DWORD *)(a1 + 64);
  }
  v55 = v23;
  v57 = v20;
  v29 = malloc(v16 + 17);
  v30 = v57;
  v31 = v55;
  v32 = (_QWORD *)v29;
  if ( !v29 )
  {
    if ( v16 == -17 )
    {
      v34 = malloc(1u);
      v30 = v57;
      v31 = v55;
      v32 = 0;
      if ( v34 )
      {
        v33 = (void *)(v34 + 16);
        v32 = (_QWORD *)v34;
        goto LABEL_23;
      }
    }
    v56 = v32;
    v59 = v31;
    v61 = v30;
    sub_16BD1C0("Allocation failed");
    v30 = v61;
    v31 = v59;
    v32 = v56;
  }
  v33 = v32 + 2;
  if ( v16 + 1 > 1 )
  {
LABEL_23:
    v58 = v32;
    v60 = v31;
    v62 = v30;
    v35 = memcpy(v33, v15, v16);
    v32 = v58;
    v31 = v60;
    v30 = v62;
    v33 = v35;
  }
  *((_BYTE *)v33 + v16) = 0;
  v18 = v30;
  v17 = (_QWORD *)(a1 + 48);
  *v32 = v16;
  v32[1] = 0;
  *v31 = v32;
  ++*(_DWORD *)(a1 + 60);
  v19 = (__int64 *)(*(_QWORD *)(a1 + 48) + 8LL * (unsigned int)sub_16D1CD0(a1 + 48, v30));
  v24 = *v19;
  if ( *v19 )
    goto LABEL_19;
  do
  {
    do
    {
      v24 = v19[1];
      ++v19;
    }
    while ( !v24 );
LABEL_19:
    ;
  }
  while ( v24 == -8 );
LABEL_6:
  *(_QWORD *)(v24 + 8) = v6;
  v25 = *(_QWORD *)(a1 + 104);
  v26 = *(_QWORD *)(a1 + 112);
  while ( v26 != v25 )
  {
    while ( 1 )
    {
      v17 = *(_QWORD **)v25;
      v27 = *(void (**)())(**(_QWORD **)v25 + 16LL);
      if ( v27 != nullsub_570 )
        break;
      v25 += 8;
      if ( v26 == v25 )
        goto LABEL_11;
    }
    v25 += 8;
    v18 = (__int64)v6;
    ((void (__fastcall *)(_QWORD *, _QWORD *, __int64 *, __int64, __int64, __int64))v27)(v17, v6, v19, v20, v21, v22);
  }
LABEL_11:
  if ( v63 )
  {
    v64[0] = (__int64)v6;
    v18 = *(_QWORD *)(a1 + 88);
    if ( v18 == *(_QWORD *)(a1 + 96) )
    {
      v17 = (_QWORD *)(a1 + 80);
      sub_163A620((char **)(a1 + 80), (char *)v18, v64);
      v6 = (_QWORD *)v64[0];
      if ( !v64[0] )
        goto LABEL_12;
    }
    else
    {
      if ( v18 )
      {
        *(_QWORD *)v18 = v6;
        *(_QWORD *)(a1 + 88) += 8LL;
        goto LABEL_12;
      }
      *(_QWORD *)(a1 + 88) = 8;
    }
    v45 = v6[6];
    if ( v45 )
      j_j___libc_free_0(v45, v6[8] - v45);
    v18 = 80;
    v17 = v6;
    j_j___libc_free_0(v6, 80);
  }
LABEL_12:
  result = sub_16D5D40(v17, v18, v19, v20, v21, v22);
  if ( (_BYTE)result )
    return sub_16C90A0(a1);
  --*(_DWORD *)(a1 + 12);
  return result;
}
