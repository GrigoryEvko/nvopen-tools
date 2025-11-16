// Function: sub_12BFF60
// Address: 0x12bff60
//
__int64 __fastcall sub_12BFF60(__int64 a1, __int64 a2, unsigned int a3)
{
  unsigned int v4; // ebx
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // r13
  __int64 *v8; // rdx
  __int64 *v9; // rax
  __int64 v10; // rcx
  unsigned int v11; // r12d
  unsigned int v12; // eax
  __int64 *v13; // r8
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 v16; // rbx
  unsigned int v17; // eax
  __int64 *v18; // r9
  unsigned int v19; // ecx
  unsigned int v20; // edx
  __int64 v21; // rax
  __int64 *v22; // rbx
  __int64 v23; // rdx
  _QWORD *v24; // r12
  _QWORD *v25; // r13
  char *v26; // rdi
  __int64 v27; // r15
  unsigned int v28; // r14d
  __int64 v29; // rdx
  __int64 v30; // rax
  __int64 v31; // rdx
  unsigned int v32; // r12d
  __int64 v34; // rdx
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rdx
  _QWORD *v38; // r12
  _QWORD *v39; // r13
  char *v40; // rdi
  int v41; // r10d
  unsigned int v42; // edx
  __int64 v43; // rdi
  int v44; // r8d
  __int64 *v45; // rax
  char *v46; // rdi
  char *v47; // rdi
  unsigned int v48; // edx
  __int64 v49; // rdi
  int v50; // r8d
  unsigned __int8 v52; // [rsp+15h] [rbp-DBh]
  bool v53; // [rsp+16h] [rbp-DAh]
  char v54; // [rsp+17h] [rbp-D9h]
  __int64 *v55; // [rsp+18h] [rbp-D8h]
  const char *v56; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v57; // [rsp+28h] [rbp-C8h]
  __int64 v58; // [rsp+30h] [rbp-C0h] BYREF
  unsigned int v59; // [rsp+38h] [rbp-B8h]
  __int64 *v60; // [rsp+70h] [rbp-80h] BYREF
  __int64 v61; // [rsp+78h] [rbp-78h]
  _BYTE v62[112]; // [rsp+80h] [rbp-70h] BYREF

  v4 = a3 >> 4;
  v56 = "nvvmir.version";
  LOWORD(v58) = 259;
  v5 = sub_1632310(a2, &v56);
  v6 = (__int64)&v56;
  v7 = v5;
  LOWORD(v58) = 259;
  v52 = v4 & 1;
  v56 = "llvm.dbg.cu";
  v53 = sub_1632310(a2, &v56) != 0;
  if ( !v7 )
  {
    v46 = getenv("NVVM_IR_VER_CHK");
    if ( (!v46 || (unsigned int)strtol(v46, 0, 10)) && !(unsigned __int8)sub_12BDA30(a1, 1, 0) )
      return 3;
    if ( v53
      && v52
      && ((v47 = getenv("NVVM_IR_VER_CHK")) == 0 || (unsigned int)strtol(v47, 0, 10))
      && !(unsigned __int8)sub_12BD890(a1, 1, 0) )
    {
      return 3;
    }
    else
    {
      return 0;
    }
  }
  v56 = 0;
  v9 = &v58;
  v57 = 1;
  v10 = (__int64)&v58;
  do
    *v9++ = -8;
  while ( v9 != (__int64 *)&v60 );
  v11 = 0;
  v60 = (__int64 *)v62;
  v61 = 0x800000000LL;
  while ( (unsigned int)sub_161F520(v7, v6, v8, v10) > v11 )
  {
    v15 = sub_161F530(v7, v11);
    v16 = v15;
    v8 = (__int64 *)(v57 & 1);
    if ( (v57 & 1) != 0 )
    {
      v6 = 7;
      v10 = (__int64)&v58;
    }
    else
    {
      v6 = v59;
      v10 = v58;
      if ( !v59 )
      {
        v17 = v57;
        ++v56;
        v18 = 0;
        v19 = ((unsigned int)v57 >> 1) + 1;
        goto LABEL_13;
      }
      v6 = v59 - 1;
    }
    v12 = v6 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
    v13 = (__int64 *)(v10 + 8LL * ((unsigned int)v6 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4))));
    v14 = *v13;
    if ( v16 == *v13 )
      goto LABEL_7;
    v41 = 1;
    v18 = 0;
    while ( v14 != -8 )
    {
      if ( v18 || v14 != -16 )
        v13 = v18;
      v12 = v6 & (v41 + v12);
      v14 = *(_QWORD *)(v10 + 8LL * v12);
      if ( v16 == v14 )
        goto LABEL_7;
      ++v41;
      v18 = v13;
      v13 = (__int64 *)(v10 + 8LL * v12);
    }
    v17 = v57;
    if ( !v18 )
      v18 = v13;
    ++v56;
    v19 = ((unsigned int)v57 >> 1) + 1;
    if ( !(_BYTE)v8 )
    {
      v6 = v59;
LABEL_13:
      if ( 4 * v19 < 3 * (int)v6 )
        goto LABEL_14;
      goto LABEL_73;
    }
    v6 = 8;
    if ( 4 * v19 < 0x18 )
    {
LABEL_14:
      v20 = v6 - HIDWORD(v57) - v19;
      v10 = (unsigned int)v6 >> 3;
      if ( v20 > (unsigned int)v10 )
        goto LABEL_15;
      sub_12BFBB0((__int64)&v56, v6);
      if ( (v57 & 1) != 0 )
      {
        v10 = 7;
        v6 = (__int64)&v58;
      }
      else
      {
        v6 = v58;
        if ( !v59 )
        {
LABEL_118:
          LODWORD(v57) = (2 * ((unsigned int)v57 >> 1) + 2) | v57 & 1;
          BUG();
        }
        v10 = v59 - 1;
      }
      v48 = v10 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v18 = (__int64 *)(v6 + 8LL * v48);
      v17 = v57;
      v49 = *v18;
      if ( v16 == *v18 )
        goto LABEL_15;
      v50 = 1;
      v45 = 0;
      while ( v49 != -8 )
      {
        if ( !v45 && v49 == -16 )
          v45 = v18;
        v48 = v10 & (v50 + v48);
        v18 = (__int64 *)(v6 + 8LL * v48);
        v49 = *v18;
        if ( v16 == *v18 )
          goto LABEL_80;
        ++v50;
      }
      goto LABEL_78;
    }
LABEL_73:
    sub_12BFBB0((__int64)&v56, 2 * v6);
    if ( (v57 & 1) != 0 )
    {
      v10 = 7;
      v6 = (__int64)&v58;
    }
    else
    {
      v6 = v58;
      if ( !v59 )
        goto LABEL_118;
      v10 = v59 - 1;
    }
    v42 = v10 & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
    v18 = (__int64 *)(v6 + 8LL * v42);
    v17 = v57;
    v43 = *v18;
    if ( v16 == *v18 )
      goto LABEL_15;
    v44 = 1;
    v45 = 0;
    while ( v43 != -8 )
    {
      if ( v43 == -16 && !v45 )
        v45 = v18;
      v42 = v10 & (v44 + v42);
      v18 = (__int64 *)(v6 + 8LL * v42);
      v43 = *v18;
      if ( v16 == *v18 )
        goto LABEL_80;
      ++v44;
    }
LABEL_78:
    if ( v45 )
      v18 = v45;
LABEL_80:
    v17 = v57;
LABEL_15:
    LODWORD(v57) = (2 * (v17 >> 1) + 2) | v17 & 1;
    if ( *v18 != -8 )
      --HIDWORD(v57);
    *v18 = v16;
    v21 = (unsigned int)v61;
    if ( (unsigned int)v61 >= HIDWORD(v61) )
    {
      v6 = (__int64)v62;
      sub_16CD150(&v60, v62, 0, 8);
      v21 = (unsigned int)v61;
    }
    v8 = v60;
    v60[v21] = v16;
    LODWORD(v61) = v61 + 1;
LABEL_7:
    ++v11;
  }
  v22 = v60;
  v54 = 0;
  v55 = &v60[(unsigned int)v61];
  if ( v60 != v55 )
  {
    do
    {
      while ( 1 )
      {
        v27 = *v22;
        if ( !*v22 )
          goto LABEL_40;
        v28 = *(_DWORD *)(v27 + 8);
        if ( ((v28 - 2) & 0xFFFFFFFD) != 0 )
          goto LABEL_40;
        v6 = v28;
        v29 = *(_QWORD *)(v27 - 8LL * v28);
        v30 = 0;
        if ( *(_BYTE *)v29 == 1 )
        {
          v30 = *(_QWORD *)(v29 + 136);
          if ( *(_BYTE *)(v30 + 16) != 13 )
            v30 = 0;
        }
        v31 = *(_QWORD *)(v27 + 8 * (1LL - v28));
        if ( *(_BYTE *)v31 != 1 )
          goto LABEL_40;
        v23 = *(_QWORD *)(v31 + 136);
        if ( *(_BYTE *)(v23 + 16) != 13 || !v30 )
          goto LABEL_40;
        v24 = *(_QWORD **)(v30 + 24);
        if ( *(_DWORD *)(v30 + 32) > 0x40u )
          v24 = (_QWORD *)*v24;
        v25 = *(_QWORD **)(v23 + 24);
        if ( *(_DWORD *)(v23 + 32) > 0x40u )
          v25 = (_QWORD *)*v25;
        v26 = getenv("NVVM_IR_VER_CHK");
        if ( !v26 || (v6 = 0, (unsigned int)strtol(v26, 0, 10)) )
        {
          if ( v24 != (_QWORD *)2 || v25 )
          {
            v6 = (__int64)v24;
            if ( !(unsigned __int8)sub_12BDA30(a1, (__int64)v24, (__int64)v25) )
            {
LABEL_40:
              v32 = 3;
              v55 = v60;
              goto LABEL_41;
            }
          }
        }
        if ( v28 == 4 )
          break;
        if ( v55 == ++v22 )
          goto LABEL_61;
      }
      v6 = *(unsigned int *)(v27 + 8);
      v34 = *(_QWORD *)(v27 + 8 * (2 - v6));
      v35 = 0;
      if ( *(_BYTE *)v34 == 1 )
      {
        v35 = *(_QWORD *)(v34 + 136);
        if ( *(_BYTE *)(v35 + 16) != 13 )
          v35 = 0;
      }
      v36 = *(_QWORD *)(v27 + 8 * (3 - v6));
      if ( *(_BYTE *)v36 != 1 )
        goto LABEL_40;
      v37 = *(_QWORD *)(v36 + 136);
      if ( *(_BYTE *)(v37 + 16) != 13 || !v35 )
        goto LABEL_40;
      v38 = *(_QWORD **)(v35 + 24);
      if ( *(_DWORD *)(v35 + 32) > 0x40u )
        v38 = (_QWORD *)*v38;
      v39 = *(_QWORD **)(v37 + 24);
      if ( *(_DWORD *)(v37 + 32) > 0x40u )
        v39 = (_QWORD *)*v39;
      v40 = getenv("NVVM_IR_VER_CHK");
      if ( !v40 || (v6 = 0, (unsigned int)strtol(v40, 0, 10)) )
      {
        if ( v38 != (_QWORD *)3 || (unsigned __int64)v39 > 2 )
        {
          v6 = (__int64)v38;
          if ( !(unsigned __int8)sub_12BD890(a1, (__int64)v38, (__int64)v39) )
            goto LABEL_40;
        }
      }
      v54 = 1;
      ++v22;
    }
    while ( v55 != v22 );
LABEL_61:
    v55 = v60;
  }
  if ( (v52 & v53) == 0 || (v32 = 3, v54) )
    v32 = 0;
LABEL_41:
  if ( v55 != (__int64 *)v62 )
    _libc_free(v55, v6);
  if ( (v57 & 1) == 0 )
    j___libc_free_0(v58);
  return v32;
}
