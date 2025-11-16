// Function: sub_26C8B80
// Address: 0x26c8b80
//
__int64 __fastcall sub_26C8B80(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v3; // rbx
  unsigned int v4; // r12d
  __int64 v5; // r8
  unsigned __int64 v6; // r14
  unsigned __int64 v7; // r13
  const void *v8; // rsi
  const void *v9; // rdi
  size_t v10; // rdx
  unsigned int v11; // eax
  __int64 v12; // rax
  char v13; // cl
  bool v14; // cf
  bool v15; // zf
  __int64 result; // rax
  unsigned __int64 v17; // r10
  unsigned __int64 v18; // r9
  unsigned __int64 v19; // r11
  __int64 v20; // r14
  __int64 v21; // r15
  __int64 v22; // r13
  unsigned __int64 v23; // rcx
  unsigned __int64 v24; // r9
  const void *v25; // rsi
  const void *v26; // rdi
  size_t v27; // rdx
  int v28; // eax
  unsigned int v29; // eax
  unsigned int v30; // eax
  unsigned int v31; // eax
  unsigned __int64 v33; // [rsp+10h] [rbp-60h]
  unsigned __int64 v34; // [rsp+18h] [rbp-58h]
  unsigned __int64 v35; // [rsp+20h] [rbp-50h]
  __int64 v36; // [rsp+28h] [rbp-48h]
  unsigned __int64 v37; // [rsp+30h] [rbp-40h]
  __int64 v38; // [rsp+38h] [rbp-38h]
  unsigned __int64 v39; // [rsp+38h] [rbp-38h]

  v2 = a2;
  v3 = *(_QWORD *)(a1 + 16);
  if ( !v3 )
  {
    v3 = a1 + 8;
    goto LABEL_47;
  }
  v4 = *(_DWORD *)(a2 + 32);
  v5 = a2;
  while ( 1 )
  {
    v14 = v4 < *(_DWORD *)(v3 + 64);
    if ( v4 != *(_DWORD *)(v3 + 64) )
    {
LABEL_15:
      if ( v14 )
        goto LABEL_16;
      goto LABEL_12;
    }
    if ( !v4 )
    {
      v6 = *(_QWORD *)(v3 + 40);
      v7 = *(_QWORD *)(v5 + 8);
      v8 = *(const void **)(v3 + 32);
      v9 = *(const void **)v5;
      if ( v6 < v7 )
      {
        if ( v9 == v8 )
          goto LABEL_11;
        v10 = *(_QWORD *)(v3 + 40);
      }
      else
      {
        if ( v9 == v8 )
          goto LABEL_10;
        v10 = *(_QWORD *)(v5 + 8);
      }
      if ( !v9 )
        goto LABEL_16;
      if ( !v8 )
        goto LABEL_12;
      v38 = v5;
      v11 = memcmp(v9, v8, v10);
      v5 = v38;
      if ( !v11 )
      {
LABEL_10:
        if ( v6 == v7 )
          goto LABEL_12;
LABEL_11:
        if ( v6 > v7 )
          goto LABEL_16;
        goto LABEL_12;
      }
      v31 = v11 >> 31;
      goto LABEL_50;
    }
    v17 = *(_QWORD *)(v3 + 56);
    v18 = *(_QWORD *)(v5 + 24);
    v19 = v18;
    if ( v17 <= v18 )
      v19 = *(_QWORD *)(v3 + 56);
    if ( v19 )
      break;
LABEL_41:
    LOBYTE(v31) = v17 > v18;
LABEL_50:
    if ( (_BYTE)v31 )
      goto LABEL_16;
LABEL_12:
    v12 = *(_QWORD *)(v3 + 24);
    v13 = 0;
    if ( !v12 )
      goto LABEL_17;
LABEL_13:
    v3 = v12;
  }
  v36 = 0;
  v20 = *(_QWORD *)(v3 + 48);
  v21 = v5;
  v34 = *(_QWORD *)(v3 + 56);
  v22 = *(_QWORD *)(v5 + 16);
  v33 = *(_QWORD *)(v5 + 24);
  v35 = v19;
  while ( 1 )
  {
    v23 = *(_QWORD *)(v20 + 8);
    v24 = *(_QWORD *)(v22 + 8);
    v25 = *(const void **)v20;
    v26 = *(const void **)v22;
    if ( v23 < v24 )
    {
      if ( v25 == v26 )
      {
LABEL_32:
        v5 = v21;
        if ( v23 <= v24 )
          goto LABEL_12;
        goto LABEL_16;
      }
      v27 = *(_QWORD *)(v20 + 8);
    }
    else
    {
      if ( v25 == v26 )
        goto LABEL_31;
      v27 = *(_QWORD *)(v22 + 8);
    }
    v37 = *(_QWORD *)(v22 + 8);
    v39 = *(_QWORD *)(v20 + 8);
    if ( !v26 )
    {
      v5 = v21;
      goto LABEL_16;
    }
    if ( !v25 )
    {
      v5 = v21;
      goto LABEL_12;
    }
    v28 = memcmp(v26, v25, v27);
    v23 = v39;
    v24 = v37;
    if ( v28 )
      break;
LABEL_31:
    if ( v23 != v24 )
      goto LABEL_32;
    v29 = *(_DWORD *)(v20 + 16);
    v14 = *(_DWORD *)(v22 + 16) < v29;
    if ( *(_DWORD *)(v22 + 16) != v29
      || (v30 = *(_DWORD *)(v20 + 20), v14 = *(_DWORD *)(v22 + 20) < v30, *(_DWORD *)(v22 + 20) != v30) )
    {
      v5 = v21;
      goto LABEL_15;
    }
    ++v36;
    v20 += 24;
    v22 += 24;
    if ( v36 == v35 )
    {
      v17 = v34;
      v18 = v33;
      v5 = v21;
      goto LABEL_41;
    }
  }
  v5 = v21;
  if ( v28 >= 0 )
    goto LABEL_12;
LABEL_16:
  v12 = *(_QWORD *)(v3 + 16);
  v13 = 1;
  if ( v12 )
    goto LABEL_13;
LABEL_17:
  v2 = v5;
  if ( !v13 )
  {
LABEL_18:
    v15 = sub_26BDDA0(v3 + 32, v2) == 0;
    result = v3;
    if ( !v15 )
      return 0;
    return result;
  }
LABEL_47:
  if ( *(_QWORD *)(a1 + 24) != v3 )
  {
    v3 = sub_220EF80(v3);
    goto LABEL_18;
  }
  return 0;
}
