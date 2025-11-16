// Function: sub_26C3A50
// Address: 0x26c3a50
//
__int64 __fastcall sub_26C3A50(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  unsigned int v3; // r12d
  const void **v4; // r11
  const void *v5; // r14
  const void *v6; // r13
  const void *v7; // rsi
  const void *v8; // rdi
  size_t v9; // rdx
  int v10; // eax
  bool v11; // sf
  bool v12; // cf
  __int64 v13; // r15
  char *v15; // r9
  char *v16; // r8
  char *v17; // r10
  __int64 v18; // rax
  _QWORD *v19; // r14
  const void **v20; // r15
  __int64 v21; // r13
  unsigned int v22; // ebx
  __int64 v23; // r12
  unsigned __int64 v24; // rcx
  unsigned __int64 v25; // r8
  const void *v26; // rsi
  const void *v27; // rdi
  size_t v28; // rdx
  int v29; // eax
  __int64 v30; // rax
  unsigned int v31; // eax
  __int64 v32; // rax
  __int64 v33; // rax
  unsigned int v34; // eax
  __int64 v35; // rax
  __int64 v36; // rax
  __int64 v37; // rax
  char *v38; // [rsp+0h] [rbp-70h]
  char *v39; // [rsp+8h] [rbp-68h]
  char *v40; // [rsp+10h] [rbp-60h]
  char *v41; // [rsp+18h] [rbp-58h]
  unsigned __int64 v42; // [rsp+20h] [rbp-50h]
  const void **v43; // [rsp+28h] [rbp-48h]
  unsigned __int64 v44; // [rsp+28h] [rbp-48h]
  __int64 v45; // [rsp+30h] [rbp-40h]
  __int64 v46; // [rsp+38h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 16);
  v45 = a1 + 8;
  if ( !v2 )
    return v45;
  v46 = a1 + 8;
  v3 = *(_DWORD *)(a2 + 32);
  v4 = (const void **)a2;
  do
  {
    while ( 1 )
    {
      v12 = *(_DWORD *)(v2 + 64) < v3;
      if ( *(_DWORD *)(v2 + 64) != v3 )
      {
LABEL_14:
        if ( v12 )
          goto LABEL_15;
        goto LABEL_12;
      }
      if ( v3 )
        break;
      v5 = v4[1];
      v6 = *(const void **)(v2 + 40);
      v7 = *v4;
      v8 = *(const void **)(v2 + 32);
      if ( v5 < v6 )
      {
        if ( v7 == v8 )
        {
LABEL_11:
          if ( v5 > v6 )
            goto LABEL_15;
          goto LABEL_12;
        }
        v9 = (size_t)v4[1];
      }
      else
      {
        if ( v7 == v8 )
          goto LABEL_10;
        v9 = *(_QWORD *)(v2 + 40);
      }
      if ( !v8 )
        goto LABEL_15;
      if ( !v7 )
        goto LABEL_12;
      v43 = v4;
      v10 = memcmp(v8, v7, v9);
      v4 = v43;
      v11 = v10 < 0;
      if ( v10 )
      {
LABEL_37:
        if ( v11 )
          goto LABEL_15;
        goto LABEL_12;
      }
LABEL_10:
      if ( v5 != v6 )
        goto LABEL_11;
LABEL_12:
      v46 = v2;
      v2 = *(_QWORD *)(v2 + 16);
      if ( !v2 )
        goto LABEL_16;
    }
    v15 = (char *)v4[3];
    v16 = *(char **)(v2 + 56);
    v17 = v16;
    if ( v15 <= v16 )
      v17 = (char *)v4[3];
    if ( !v17 )
    {
LABEL_44:
      if ( v15 > v16 )
        goto LABEL_15;
      goto LABEL_12;
    }
    v39 = (char *)v4[3];
    v18 = v2;
    v19 = v4[2];
    v20 = v4;
    v41 = 0;
    v21 = *(_QWORD *)(v2 + 48);
    v22 = v3;
    v23 = v18;
    v38 = v16;
    v40 = v17;
    while ( 1 )
    {
      v24 = v19[1];
      v25 = *(_QWORD *)(v21 + 8);
      v26 = (const void *)*v19;
      v27 = *(const void **)v21;
      if ( v24 < v25 )
        break;
      if ( v27 == v26 )
        goto LABEL_31;
      v28 = *(_QWORD *)(v21 + 8);
LABEL_28:
      v42 = *(_QWORD *)(v21 + 8);
      v44 = v19[1];
      if ( !v27 )
      {
        v37 = v23;
        v4 = v20;
        v3 = v22;
        v2 = v37;
        goto LABEL_15;
      }
      if ( !v26 )
      {
        v36 = v23;
        v4 = v20;
        v3 = v22;
        v2 = v36;
        goto LABEL_12;
      }
      v29 = memcmp(v27, v26, v28);
      v24 = v44;
      v25 = v42;
      v11 = v29 < 0;
      if ( v29 )
      {
        v33 = v23;
        v4 = v20;
        v3 = v22;
        v2 = v33;
        goto LABEL_37;
      }
LABEL_31:
      if ( v24 != v25 )
        goto LABEL_32;
      v31 = *((_DWORD *)v19 + 4);
      v12 = *(_DWORD *)(v21 + 16) < v31;
      if ( *(_DWORD *)(v21 + 16) != v31
        || (v34 = *((_DWORD *)v19 + 5), v12 = *(_DWORD *)(v21 + 20) < v34, *(_DWORD *)(v21 + 20) != v34) )
      {
        v32 = v23;
        v4 = v20;
        v3 = v22;
        v2 = v32;
        goto LABEL_14;
      }
      ++v41;
      v19 += 3;
      v21 += 24;
      if ( v41 == v40 )
      {
        v35 = v23;
        v15 = v39;
        v16 = v38;
        v3 = v22;
        v4 = v20;
        v2 = v35;
        goto LABEL_44;
      }
    }
    if ( v27 != v26 )
    {
      v28 = v19[1];
      goto LABEL_28;
    }
LABEL_32:
    v30 = v23;
    v4 = v20;
    v3 = v22;
    v2 = v30;
    if ( v24 <= v25 )
      goto LABEL_12;
LABEL_15:
    v2 = *(_QWORD *)(v2 + 24);
  }
  while ( v2 );
LABEL_16:
  v13 = v46;
  if ( v45 != v46 )
  {
    if ( sub_26BDDA0((__int64)v4, v46 + 32) )
      return v45;
    return v13;
  }
  return v45;
}
