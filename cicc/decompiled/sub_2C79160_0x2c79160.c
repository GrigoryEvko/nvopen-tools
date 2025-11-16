// Function: sub_2C79160
// Address: 0x2c79160
//
void __fastcall sub_2C79160(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rbx
  unsigned __int8 v6; // al
  unsigned int v7; // esi
  __int64 v8; // r8
  __int64 v9; // r9
  int v10; // r11d
  _QWORD *v11; // r10
  unsigned int v12; // ecx
  _QWORD *v13; // rdx
  __int64 v14; // rax
  int v15; // eax
  int v16; // ecx
  __int64 v17; // rdi
  unsigned int v18; // eax
  __int64 v19; // rsi
  int v20; // edx
  int v21; // r9d
  _QWORD *v22; // r8
  int v23; // eax
  unsigned __int8 v24; // al
  char v25; // al
  size_t v26; // rax
  __int64 v27; // rax
  unsigned __int8 *v28; // rsi
  __int64 v29; // rdx
  char *v30; // rcx
  _BYTE *v31; // rax
  __int64 v32; // rdi
  __int64 *v33; // r12
  __int64 *v34; // r15
  __int64 v35; // rsi
  __int64 v36; // rax
  int v37; // eax
  int v38; // eax
  __int64 v39; // rsi
  int v40; // r8d
  _QWORD *v41; // rdi
  unsigned int v42; // r12d
  __int64 v43; // rcx
  unsigned __int8 *v44; // [rsp+0h] [rbp-50h] BYREF
  size_t v45; // [rsp+8h] [rbp-48h]
  _BYTE v46[64]; // [rsp+10h] [rbp-40h] BYREF

  v5 = a2;
  if ( *(_BYTE *)(a2 + 8) != 14 )
    goto LABEL_5;
  v6 = *(_BYTE *)a3;
  if ( *(_BYTE *)a3 <= 0x1Cu )
  {
    if ( v6 > 3u )
      return;
    v5 = *(_QWORD *)(a3 + 24);
LABEL_5:
    v7 = *(_DWORD *)(a1 + 144);
    v8 = a1 + 120;
    if ( v7 )
      goto LABEL_6;
LABEL_12:
    ++*(_QWORD *)(a1 + 120);
    goto LABEL_13;
  }
  if ( v6 != 60 )
  {
    if ( v6 == 62 )
    {
      v5 = *(_QWORD *)(*(_QWORD *)(a3 - 64) + 8LL);
      goto LABEL_5;
    }
    if ( v6 != 63 )
      return;
  }
  v7 = *(_DWORD *)(a1 + 144);
  v5 = *(_QWORD *)(a3 + 72);
  v8 = a1 + 120;
  if ( !v7 )
    goto LABEL_12;
LABEL_6:
  v9 = *(_QWORD *)(a1 + 128);
  v10 = 1;
  v11 = 0;
  v12 = (v7 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v13 = (_QWORD *)(v9 + 8LL * v12);
  v14 = *v13;
  if ( *v13 == v5 )
    return;
  while ( v14 != -4096 )
  {
    if ( v14 != -8192 || v11 )
      v13 = v11;
    v12 = (v7 - 1) & (v10 + v12);
    v14 = *(_QWORD *)(v9 + 8LL * v12);
    if ( v14 == v5 )
      return;
    ++v10;
    v11 = v13;
    v13 = (_QWORD *)(v9 + 8LL * v12);
  }
  v23 = *(_DWORD *)(a1 + 136);
  if ( !v11 )
    v11 = v13;
  ++*(_QWORD *)(a1 + 120);
  v20 = v23 + 1;
  if ( 4 * (v23 + 1) < 3 * v7 )
  {
    if ( v7 - *(_DWORD *)(a1 + 140) - v20 > v7 >> 3 )
      goto LABEL_29;
    sub_2C78F90(v8, v7);
    v37 = *(_DWORD *)(a1 + 144);
    if ( v37 )
    {
      v38 = v37 - 1;
      v39 = *(_QWORD *)(a1 + 128);
      v40 = 1;
      v41 = 0;
      v42 = v38 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v11 = (_QWORD *)(v39 + 8LL * v42);
      v43 = *v11;
      v20 = *(_DWORD *)(a1 + 136) + 1;
      if ( v5 != *v11 )
      {
        while ( v43 != -4096 )
        {
          if ( !v41 && v43 == -8192 )
            v41 = v11;
          v42 = v38 & (v40 + v42);
          v11 = (_QWORD *)(v39 + 8LL * v42);
          v43 = *v11;
          if ( *v11 == v5 )
            goto LABEL_29;
          ++v40;
        }
        if ( v41 )
          v11 = v41;
      }
      goto LABEL_29;
    }
LABEL_76:
    ++*(_DWORD *)(a1 + 136);
    BUG();
  }
LABEL_13:
  sub_2C78F90(v8, 2 * v7);
  v15 = *(_DWORD *)(a1 + 144);
  if ( !v15 )
    goto LABEL_76;
  v16 = v15 - 1;
  v17 = *(_QWORD *)(a1 + 128);
  v18 = (v15 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
  v11 = (_QWORD *)(v17 + 8LL * v18);
  v19 = *v11;
  v20 = *(_DWORD *)(a1 + 136) + 1;
  if ( *v11 != v5 )
  {
    v21 = 1;
    v22 = 0;
    while ( v19 != -4096 )
    {
      if ( v19 == -8192 && !v22 )
        v22 = v11;
      v18 = v16 & (v21 + v18);
      v11 = (_QWORD *)(v17 + 8LL * v18);
      v19 = *v11;
      if ( *v11 == v5 )
        goto LABEL_29;
      ++v21;
    }
    if ( v22 )
      v11 = v22;
  }
LABEL_29:
  *(_DWORD *)(a1 + 136) = v20;
  if ( *v11 != -4096 )
    --*(_DWORD *)(a1 + 140);
  *v11 = v5;
  v24 = *(_BYTE *)(v5 + 8);
  if ( v24 == 16 || (unsigned int)v24 - 17 <= 1 )
  {
    sub_2C79160(a1, *(_QWORD *)(v5 + 24), a3);
  }
  else if ( v24 == 15 )
  {
    v33 = *(__int64 **)(v5 + 16);
    v34 = &v33[*(unsigned int *)(v5 + 12)];
    while ( v33 != v34 )
    {
      v35 = *v33++;
      sub_2C79160(a1, v35, a3);
    }
  }
  v45 = 0;
  v46[0] = 0;
  v25 = *(_BYTE *)(v5 + 8);
  v44 = v46;
  switch ( v25 )
  {
    case 5:
      sub_2241130((unsigned __int64 *)&v44, 0, 0, "fp128 type is not supported\n", 0x1Cu);
      v26 = v45;
      break;
    case 6:
      sub_2241130((unsigned __int64 *)&v44, 0, 0, "ppc_fp128 type is not supported\n", 0x20u);
      v26 = v45;
      break;
    case 4:
      sub_2241130((unsigned __int64 *)&v44, 0, 0, "x86_fp80 type is not supported\n", 0x1Fu);
      v26 = v45;
      break;
    default:
      return;
  }
  if ( v26 )
  {
    if ( *(_BYTE *)a3 <= 0x1Cu )
    {
      if ( *(_BYTE *)a3 == 3 )
      {
        v36 = sub_2C767C0(a1, a3, 0);
        v28 = v44;
        sub_CB6200(v36, v44, v45);
      }
      else
      {
        sub_904010(*(_QWORD *)(a1 + 24), "Error: ");
        v28 = v44;
        sub_CB6200(*(_QWORD *)(a1 + 24), v44, v45);
      }
    }
    else
    {
      v27 = sub_2C76A00(a1, a3, 0);
      v28 = v44;
      sub_CB6200(v27, v44, v45);
    }
    v31 = *(_BYTE **)(a1 + 16);
    if ( v31 )
      *v31 = 0;
    if ( !*(_DWORD *)(a1 + 4) )
    {
      v32 = *(_QWORD *)(a1 + 24);
      if ( *(_QWORD *)(v32 + 32) != *(_QWORD *)(v32 + 16) )
      {
        sub_CB5AE0((__int64 *)v32);
        v32 = *(_QWORD *)(a1 + 24);
      }
      sub_CEB520(*(_QWORD **)(v32 + 48), (__int64)v28, v29, v30);
    }
  }
  if ( v44 != v46 )
    j_j___libc_free_0((unsigned __int64)v44);
}
