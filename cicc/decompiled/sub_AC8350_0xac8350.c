// Function: sub_AC8350
// Address: 0xac8350
//
__int64 __fastcall sub_AC8350(__int64 a1, __int64 a2, __int64 **a3)
{
  int v4; // r13d
  int v5; // r13d
  __int64 v6; // r8
  int v8; // r10d
  __int64 *v9; // r9
  char *v10; // r11
  unsigned int i; // ecx
  __int64 *v12; // r15
  __int64 v13; // r14
  __int64 v14; // rax
  unsigned int v15; // ecx
  __int64 v17; // rax
  const void *v18; // rsi
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rdx
  size_t v22; // rdx
  int v23; // eax
  __int64 v24; // rax
  unsigned int v25; // edi
  _QWORD *v26; // rdx
  _QWORD *v27; // rax
  __int64 v28; // rdi
  bool v29; // zf
  char v30; // al
  char v31; // al
  char *v32; // [rsp+0h] [rbp-90h]
  char *v33; // [rsp+8h] [rbp-88h]
  __int64 v34; // [rsp+8h] [rbp-88h]
  __int64 v35; // [rsp+8h] [rbp-88h]
  __int64 v36; // [rsp+10h] [rbp-80h]
  int v37; // [rsp+10h] [rbp-80h]
  int v38; // [rsp+10h] [rbp-80h]
  int v39; // [rsp+1Ch] [rbp-74h]
  unsigned int v40; // [rsp+1Ch] [rbp-74h]
  unsigned int v41; // [rsp+1Ch] [rbp-74h]
  unsigned int v42; // [rsp+20h] [rbp-70h]
  __int64 *v43; // [rsp+20h] [rbp-70h]
  __int64 *v44; // [rsp+20h] [rbp-70h]
  __int64 *v45; // [rsp+28h] [rbp-68h]
  __int64 v46; // [rsp+28h] [rbp-68h]
  char *v47; // [rsp+28h] [rbp-68h]
  char v48; // [rsp+30h] [rbp-60h] BYREF
  int v49; // [rsp+38h] [rbp-58h]
  __int64 v50; // [rsp+40h] [rbp-50h] BYREF
  char v51; // [rsp+50h] [rbp-40h]

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v5 = v4 - 1;
  v6 = *(_QWORD *)(a1 + 8);
  v8 = 1;
  v9 = 0;
  v10 = &v48;
  for ( i = v5 & *(_DWORD *)a2; ; i = v5 & v15 )
  {
    v12 = (__int64 *)(v6 + 8LL * i);
    v13 = *v12;
    v14 = *v12;
    if ( *v12 == -8192 || v13 == -4096 )
      goto LABEL_14;
    if ( *(_QWORD *)(a2 + 8) != *(_QWORD *)(v13 + 8) )
      goto LABEL_6;
    if ( *(unsigned __int8 *)(a2 + 16) != *(unsigned __int16 *)(v13 + 2) )
      goto LABEL_6;
    if ( *(_BYTE *)(a2 + 17) != *(_BYTE *)(v13 + 1) >> 1 )
      goto LABEL_6;
    v25 = *(_DWORD *)(v13 + 4) & 0x7FFFFFF;
    if ( *(_QWORD *)(a2 + 32) != v25 )
      goto LABEL_6;
    if ( !v25 )
      break;
    v26 = (_QWORD *)(v13 - 32LL * v25);
    v27 = *(_QWORD **)(a2 + 24);
    v28 = (__int64)&v27[v25];
    while ( *v27 == *v26 )
    {
      ++v27;
      v26 += 4;
      if ( (_QWORD *)v28 == v27 )
        goto LABEL_25;
    }
LABEL_6:
    v15 = v8 + i;
    ++v8;
  }
LABEL_25:
  if ( *(_WORD *)(v13 + 2) == 63 )
  {
    v33 = v10;
    v36 = v6;
    v39 = v8;
    v42 = i;
    v45 = v9;
    v17 = sub_AC35F0(v13);
    v9 = v45;
    i = v42;
    v18 = (const void *)v17;
    v20 = v19;
    v21 = *(_QWORD *)(a2 + 48);
    v8 = v39;
    v6 = v36;
    v10 = v33;
    if ( v21 != v20 )
    {
      v13 = *v12;
      goto LABEL_27;
    }
    v22 = 4 * v21;
    if ( v22 )
    {
      v23 = memcmp(*(const void **)(a2 + 40), v18, v22);
      v9 = v45;
      i = v42;
      v8 = v39;
      v6 = v36;
      v10 = v33;
      if ( v23 )
        goto LABEL_13;
    }
  }
  else if ( *(_QWORD *)(a2 + 48) )
  {
LABEL_27:
    v14 = v13;
    goto LABEL_14;
  }
  if ( *(_WORD *)(v13 + 2) != 34 )
  {
    if ( *(_QWORD *)(a2 + 56) )
      goto LABEL_13;
    goto LABEL_33;
  }
  v32 = v10;
  v34 = v6;
  v37 = v8;
  v40 = i;
  v43 = v9;
  v46 = *(_QWORD *)(a2 + 56);
  v24 = sub_AC5180(v13);
  v9 = v43;
  i = v40;
  v8 = v37;
  v6 = v34;
  v10 = v32;
  if ( v46 != v24 )
    goto LABEL_13;
  if ( *(_WORD *)(v13 + 2) != 34 )
  {
LABEL_33:
    v29 = *(_BYTE *)(a2 + 96) == 0;
    v51 = 0;
    if ( v29 )
      goto LABEL_34;
    goto LABEL_13;
  }
  sub_AC51A0((__int64)v32, v13);
  v10 = v32;
  v9 = v43;
  i = v40;
  v8 = v37;
  v6 = v34;
  if ( !*(_BYTE *)(a2 + 96) )
  {
    if ( !v51 )
      goto LABEL_34;
    sub_9963D0((__int64)v32);
    v6 = v34;
    v8 = v37;
    i = v40;
    v9 = v43;
    v10 = v32;
    goto LABEL_13;
  }
  if ( !v51 )
  {
LABEL_13:
    v14 = *v12;
    goto LABEL_14;
  }
  if ( *(_DWORD *)(a2 + 72) == v49 )
  {
    v30 = sub_AAD8B0(a2 + 64, v32);
    v10 = v32;
    v9 = v43;
    i = v40;
    v8 = v37;
    v6 = v34;
    if ( v30 )
    {
      v31 = sub_AAD8B0(a2 + 80, &v50);
      v10 = v32;
      if ( v31 )
      {
        sub_9963D0((__int64)v32);
LABEL_34:
        *a3 = v12;
        return 1;
      }
      v9 = v43;
      i = v40;
      v8 = v37;
      v6 = v34;
    }
  }
  v35 = v6;
  v38 = v8;
  v41 = i;
  v44 = v9;
  v47 = v10;
  sub_9963D0((__int64)v10);
  v14 = *v12;
  v10 = v47;
  v9 = v44;
  i = v41;
  v8 = v38;
  v6 = v35;
LABEL_14:
  if ( v14 != -4096 )
  {
    if ( !v9 && v14 == -8192 )
      v9 = v12;
    goto LABEL_6;
  }
  if ( !v9 )
    v9 = v12;
  *a3 = v9;
  return 0;
}
