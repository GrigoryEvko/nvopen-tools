// Function: sub_394A9D0
// Address: 0x394a9d0
//
void __fastcall sub_394A9D0(__int64 a1, char **a2)
{
  char *v2; // r14
  char *v3; // rbx
  char *v4; // rbx
  int v5; // r12d
  unsigned int v6; // r15d
  int i; // r13d
  int v8; // edx
  char *v9; // r12
  int v10; // r11d
  unsigned __int64 v11; // rax
  int *v12; // rdi
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rax
  int v16; // r8d
  int v17; // r9d
  __int64 v18; // rdx
  __int64 v19; // r11
  unsigned __int64 v20; // rdi
  _BYTE *v21; // rsi
  __int64 v22; // [rsp+8h] [rbp-98h]
  __int64 v23; // [rsp+10h] [rbp-90h]
  unsigned __int64 *v25; // [rsp+28h] [rbp-78h]
  int v26; // [rsp+38h] [rbp-68h] BYREF
  unsigned int v27; // [rsp+3Ch] [rbp-64h] BYREF
  char v28[8]; // [rsp+40h] [rbp-60h] BYREF
  int v29; // [rsp+48h] [rbp-58h] BYREF
  unsigned __int64 v30; // [rsp+50h] [rbp-50h]
  int *v31; // [rsp+58h] [rbp-48h]
  int *v32; // [rsp+60h] [rbp-40h]
  __int64 v33; // [rsp+68h] [rbp-38h]

  if ( *(_BYTE *)a1 )
    return;
  v2 = *a2;
  v29 = 0;
  v31 = &v29;
  v32 = &v29;
  v3 = a2[1];
  v30 = 0;
  v4 = &v3[(_QWORD)v2];
  v33 = 0;
  v26 = 0;
  v27 = 0;
  if ( v2 == v4 )
  {
    v20 = 0;
LABEL_38:
    *(_BYTE *)a1 = 1;
LABEL_23:
    sub_394A3F0(v20);
    return;
  }
  v5 = *v2;
  v6 = 0;
  v25 = (unsigned __int64 *)(a1 + 32);
  for ( i = v5; ; i = v5 )
  {
    while ( 1 )
    {
      if ( v5 != 92 )
        goto LABEL_5;
LABEL_20:
      v9 = v2 + 1;
      if ( v4 == v2 + 1 )
        goto LABEL_36;
      i = v2[1];
      if ( (unsigned __int8)(v2[1] - 49) <= 8u )
      {
LABEL_22:
        v20 = v30;
        *(_BYTE *)a1 = 1;
        goto LABEL_23;
      }
LABEL_7:
      ++v6;
      v2 = v9 + 1;
      v27 = (i + (v27 << 8)) & 0xFFFFFF;
      if ( v6 > 2 && *(_DWORD *)(sub_394A760(v25, &v27) + 8) <= 3u )
        break;
      if ( v4 == v2 )
        goto LABEL_36;
      i = v9[1];
      v5 = i;
    }
    v10 = v26 + 1;
    v11 = v30;
    ++v26;
    if ( !v30 )
      goto LABEL_16;
    v12 = &v29;
    do
    {
      while ( 1 )
      {
        v13 = *(_QWORD *)(v11 + 16);
        v14 = *(_QWORD *)(v11 + 24);
        if ( *(_DWORD *)(v11 + 32) >= v27 )
          break;
        v11 = *(_QWORD *)(v11 + 24);
        if ( !v14 )
          goto LABEL_14;
      }
      v12 = (int *)v11;
      v11 = *(_QWORD *)(v11 + 16);
    }
    while ( v13 );
LABEL_14:
    if ( v12 == &v29 || v27 < v12[8] )
    {
LABEL_16:
      v15 = sub_394A760(v25, &v27);
      v18 = *(unsigned int *)(v15 + 8);
      v19 = (__int64)(*(_QWORD *)(a1 + 16) - *(_QWORD *)(a1 + 8)) >> 2;
      if ( (unsigned int)v18 >= *(_DWORD *)(v15 + 12) )
      {
        v22 = (__int64)(*(_QWORD *)(a1 + 16) - *(_QWORD *)(a1 + 8)) >> 2;
        v23 = v15;
        sub_16CD150(v15, (const void *)(v15 + 16), 0, 8, v16, v17);
        v15 = v23;
        v19 = v22;
        v18 = *(unsigned int *)(v23 + 8);
      }
      *(_QWORD *)(*(_QWORD *)v15 + 8 * v18) = v19;
      ++*(_DWORD *)(v15 + 8);
      sub_B99820((__int64)v28, &v27);
      if ( v4 == v2 )
      {
LABEL_36:
        v10 = v26;
        if ( !v26 )
          goto LABEL_37;
        goto LABEL_29;
      }
      goto LABEL_19;
    }
    if ( v4 == v2 )
      break;
LABEL_19:
    v5 = v9[1];
    i = v5;
    if ( v5 == 92 )
      goto LABEL_20;
LABEL_5:
    if ( strchr("()^$|+?[]\\{}", v5) )
      goto LABEL_22;
    v8 = v5;
    v9 = v2;
    if ( (v8 & 0xFFFFFFFB) != 0x2A )
      goto LABEL_7;
    v27 = 0;
    if ( v4 == v2 + 1 )
      goto LABEL_36;
    v5 = v2[1];
    v6 = 0;
    ++v2;
  }
  if ( !v10 )
  {
LABEL_37:
    v20 = v30;
    goto LABEL_38;
  }
LABEL_29:
  v21 = *(_BYTE **)(a1 + 16);
  if ( v21 == *(_BYTE **)(a1 + 24) )
  {
    sub_B8BBF0(a1 + 8, v21, &v26);
  }
  else
  {
    if ( v21 )
    {
      *(_DWORD *)v21 = v10;
      v21 = *(_BYTE **)(a1 + 16);
    }
    *(_QWORD *)(a1 + 16) = v21 + 4;
  }
  sub_394A3F0(v30);
}
