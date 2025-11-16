// Function: sub_1680240
// Address: 0x1680240
//
void __fastcall sub_1680240(__int64 a1, char a2)
{
  __int64 v3; // rax
  __int64 v4; // rbx
  char *v5; // rax
  char *v6; // r13
  int v7; // edx
  char *v8; // rax
  __int64 v9; // rbx
  __int64 v10; // r14
  int v11; // edx
  int v12; // eax
  void *v13; // rdi
  unsigned __int64 v14; // rsi
  const void ***v15; // r15
  char *v16; // r10
  unsigned __int64 v17; // r8
  const void ***v18; // r14
  const void **v19; // rcx
  __int64 v20; // r9
  unsigned __int64 v21; // rbx
  unsigned __int64 v22; // r13
  char *v23; // rsi
  int v24; // eax
  const void *v25; // rax
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rax
  bool v28; // zf
  _BYTE *v29; // [rsp+0h] [rbp-90h]
  __int64 v30; // [rsp+8h] [rbp-88h]
  const void **v31; // [rsp+10h] [rbp-80h]
  unsigned __int64 v32; // [rsp+18h] [rbp-78h]
  char *v33; // [rsp+20h] [rbp-70h]
  __int64 v34; // [rsp+38h] [rbp-58h] BYREF
  void *src; // [rsp+40h] [rbp-50h] BYREF
  char *v36; // [rsp+48h] [rbp-48h]
  char *v37; // [rsp+50h] [rbp-40h]

  *(_BYTE *)(a1 + 48) = 1;
  if ( !a2 )
    goto LABEL_2;
  v3 = *(unsigned int *)(a1 + 16);
  src = 0;
  v36 = 0;
  v37 = 0;
  if ( !v3 )
  {
    v13 = 0;
LABEL_27:
    v14 = 0;
    goto LABEL_28;
  }
  v4 = 8 * v3;
  v5 = (char *)sub_22077B0(8 * v3);
  v6 = v5;
  if ( v36 - (_BYTE *)src > 0 )
  {
    memmove(v5, src, v36 - (_BYTE *)src);
    j_j___libc_free_0(src, v37 - (_BYTE *)src);
  }
  v7 = *(_DWORD *)(a1 + 16);
  v8 = &v6[v4];
  src = v6;
  v36 = v6;
  v37 = &v6[v4];
  if ( !v7 || (v9 = *(_QWORD *)(a1 + 8), v10 = v9 + 24LL * *(unsigned int *)(a1 + 24), v9 == v10) )
  {
LABEL_46:
    v13 = v6;
    goto LABEL_27;
  }
  while ( 1 )
  {
    v11 = *(_DWORD *)(v9 + 12);
    if ( !v11 )
    {
      if ( *(_QWORD *)v9 != -1 )
        break;
      goto LABEL_45;
    }
    if ( v11 != 1 || *(_QWORD *)v9 != -2 )
      break;
LABEL_45:
    v9 += 24;
    if ( v10 == v9 )
      goto LABEL_46;
  }
  if ( v10 == v9 )
    goto LABEL_46;
  while ( 2 )
  {
    v34 = v9;
    if ( v6 == v8 )
    {
      sub_16800B0((__int64)&src, v6, &v34);
      v6 = v36;
    }
    else
    {
      if ( v6 )
      {
        *(_QWORD *)v6 = v9;
        v6 = v36;
      }
      v6 += 8;
      v36 = v6;
    }
    v9 += 24;
    if ( v9 == v10 )
      break;
    while ( 2 )
    {
      v12 = *(_DWORD *)(v9 + 12);
      if ( !v12 )
      {
        if ( *(_QWORD *)v9 != -1 )
          break;
        goto LABEL_24;
      }
      if ( v12 == 1 && *(_QWORD *)v9 == -2 )
      {
LABEL_24:
        v9 += 24;
        if ( v10 == v9 )
          goto LABEL_25;
        continue;
      }
      break;
    }
    if ( v10 != v9 )
    {
      v8 = v37;
      continue;
    }
    break;
  }
LABEL_25:
  v13 = src;
  v14 = (v6 - (_BYTE *)src) >> 3;
LABEL_28:
  sub_167F8F0((__int64)v13, v14, 0);
  sub_167FA60(a1);
  v15 = (const void ***)v36;
  v16 = 0;
  v17 = 0;
  v29 = src;
  v18 = (const void ***)src;
  if ( v36 != src )
  {
    while ( 1 )
    {
      v19 = *v18;
      v20 = *(_QWORD *)(a1 + 32);
      v21 = *(unsigned int *)(a1 + 44);
      v22 = *((unsigned int *)*v18 + 2);
      v23 = (char *)**v18;
      if ( v22 <= v17 )
      {
        if ( !*((_DWORD *)*v18 + 2) )
          goto LABEL_51;
        v30 = *(_QWORD *)(a1 + 32);
        v31 = *v18;
        v32 = v17;
        v33 = v16;
        v24 = memcmp(&v16[v17 - v22], v23, *((unsigned int *)*v18 + 2));
        v16 = v33;
        v17 = v32;
        v19 = v31;
        v20 = v30;
        if ( !v24 )
        {
LABEL_51:
          v25 = (const void *)(v20 - (*(_DWORD *)(a1 + 40) != 3) - v22);
          if ( ((unsigned int)v25 & ((_DWORD)v21 - 1)) == 0 )
            break;
        }
      }
      v26 = v21 * ((v21 + v20 - 1) / v21);
      v19[2] = (const void *)v26;
      v27 = v22 + v26;
      v28 = *(_DWORD *)(a1 + 40) == 3;
      *(_QWORD *)(a1 + 32) = v27;
      if ( v28 )
      {
        ++v18;
        v17 = v22;
        v16 = v23;
        if ( v15 == v18 )
          goto LABEL_38;
      }
      else
      {
        v17 = v22;
        v16 = v23;
        *(_QWORD *)(a1 + 32) = v27 + 1;
LABEL_31:
        if ( v15 == ++v18 )
          goto LABEL_38;
      }
    }
    v19[2] = v25;
    goto LABEL_31;
  }
LABEL_38:
  if ( v29 )
    j_j___libc_free_0(v29, v37 - v29);
LABEL_2:
  if ( *(_DWORD *)(a1 + 40) == 2 )
    *(_QWORD *)(a1 + 32) = (*(_QWORD *)(a1 + 32) + 3LL) & 0xFFFFFFFFFFFFFFFCLL;
}
