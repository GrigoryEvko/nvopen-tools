// Function: sub_C93960
// Address: 0xc93960
//
void __fastcall sub_C93960(char **a1, __int64 a2, char a3, int a4, __int64 a5, __int64 a6)
{
  char *v6; // r13
  size_t v7; // r15
  int v8; // ebx
  int v9; // r12d
  _BYTE *v10; // rax
  size_t v11; // rax
  size_t v12; // r11
  __int64 v13; // rax
  _QWORD *v14; // rax
  __int64 v15; // rdx
  _QWORD *v16; // rdx
  size_t v17; // r11
  size_t v18; // [rsp+0h] [rbp-50h]
  size_t v19; // [rsp+8h] [rbp-48h]
  char v20; // [rsp+18h] [rbp-38h]
  char v21; // [rsp+1Fh] [rbp-31h]

  v6 = *a1;
  v7 = (size_t)a1[1];
  v20 = a5;
  v21 = a5;
  if ( !a4 )
  {
LABEL_20:
    if ( !v20 && !v7 )
      return;
    goto LABEL_10;
  }
  v8 = a3;
  v9 = a4;
  while ( v7 )
  {
    v10 = memchr(v6, v8, v7);
    if ( !v10 )
      goto LABEL_10;
    v11 = v10 - v6;
    if ( v11 == -1 )
      goto LABEL_10;
    v12 = v11 + 1;
    if ( !v11 && !v21 )
      goto LABEL_7;
    v15 = *(unsigned int *)(a2 + 8);
    if ( v11 > v7 )
      v11 = v7;
    a5 = v15 + 1;
    if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
    {
      v18 = v11;
      v19 = v12;
      sub_C8D5F0(a2, (const void *)(a2 + 16), v15 + 1, 0x10u, a5, a6);
      v15 = *(unsigned int *)(a2 + 8);
      v11 = v18;
      v12 = v19;
    }
    v16 = (_QWORD *)(*(_QWORD *)a2 + 16 * v15);
    *v16 = v6;
    v16[1] = v11;
    ++*(_DWORD *)(a2 + 8);
    if ( v12 > v7 )
    {
      v17 = v7;
      v7 = 0;
      v6 += v17;
      if ( !--v9 )
        goto LABEL_20;
    }
    else
    {
LABEL_7:
      v7 -= v12;
      v6 += v12;
      if ( !--v9 )
        goto LABEL_20;
    }
  }
  if ( v20 )
  {
LABEL_10:
    v13 = *(unsigned int *)(a2 + 8);
    if ( v13 + 1 > (unsigned __int64)*(unsigned int *)(a2 + 12) )
    {
      sub_C8D5F0(a2, (const void *)(a2 + 16), v13 + 1, 0x10u, a5, a6);
      v13 = *(unsigned int *)(a2 + 8);
    }
    v14 = (_QWORD *)(*(_QWORD *)a2 + 16 * v13);
    *v14 = v6;
    v14[1] = v7;
    ++*(_DWORD *)(a2 + 8);
  }
}
