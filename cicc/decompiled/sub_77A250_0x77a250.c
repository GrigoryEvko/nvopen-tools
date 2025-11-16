// Function: sub_77A250
// Address: 0x77a250
//
_QWORD *__fastcall sub_77A250(__int64 a1, unsigned __int64 a2, _DWORD *a3)
{
  unsigned __int64 v5; // rbx
  char i; // al
  __int64 v7; // r14
  size_t v8; // r8
  __int64 v9; // rax
  __int64 v10; // r15
  int v11; // ecx
  __int64 v12; // rdx
  bool v13; // zf
  char *v14; // rcx
  _QWORD *v15; // r14
  char *v16; // r15
  unsigned int v17; // r8d
  __int64 v18; // rdi
  unsigned int j; // eax
  unsigned __int64 *v20; // rdx
  int v21; // eax
  unsigned int v23; // eax
  int v24; // edx
  unsigned int v25; // eax
  int v26; // ecx
  __int64 v27; // rax
  __int64 v28; // rsi
  size_t v29; // [rsp+0h] [rbp-50h]
  size_t v30; // [rsp+0h] [rbp-50h]
  int v31; // [rsp+Ch] [rbp-44h]
  unsigned int v32; // [rsp+Ch] [rbp-44h]
  int v33[13]; // [rsp+1Ch] [rbp-34h] BYREF

  v5 = *(_QWORD *)(a2 + 120);
  v33[0] = 1;
  for ( i = *(_BYTE *)(v5 + 140); i == 12; i = *(_BYTE *)(v5 + 140) )
    v5 = *(_QWORD *)(v5 + 160);
  if ( (unsigned __int8)(i - 2) <= 1u )
  {
    if ( (unsigned __int8)(*(_BYTE *)(v5 + 140) - 8) > 3u )
    {
      v7 = 16;
      v8 = 8;
      v9 = 48;
      v10 = 16;
      goto LABEL_6;
    }
    v24 = 32;
    v10 = 16;
    v26 = 5;
    v25 = 13;
    goto LABEL_32;
  }
  v23 = sub_7764B0(a1, v5, v33);
  if ( !v33[0] )
  {
    *a3 = 0;
    return 0;
  }
  if ( (v23 & 7) != 0 )
    v23 = v23 + 8 - (v23 & 7);
  v10 = v23;
  v24 = v23 + 16;
  if ( (unsigned __int8)(*(_BYTE *)(v5 + 140) - 8) <= 3u )
  {
    v25 = ((v23 + 23) >> 3) + 9;
    v26 = v25 & 7;
    if ( (v25 & 7) == 0 )
    {
LABEL_26:
      v7 = v25;
      v8 = v25 - 8LL;
      goto LABEL_27;
    }
LABEL_32:
    v25 = v25 + 8 - v26;
    goto LABEL_26;
  }
  v8 = 8;
  v7 = 16;
  v25 = 16;
LABEL_27:
  v9 = v24 + v25;
  if ( (unsigned int)v9 > 0x400 )
  {
    v29 = v8;
    v31 = v9 + 16;
    v27 = sub_822B10((unsigned int)(v9 + 16));
    v8 = v29;
    *(_QWORD *)v27 = *(_QWORD *)(a1 + 32);
    v14 = (char *)(v27 + 16);
    *(_DWORD *)(v27 + 8) = v31;
    *(_DWORD *)(v27 + 12) = *(_DWORD *)(a1 + 40);
    *(_QWORD *)(a1 + 32) = v27;
    goto LABEL_11;
  }
LABEL_6:
  v11 = v9 & 7;
  v12 = (unsigned int)(v9 + 8 - v11);
  v13 = v11 == 0;
  v14 = *(char **)(a1 + 16);
  if ( !v13 )
    v9 = v12;
  if ( 0x10000 - ((int)v14 - *(_DWORD *)(a1 + 24)) < (unsigned int)v9 )
  {
    v30 = v8;
    v32 = v9;
    sub_772E70((_QWORD *)(a1 + 16));
    v14 = *(char **)(a1 + 16);
    v8 = v30;
    v9 = v32;
  }
  *(_QWORD *)(a1 + 16) = &v14[v9];
LABEL_11:
  v15 = (char *)memset(v14, 0, v8) + v7;
  *(v15 - 1) = v5;
  if ( (unsigned __int8)(*(_BYTE *)(v5 + 140) - 9) <= 2u )
    *v15 = 0;
  v16 = (char *)v15 + v10;
  *(_DWORD *)v16 = *(_DWORD *)(a1 + 40);
  v17 = *(_DWORD *)(a1 + 8);
  v18 = *(_QWORD *)a1;
  for ( j = v17 & (a2 >> 3); ; j = v17 & (j + 1) )
  {
    v20 = (unsigned __int64 *)(v18 + 16LL * j);
    if ( !*v20 )
      break;
    if ( *v20 == a2 )
    {
      v28 = v18 + 16LL * j;
      *((_QWORD *)v16 + 1) = *(_QWORD *)(v28 + 8);
      *(_QWORD *)(v28 + 8) = v15;
      return v15;
    }
  }
  *v20 = a2;
  v20[1] = (unsigned __int64)v15;
  v21 = *(_DWORD *)(a1 + 12) + 1;
  *(_DWORD *)(a1 + 12) = v21;
  if ( 2 * v21 > v17 )
    sub_7704A0(a1);
  *((_QWORD *)v16 + 1) = 0;
  return v15;
}
