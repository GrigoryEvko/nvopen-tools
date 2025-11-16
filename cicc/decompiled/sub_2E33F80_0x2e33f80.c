// Function: sub_2E33F80
// Address: 0x2e33f80
//
__int64 __fastcall sub_2E33F80(__int64 a1, __int64 a2, int a3, __int64 a4, __int64 a5, __int64 a6)
{
  char *v7; // rsi
  __int64 v8; // rax
  unsigned __int64 v9; // rcx
  unsigned __int64 v10; // rdx
  __int64 v11; // rdx
  _DWORD v13[5]; // [rsp+Ch] [rbp-14h] BYREF

  v13[0] = a3;
  v7 = *(char **)(a1 + 152);
  if ( *(char **)(a1 + 144) != v7 )
  {
    if ( v7 != *(char **)(a1 + 160) )
      goto LABEL_3;
LABEL_10:
    sub_2E33E00((unsigned __int64 *)(a1 + 144), v7, v13);
    v8 = *(unsigned int *)(a1 + 120);
    v9 = *(unsigned int *)(a1 + 124);
    v10 = v8 + 1;
    if ( v8 + 1 <= v9 )
      goto LABEL_7;
LABEL_11:
    sub_C8D5F0(a1 + 112, (const void *)(a1 + 128), v10, 8u, a5, a6);
    v8 = *(unsigned int *)(a1 + 120);
    goto LABEL_7;
  }
  v8 = *(unsigned int *)(a1 + 120);
  if ( (_DWORD)v8 )
    goto LABEL_6;
  if ( v7 == *(char **)(a1 + 160) )
    goto LABEL_10;
LABEL_3:
  if ( v7 )
  {
    *(_DWORD *)v7 = v13[0];
    v7 = *(char **)(a1 + 152);
  }
  v8 = *(unsigned int *)(a1 + 120);
  *(_QWORD *)(a1 + 152) = v7 + 4;
LABEL_6:
  v9 = *(unsigned int *)(a1 + 124);
  v10 = v8 + 1;
  if ( v8 + 1 > v9 )
    goto LABEL_11;
LABEL_7:
  v11 = *(_QWORD *)(a1 + 112);
  *(_QWORD *)(v11 + 8 * v8) = a2;
  ++*(_DWORD *)(a1 + 120);
  return sub_2E32160(a2, a1, v11, v9, a5, a6);
}
