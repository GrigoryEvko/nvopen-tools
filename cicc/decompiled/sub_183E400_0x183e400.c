// Function: sub_183E400
// Address: 0x183e400
//
__int64 __fastcall sub_183E400(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // rdi
  unsigned __int64 v6; // r13
  char *v7; // rcx
  const void *v8; // rax
  const void *v9; // rsi
  size_t v10; // rbx
  unsigned __int8 v12; // al
  __int64 v13; // r13
  __int64 *v14; // rax
  unsigned __int64 v15; // r13
  const void *v16; // rax

  v4 = a3;
  if ( ((a3 >> 1) & 3) != 0 )
  {
    if ( ((unsigned int)(a3 >> 1) & 3) - 1 > 1 )
    {
LABEL_5:
      *(_DWORD *)a1 = *(_DWORD *)(a2 + 40);
      v6 = *(_QWORD *)(a2 + 56) - *(_QWORD *)(a2 + 48);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 24) = 0;
      if ( v6 )
        goto LABEL_6;
      v7 = 0;
      goto LABEL_8;
    }
    v4 = a3 & 0xFFFFFFFFFFFFFFF8LL;
    if ( *(_BYTE *)((a3 & 0xFFFFFFFFFFFFFFF8LL) + 16) == 3 )
    {
      if ( !(unsigned __int8)sub_387E060() )
        goto LABEL_5;
      v4 = *(_QWORD *)(v4 - 24);
      if ( *(_BYTE *)(v4 + 16) != 15 )
        goto LABEL_22;
      goto LABEL_20;
    }
    if ( !v4 || !(unsigned __int8)sub_387E010() )
      goto LABEL_5;
LABEL_26:
    *(_DWORD *)a1 = *(_DWORD *)(a2 + 8);
    v15 = *(_QWORD *)(a2 + 24) - *(_QWORD *)(a2 + 16);
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 24) = 0;
    if ( v15 )
    {
      if ( v15 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_32;
      v7 = (char *)sub_22077B0(v15);
    }
    else
    {
      v15 = 0;
      v7 = 0;
    }
    *(_QWORD *)(a1 + 8) = v7;
    *(_QWORD *)(a1 + 16) = v7;
    *(_QWORD *)(a1 + 24) = &v7[v15];
    v16 = *(const void **)(a2 + 24);
    v9 = *(const void **)(a2 + 16);
    v10 = *(_QWORD *)(a2 + 24) - (_QWORD)v9;
    if ( v16 == v9 )
      goto LABEL_10;
    goto LABEL_9;
  }
  v4 = a3 & 0xFFFFFFFFFFFFFFF8LL;
  v12 = *(_BYTE *)((a3 & 0xFFFFFFFFFFFFFFF8LL) + 16);
  if ( v12 > 0x17u )
    goto LABEL_26;
  if ( v12 == 17 )
  {
    v4 = *(_QWORD *)(v4 + 24);
    if ( (unsigned __int8)sub_387DFE0(v4) )
      goto LABEL_26;
  }
  else if ( v12 <= 0x10u )
  {
    if ( v12 != 15 )
    {
LABEL_22:
      v13 = sub_1649C60(v4);
      if ( !*(_BYTE *)(v13 + 16) )
      {
        v14 = (__int64 *)sub_22077B0(8);
        *v14 = v13;
        *(_DWORD *)a1 = 1;
        *(_QWORD *)(a1 + 8) = v14;
        *(_QWORD *)(a1 + 16) = v14 + 1;
        *(_QWORD *)(a1 + 24) = v14 + 1;
        return a1;
      }
      goto LABEL_16;
    }
LABEL_20:
    *(_DWORD *)a1 = 1;
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 24) = 0;
    return a1;
  }
LABEL_16:
  *(_DWORD *)a1 = *(_DWORD *)(a2 + 40);
  v6 = *(_QWORD *)(a2 + 56) - *(_QWORD *)(a2 + 48);
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  if ( v6 )
  {
LABEL_6:
    if ( v6 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v7 = (char *)sub_22077B0(v6);
      goto LABEL_8;
    }
LABEL_32:
    sub_4261EA(v4, a2, a3);
  }
  v6 = 0;
  v7 = 0;
LABEL_8:
  *(_QWORD *)(a1 + 8) = v7;
  *(_QWORD *)(a1 + 16) = v7;
  *(_QWORD *)(a1 + 24) = &v7[v6];
  v8 = *(const void **)(a2 + 56);
  v9 = *(const void **)(a2 + 48);
  v10 = *(_QWORD *)(a2 + 56) - (_QWORD)v9;
  if ( v8 != v9 )
LABEL_9:
    v7 = (char *)memmove(v7, v9, v10);
LABEL_10:
  *(_QWORD *)(a1 + 16) = &v7[v10];
  return a1;
}
