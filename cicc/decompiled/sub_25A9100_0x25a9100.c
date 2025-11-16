// Function: sub_25A9100
// Address: 0x25a9100
//
__int64 __fastcall sub_25A9100(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // rdi
  unsigned __int64 v6; // r13
  char *v7; // rcx
  const void *v8; // rax
  const void *v9; // rsi
  size_t v10; // rbx
  unsigned __int8 v12; // al
  unsigned __int64 v13; // r13
  const void *v14; // rax
  unsigned __int8 *v15; // r13
  _QWORD *v16; // rax

  v4 = a3;
  if ( ((a3 >> 1) & 3) == 0 )
  {
    v4 = a3 & 0xFFFFFFFFFFFFFFF8LL;
    v12 = *(_BYTE *)(a3 & 0xFFFFFFFFFFFFFFF8LL);
    if ( v12 <= 0x1Cu )
    {
      if ( v12 != 22 )
      {
        if ( v12 > 0x15u )
          goto LABEL_5;
        if ( v12 != 20 )
        {
          v15 = sub_BD3990((unsigned __int8 *)v4, a2);
          if ( *v15 )
            goto LABEL_5;
LABEL_33:
          v16 = (_QWORD *)sub_22077B0(8u);
          *v16 = v15;
          *(_DWORD *)a1 = 1;
          *(_QWORD *)(a1 + 8) = v16;
          *(_QWORD *)(a1 + 16) = v16 + 1;
          *(_QWORD *)(a1 + 24) = v16 + 1;
          return a1;
        }
LABEL_23:
        *(_DWORD *)a1 = 1;
        *(_QWORD *)(a1 + 8) = 0;
        *(_QWORD *)(a1 + 16) = 0;
        *(_QWORD *)(a1 + 24) = 0;
        return a1;
      }
      v4 = *(_QWORD *)(v4 + 24);
      if ( !(unsigned __int8)sub_310F810(v4) )
        goto LABEL_5;
    }
LABEL_16:
    *(_DWORD *)a1 = *(_DWORD *)(a2 + 8);
    v13 = *(_QWORD *)(a2 + 24) - *(_QWORD *)(a2 + 16);
    *(_QWORD *)(a1 + 8) = 0;
    *(_QWORD *)(a1 + 16) = 0;
    *(_QWORD *)(a1 + 24) = 0;
    if ( v13 )
    {
      if ( v13 > 0x7FFFFFFFFFFFFFF8LL )
        goto LABEL_34;
      v7 = (char *)sub_22077B0(v13);
    }
    else
    {
      v13 = 0;
      v7 = 0;
    }
    *(_QWORD *)(a1 + 8) = v7;
    *(_QWORD *)(a1 + 16) = v7;
    *(_QWORD *)(a1 + 24) = &v7[v13];
    v14 = *(const void **)(a2 + 24);
    v9 = *(const void **)(a2 + 16);
    v10 = *(_QWORD *)(a2 + 24) - (_QWORD)v9;
    if ( v14 == v9 )
      goto LABEL_10;
    goto LABEL_9;
  }
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
    v6 = 0;
    v7 = 0;
    goto LABEL_8;
  }
  v4 = a3 & 0xFFFFFFFFFFFFFFF8LL;
  if ( *(_BYTE *)(a3 & 0xFFFFFFFFFFFFFFF8LL) != 3 )
  {
    if ( !v4 || !(unsigned __int8)sub_310F860() )
      goto LABEL_5;
    goto LABEL_16;
  }
  if ( !(unsigned __int8)sub_310F8B0() )
    goto LABEL_5;
  v4 = *(_QWORD *)(v4 - 32);
  if ( *(_BYTE *)v4 == 20 )
    goto LABEL_23;
  v15 = sub_BD3990((unsigned __int8 *)v4, a2);
  if ( !*v15 )
    goto LABEL_33;
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
LABEL_34:
    sub_4261EA(v4, a2, a3);
  }
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
