// Function: sub_254C2A0
// Address: 0x254c2a0
//
__int64 __fastcall sub_254C2A0(char *a1, __int64 a2)
{
  __int64 v3; // rdi
  char v4; // al
  _DWORD *v5; // r8
  _BYTE *v6; // rcx
  size_t v7; // rdx
  char *v8; // rsi
  __int64 v9; // rax
  __int64 result; // rax
  bool v11; // cc
  size_t v12; // rdx
  char *v13; // rsi
  _DWORD *v14; // rcx
  unsigned __int64 v15; // r9
  char *v16; // r8
  char *v17; // rsi
  unsigned int v18; // eax
  unsigned int v19; // ecx
  __int64 v20; // r8
  __int64 v21; // r8
  unsigned __int64 v22; // r9
  char *v23; // rcx
  char *v24; // rsi
  unsigned int v25; // ecx
  __int64 v26; // r8

  v3 = a2;
  v4 = *a1;
  v5 = *(_DWORD **)(a2 + 32);
  if ( *a1 == 2 )
  {
    v7 = 13;
    v8 = "positive-zero";
    goto LABEL_6;
  }
  v6 = *(_BYTE **)(a2 + 32);
  if ( v4 <= 2 )
  {
    if ( v4 )
    {
      if ( v4 != 1 )
        goto LABEL_8;
      v7 = 13;
      v8 = "preserve-sign";
    }
    else
    {
      v7 = 4;
      v8 = "ieee";
    }
LABEL_6:
    if ( v7 > *(_QWORD *)(v3 + 24) - (_QWORD)v5 )
      goto LABEL_7;
    goto LABEL_18;
  }
  v7 = 7;
  v8 = "dynamic";
  if ( v4 != 3 )
  {
LABEL_8:
    if ( *(_QWORD *)(v3 + 24) > (unsigned __int64)v6 )
      goto LABEL_9;
LABEL_24:
    v3 = sub_CB5D20(v3, 44);
    result = (unsigned __int8)a1[1];
    v11 = (char)result <= 2;
    if ( (_BYTE)result != 2 )
      goto LABEL_10;
LABEL_25:
    v14 = *(_DWORD **)(v3 + 32);
    v12 = 13;
    v13 = "positive-zero";
    if ( *(_QWORD *)(v3 + 24) - (_QWORD)v14 < 0xDu )
      return sub_CB6200(v3, (unsigned __int8 *)v13, v12);
    goto LABEL_26;
  }
  if ( *(_QWORD *)(v3 + 24) - (_QWORD)v5 < 7u )
  {
LABEL_7:
    v9 = sub_CB6200(v3, (unsigned __int8 *)v8, v7);
    v6 = *(_BYTE **)(v9 + 32);
    v3 = v9;
    goto LABEL_8;
  }
LABEL_18:
  if ( (unsigned int)v7 < 8 )
  {
    *v5 = *(_DWORD *)v8;
    *(_DWORD *)((char *)v5 + (unsigned int)v7 - 4) = *(_DWORD *)&v8[(unsigned int)v7 - 4];
    v21 = *(_QWORD *)(v3 + 32);
  }
  else
  {
    v15 = (unsigned __int64)(v5 + 2) & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v5 = *(_QWORD *)v8;
    *(_QWORD *)((char *)v5 + v7 - 8) = *(_QWORD *)&v8[v7 - 8];
    v16 = (char *)v5 - v15;
    v17 = (char *)(v8 - v16);
    if ( (((_DWORD)v7 + (_DWORD)v16) & 0xFFFFFFF8) >= 8 )
    {
      v18 = (v7 + (_DWORD)v16) & 0xFFFFFFF8;
      v19 = 0;
      do
      {
        v20 = v19;
        v19 += 8;
        *(_QWORD *)(v15 + v20) = *(_QWORD *)&v17[v20];
      }
      while ( v19 < v18 );
    }
    v21 = *(_QWORD *)(v3 + 32);
  }
  v6 = (_BYTE *)(v21 + v7);
  *(_QWORD *)(v3 + 32) = v21 + v7;
  if ( *(_QWORD *)(v3 + 24) <= v21 + v7 )
    goto LABEL_24;
LABEL_9:
  *(_QWORD *)(v3 + 32) = v6 + 1;
  *v6 = 44;
  result = (unsigned __int8)a1[1];
  v11 = (char)result <= 2;
  if ( (_BYTE)result == 2 )
    goto LABEL_25;
LABEL_10:
  if ( v11 )
  {
    if ( (_BYTE)result )
    {
      if ( (_BYTE)result != 1 )
        return result;
      v12 = 13;
      v13 = "preserve-sign";
    }
    else
    {
      v12 = 4;
      v13 = "ieee";
    }
  }
  else
  {
    v12 = 7;
    v13 = "dynamic";
    if ( (_BYTE)result != 3 )
      return result;
  }
  v14 = *(_DWORD **)(v3 + 32);
  if ( *(_QWORD *)(v3 + 24) - (_QWORD)v14 < v12 )
    return sub_CB6200(v3, (unsigned __int8 *)v13, v12);
LABEL_26:
  result = (unsigned int)v12;
  if ( (unsigned int)v12 < 8 )
  {
    *v14 = *(_DWORD *)v13;
    *(_DWORD *)((char *)v14 + (unsigned int)v12 - 4) = *(_DWORD *)&v13[(unsigned int)v12 - 4];
  }
  else
  {
    v22 = (unsigned __int64)(v14 + 2) & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v14 = *(_QWORD *)v13;
    *(_QWORD *)((char *)v14 + v12 - 8) = *(_QWORD *)&v13[v12 - 8];
    v23 = (char *)v14 - v22;
    v24 = (char *)(v13 - v23);
    result = ((_DWORD)v12 + (_DWORD)v23) & 0xFFFFFFF8;
    if ( (unsigned int)result >= 8 )
    {
      result = ((_DWORD)v12 + (_DWORD)v23) & 0xFFFFFFF8;
      v25 = 0;
      do
      {
        v26 = v25;
        v25 += 8;
        *(_QWORD *)(v22 + v26) = *(_QWORD *)&v24[v26];
      }
      while ( v25 < (unsigned int)result );
    }
  }
  *(_QWORD *)(v3 + 32) += v12;
  return result;
}
