// Function: sub_795190
// Address: 0x795190
//
__int64 __fastcall sub_795190(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 *a4)
{
  __int64 v4; // r13
  __int64 v7; // rbx
  __int64 v8; // rdx
  __int64 result; // rax
  _QWORD *v10; // rbx
  char i; // al
  __int64 v12; // r15
  size_t v13; // rdx
  __int64 v14; // rax
  char *v15; // rdi
  char *v16; // rax
  unsigned __int64 *v17; // rcx
  char *v18; // r15
  unsigned __int64 *v19; // rsi
  unsigned __int64 j; // r14
  _QWORD *v21; // rdx
  __int64 v22; // rdi
  unsigned __int64 v23; // r9
  int v24; // r9d
  int v25; // eax
  int v26; // edi
  unsigned int v27; // edx
  unsigned int v28; // eax
  __int64 v29; // rax
  unsigned int v30; // esi
  unsigned __int64 *v31; // [rsp+0h] [rbp-60h]
  unsigned __int64 *v32; // [rsp+0h] [rbp-60h]
  size_t v33; // [rsp+8h] [rbp-58h]
  unsigned int v34; // [rsp+8h] [rbp-58h]
  unsigned __int64 *v35; // [rsp+10h] [rbp-50h]
  unsigned __int64 *v36; // [rsp+10h] [rbp-50h]
  int v37; // [rsp+10h] [rbp-50h]
  size_t v38; // [rsp+10h] [rbp-50h]
  __int64 v40; // [rsp+18h] [rbp-48h]
  int v41[13]; // [rsp+2Ch] [rbp-34h] BYREF

  v4 = *(_QWORD *)(a2 + 72);
  if ( *(_BYTE *)(v4 + 24) == 20 )
    return *(_QWORD *)(v4 + 56);
  if ( (unsigned __int8)(*(_BYTE *)(a2 + 56) - 108) > 1u )
  {
    v7 = *(_QWORD *)(a1 + 16);
    v8 = qword_4F080A8;
    if ( (unsigned int)(0x10000 - (*(_DWORD *)(a1 + 16) - *(_DWORD *)(a1 + 24))) <= 0x2F )
    {
      sub_772E70((_QWORD *)(a1 + 16));
      v8 = qword_4F080A8;
      v7 = *(_QWORD *)(a1 + 16);
    }
    *(_QWORD *)(a1 + 16) = v7 + 48;
    *(_QWORD *)v7 = 0;
    *(_QWORD *)(v7 + 8) = v8;
    if ( (unsigned __int8)(*(_BYTE *)(v8 + 140) - 9) <= 2u )
      *(_QWORD *)(v7 + 16) = 0;
    if ( (unsigned int)sub_786210(a1, (_QWORD **)v4, v7 + 16, (char *)(v7 + 16)) )
    {
      if ( (*(_BYTE *)(v7 + 24) & 0x20) != 0 )
      {
        result = *(_QWORD *)(v7 + 32);
        if ( result )
          return result;
      }
      else
      {
        result = *(_QWORD *)(v7 + 16);
        if ( result )
          sub_721090();
      }
      goto LABEL_10;
    }
    return 0;
  }
  v10 = *(_QWORD **)v4;
  for ( i = *(_BYTE *)(*(_QWORD *)v4 + 140LL); i == 12; i = *((_BYTE *)v10 + 140) )
    v10 = (_QWORD *)v10[20];
  v41[0] = 1;
  if ( (unsigned __int8)(i - 2) > 1u )
  {
    v36 = a4;
    v25 = sub_7764B0(a1, (unsigned __int64)v10, v41);
    a4 = v36;
    v26 = v25;
    if ( (unsigned __int8)(*((_BYTE *)v10 + 140) - 8) > 3u )
    {
      v14 = (unsigned int)(v25 + 16);
      v13 = 8;
      v12 = 16;
    }
    else
    {
      v27 = (unsigned int)(v25 + 7) >> 3;
      v28 = v27 + 9;
      if ( (((_BYTE)v27 + 9) & 7) != 0 )
      {
        v30 = v27 + 17 - (((_BYTE)v27 + 9) & 7);
        v12 = v30;
        v14 = v26 + v30;
        v13 = v30 - 8LL;
      }
      else
      {
        v12 = v28;
        v14 = v26 + v28;
        v13 = v12 - 8;
      }
    }
    if ( (unsigned int)v14 > 0x400 )
    {
      v31 = v36;
      v33 = v13;
      v37 = v14 + 16;
      v29 = sub_822B10((unsigned int)(v14 + 16));
      v13 = v33;
      a4 = v31;
      *(_QWORD *)v29 = *(_QWORD *)(a1 + 32);
      v15 = (char *)(v29 + 16);
      *(_DWORD *)(v29 + 8) = v37;
      *(_DWORD *)(v29 + 12) = *(_DWORD *)(a1 + 40);
      *(_QWORD *)(a1 + 32) = v29;
      goto LABEL_19;
    }
    if ( (v14 & 7) != 0 )
      v14 = (_DWORD)v14 + 8 - (unsigned int)(v14 & 7);
  }
  else
  {
    v12 = 16;
    v13 = 8;
    v14 = 32;
  }
  v15 = *(char **)(a1 + 16);
  if ( 0x10000 - (*(_DWORD *)(a1 + 16) - *(_DWORD *)(a1 + 24)) < (unsigned int)v14 )
  {
    v32 = a4;
    v34 = v14;
    v38 = v13;
    sub_772E70((_QWORD *)(a1 + 16));
    v15 = *(char **)(a1 + 16);
    a4 = v32;
    v14 = v34;
    v13 = v38;
  }
  *(_QWORD *)(a1 + 16) = &v15[v14];
LABEL_19:
  v35 = a4;
  v16 = (char *)memset(v15, 0, v13);
  v17 = v35;
  v18 = &v16[v12];
  *((_QWORD *)v18 - 1) = v10;
  if ( (unsigned __int8)(*((_BYTE *)v10 + 140) - 9) <= 2u )
    *(_QWORD *)v18 = 0;
  *(_QWORD *)a3 = v18;
  if ( (*(_BYTE *)(a2 + 60) & 1) != 0 )
  {
    v19 = *(unsigned __int64 **)(v4 + 16);
    for ( j = *v19; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
    v21 = *(_QWORD **)(a1 + 16);
    v22 = qword_4F080A8;
    if ( (unsigned int)(0x10000 - (*(_DWORD *)(a1 + 16) - *(_DWORD *)(a1 + 24))) <= 0x2F )
    {
      sub_772E70((_QWORD *)(a1 + 16));
      v22 = qword_4F080A8;
      v21 = *(_QWORD **)(a1 + 16);
      v17 = v35;
    }
    v23 = (unsigned __int64)(v21 + 2);
    *(_QWORD *)(a1 + 16) = v21 + 6;
    *v21 = 0;
    v21[1] = v22;
    *v17 = (unsigned __int64)(v21 + 2);
    if ( (unsigned __int8)(*(_BYTE *)(v22 + 140) - 9) <= 2u )
    {
      v21[2] = 0;
      v23 = *v17;
    }
    if ( !(unsigned int)sub_794F10(a1, (__int64)v19, j, v23) )
      return 0;
  }
  v24 = sub_786210(a1, (_QWORD **)v4, (unsigned __int64)v18, v18);
  result = 0;
  if ( v24 )
  {
    result = *(_QWORD *)(*(_QWORD *)a3 + 8LL);
    if ( !result )
    {
LABEL_10:
      if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
      {
        v40 = result;
        sub_6855B0(0xA89u, (FILE *)(v4 + 28), (_QWORD *)(a1 + 96));
        sub_770D30(a1);
        return v40;
      }
    }
  }
  return result;
}
