// Function: sub_1277C40
// Address: 0x1277c40
//
__int64 __fastcall sub_1277C40(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  int v5; // esi
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 *v9; // rdx
  __int64 v10; // r9
  _QWORD *v11; // rdx
  _QWORD *v12; // rax
  _QWORD *v13; // r13
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  __int64 v16; // r8
  __int64 v17; // rbx
  __int64 v18; // rsi
  char i; // al
  unsigned int v21; // eax
  __int64 v22; // rdx
  int v23; // edx
  int v24; // r10d

  v5 = *(_DWORD *)(a1 + 188);
  if ( *(_DWORD *)(a1 + 192) == v5 )
    goto LABEL_21;
  for ( ; *(_BYTE *)(a2 + 140) == 12; a2 = *(_QWORD *)(a2 + 160) )
    ;
  v7 = *(unsigned int *)(a1 + 48);
  if ( (_DWORD)v7 )
  {
    v8 = *(_QWORD *)(a1 + 32);
    a4 = ((_DWORD)v7 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v9 = (__int64 *)(v8 + 16 * a4);
    v10 = *v9;
    if ( a2 == *v9 )
    {
LABEL_6:
      if ( v9 != (__int64 *)(v8 + 16 * v7) && (*(_BYTE *)(v9[1] + 9) & 1) != 0 )
        goto LABEL_21;
    }
    else
    {
      v23 = 1;
      while ( v10 != -8 )
      {
        v24 = v23 + 1;
        a4 = ((_DWORD)v7 - 1) & (unsigned int)(v23 + a4);
        v9 = (__int64 *)(v8 + 16LL * (unsigned int)a4);
        v10 = *v9;
        if ( *v9 == a2 )
          goto LABEL_6;
        v23 = v24;
      }
    }
  }
  v11 = *(_QWORD **)(a1 + 176);
  v12 = *(_QWORD **)(a1 + 168);
  if ( v11 == v12 )
  {
    v13 = &v12[v5];
    if ( v12 == v13 )
    {
      v15 = *(_QWORD *)(a1 + 168);
    }
    else
    {
      do
      {
        if ( a2 == *v12 )
          break;
        ++v12;
      }
      while ( v13 != v12 );
      v15 = (unsigned __int64)v13;
    }
  }
  else
  {
    v13 = &v11[*(unsigned int *)(a1 + 184)];
    v12 = (_QWORD *)sub_16CC9F0(a1 + 160, a2);
    if ( a2 == *v12 )
    {
      v22 = *(_QWORD *)(a1 + 176);
      if ( v22 == *(_QWORD *)(a1 + 168) )
        a4 = *(unsigned int *)(a1 + 188);
      else
        a4 = *(unsigned int *)(a1 + 184);
      v15 = v22 + 8 * a4;
    }
    else
    {
      v14 = *(_QWORD *)(a1 + 176);
      if ( v14 != *(_QWORD *)(a1 + 168) )
      {
        v15 = *(unsigned int *)(a1 + 184);
        v12 = (_QWORD *)(v14 + 8 * v15);
        goto LABEL_12;
      }
      v12 = (_QWORD *)(v14 + 8LL * *(unsigned int *)(a1 + 188));
      v15 = (unsigned __int64)v12;
    }
  }
  if ( v12 != (_QWORD *)v15 )
  {
    while ( *v12 >= 0xFFFFFFFFFFFFFFFELL )
    {
      if ( (_QWORD *)v15 == ++v12 )
      {
        v16 = 0;
        if ( v13 != v12 )
          return (unsigned int)v16;
        goto LABEL_13;
      }
    }
  }
LABEL_12:
  v16 = 0;
  if ( v13 != v12 )
    return (unsigned int)v16;
LABEL_13:
  v17 = *(_QWORD *)(a2 + 160);
  if ( !v17 )
  {
LABEL_21:
    LODWORD(v16) = 1;
    return (unsigned int)v16;
  }
  while ( 1 )
  {
    v18 = *(_QWORD *)(v17 + 120);
    for ( i = *(_BYTE *)(v18 + 140); i == 12; i = *(_BYTE *)(v18 + 140) )
      v18 = *(_QWORD *)(v18 + 160);
    while ( i == 8 )
    {
      do
      {
        v18 = *(_QWORD *)(v18 + 160);
        i = *(_BYTE *)(v18 + 140);
      }
      while ( i == 12 );
    }
    if ( (unsigned __int8)(i - 10) <= 1u )
    {
      v21 = sub_1277C40(a1, v18, v15, a4, v16);
      v16 = v21;
      if ( !(_BYTE)v21 )
        return (unsigned int)v16;
    }
    v17 = *(_QWORD *)(v17 + 112);
    if ( !v17 )
      goto LABEL_21;
  }
}
