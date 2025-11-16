// Function: sub_198ECB0
// Address: 0x198ecb0
//
bool __fastcall sub_198ECB0(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v5; // r8
  int v6; // edi
  __int64 v7; // rsi
  int v8; // r11d
  unsigned int v9; // r10d
  __int64 *v10; // rax
  __int64 v11; // rdx
  int v12; // r10d
  int v13; // r11d
  unsigned int v14; // r9d
  __int64 *v15; // rax
  __int64 v16; // rdx
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rax
  int v22; // eax
  int v23; // eax
  int v24; // ebx
  int v25; // ebx

  v5 = *a1;
  v6 = *(_BYTE *)(*a1 + 8) & 1;
  if ( v6 )
  {
    v7 = v5 + 16;
    v8 = 15;
  }
  else
  {
    v18 = *(unsigned int *)(v5 + 24);
    v7 = *(_QWORD *)(v5 + 16);
    if ( !(_DWORD)v18 )
      goto LABEL_14;
    v8 = v18 - 1;
  }
  v9 = v8 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v10 = (__int64 *)(v7 + 16LL * v9);
  v11 = *v10;
  if ( a2 == *v10 )
    goto LABEL_4;
  v23 = 1;
  while ( v11 != -8 )
  {
    v25 = v23 + 1;
    v9 = v8 & (v23 + v9);
    v10 = (__int64 *)(v7 + 16LL * v9);
    v11 = *v10;
    if ( a2 == *v10 )
      goto LABEL_4;
    v23 = v25;
  }
  if ( (_BYTE)v6 )
  {
    v21 = 256;
    goto LABEL_15;
  }
  v18 = *(unsigned int *)(v5 + 24);
LABEL_14:
  v21 = 16 * v18;
LABEL_15:
  v10 = (__int64 *)(v7 + v21);
LABEL_4:
  v12 = *((_DWORD *)v10 + 2);
  if ( (_BYTE)v6 )
  {
    v13 = 15;
  }
  else
  {
    v19 = *(unsigned int *)(v5 + 24);
    v13 = v19 - 1;
    if ( !(_DWORD)v19 )
      goto LABEL_11;
  }
  v14 = v13 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v15 = (__int64 *)(v7 + 16LL * v14);
  v16 = *v15;
  if ( a3 == *v15 )
    return v12 < *((_DWORD *)v15 + 2);
  v22 = 1;
  while ( v16 != -8 )
  {
    v24 = v22 + 1;
    v14 = v13 & (v22 + v14);
    v15 = (__int64 *)(v7 + 16LL * v14);
    v16 = *v15;
    if ( a3 == *v15 )
      return v12 < *((_DWORD *)v15 + 2);
    v22 = v24;
  }
  if ( (_BYTE)v6 )
  {
    v20 = 256;
    return v12 < *(_DWORD *)(v7 + v20 + 8);
  }
  v19 = *(unsigned int *)(v5 + 24);
LABEL_11:
  v20 = 16 * v19;
  return v12 < *(_DWORD *)(v7 + v20 + 8);
}
