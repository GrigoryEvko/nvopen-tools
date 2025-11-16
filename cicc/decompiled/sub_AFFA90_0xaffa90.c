// Function: sub_AFFA90
// Address: 0xaffa90
//
__int64 __fastcall sub_AFFA90(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // r13d
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v9; // rcx
  unsigned __int8 v10; // dl
  unsigned __int8 v11; // dl
  __int64 v12; // rax
  int v13; // r13d
  int v14; // eax
  __int64 v15; // rsi
  int v16; // r8d
  _QWORD *v17; // rdi
  unsigned int v18; // eax
  _QWORD *v19; // rcx
  __int64 v20; // rdx
  int v21; // [rsp+0h] [rbp-40h] BYREF
  int v22; // [rsp+4h] [rbp-3Ch] BYREF
  __int64 v23; // [rsp+8h] [rbp-38h] BYREF
  __int64 v24[6]; // [rsp+10h] [rbp-30h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = *a2;
  v7 = *(_QWORD *)(a1 + 8);
  v9 = *a2 - 16;
  v21 = *(unsigned __int16 *)(*a2 + 2);
  v22 = *(_DWORD *)(v6 + 4);
  v10 = *(_BYTE *)(v6 - 16);
  if ( (v10 & 2) != 0 )
  {
    v23 = **(_QWORD **)(v6 - 32);
    v11 = *(_BYTE *)(v6 - 16);
    if ( (v11 & 2) != 0 )
    {
LABEL_5:
      v12 = *(_QWORD *)(v6 - 32);
      goto LABEL_6;
    }
  }
  else
  {
    v23 = *(_QWORD *)(v9 - 8LL * ((v10 >> 2) & 0xF));
    v11 = *(_BYTE *)(v6 - 16);
    if ( (v11 & 2) != 0 )
      goto LABEL_5;
  }
  v12 = v9 - 8LL * ((v11 >> 2) & 0xF);
LABEL_6:
  v13 = v4 - 1;
  v24[0] = *(_QWORD *)(v12 + 8);
  v14 = sub_AF7F90(&v21, &v22, &v23, v24);
  v15 = *a2;
  v16 = 1;
  v17 = 0;
  v18 = v13 & v14;
  v19 = (_QWORD *)(v7 + 8LL * v18);
  v20 = *v19;
  if ( *a2 == *v19 )
  {
LABEL_15:
    *a3 = v19;
    return 1;
  }
  else
  {
    while ( v20 != -4096 )
    {
      if ( v20 != -8192 || v17 )
        v19 = v17;
      v18 = v13 & (v16 + v18);
      v20 = *(_QWORD *)(v7 + 8LL * v18);
      if ( v20 == v15 )
      {
        v19 = (_QWORD *)(v7 + 8LL * v18);
        goto LABEL_15;
      }
      ++v16;
      v17 = v19;
      v19 = (_QWORD *)(v7 + 8LL * v18);
    }
    if ( !v17 )
      v17 = v19;
    *a3 = v17;
    return 0;
  }
}
