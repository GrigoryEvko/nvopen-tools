// Function: sub_AFE570
// Address: 0xafe570
//
__int64 __fastcall sub_AFE570(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // r13d
  __int64 v6; // rax
  __int64 v7; // r14
  unsigned __int8 v9; // dl
  __int64 v10; // rcx
  unsigned __int8 v11; // dl
  __int64 v12; // rsi
  unsigned __int8 v13; // dl
  __int64 v14; // rsi
  unsigned __int8 v15; // dl
  __int64 v16; // rcx
  int v17; // r13d
  int v18; // eax
  int v19; // r8d
  _QWORD *v20; // rdi
  unsigned int v21; // eax
  _QWORD *v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // [rsp+0h] [rbp-50h] BYREF
  __int64 v25; // [rsp+8h] [rbp-48h] BYREF
  __int64 v26; // [rsp+10h] [rbp-40h] BYREF
  __int64 v27; // [rsp+18h] [rbp-38h] BYREF
  int v28[12]; // [rsp+20h] [rbp-30h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = *a2;
  v7 = *(_QWORD *)(a1 + 8);
  v9 = *(_BYTE *)(*a2 - 16);
  v10 = *a2 - 16;
  if ( (v9 & 2) != 0 )
  {
    v24 = **(_QWORD **)(v6 - 32);
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
    v24 = *(_QWORD *)(v10 - 8LL * ((v9 >> 2) & 0xF));
    v11 = *(_BYTE *)(v6 - 16);
    if ( (v11 & 2) != 0 )
      goto LABEL_5;
  }
  v12 = v10 - 8LL * ((v11 >> 2) & 0xF);
LABEL_6:
  v25 = *(_QWORD *)(v12 + 8);
  v13 = *(_BYTE *)(v6 - 16);
  if ( (v13 & 2) != 0 )
    v14 = *(_QWORD *)(v6 - 32);
  else
    v14 = v10 - 8LL * ((v13 >> 2) & 0xF);
  v26 = *(_QWORD *)(v14 + 16);
  v15 = *(_BYTE *)(v6 - 16);
  if ( (v15 & 2) != 0 )
    v16 = *(_QWORD *)(v6 - 32);
  else
    v16 = v10 - 8LL * ((v15 >> 2) & 0xF);
  v17 = v4 - 1;
  v27 = *(_QWORD *)(v16 + 24);
  v28[0] = *(_DWORD *)(v6 + 4);
  v18 = sub_AF9890(&v24, &v25, &v26, &v27, v28);
  v19 = 1;
  v20 = 0;
  v21 = v17 & v18;
  v22 = (_QWORD *)(v7 + 8LL * v21);
  v23 = *v22;
  if ( *v22 == *a2 )
  {
LABEL_21:
    *a3 = v22;
    return 1;
  }
  else
  {
    while ( v23 != -4096 )
    {
      if ( v23 != -8192 || v20 )
        v22 = v20;
      v21 = v17 & (v19 + v21);
      v23 = *(_QWORD *)(v7 + 8LL * v21);
      if ( v23 == *a2 )
      {
        v22 = (_QWORD *)(v7 + 8LL * v21);
        goto LABEL_21;
      }
      ++v19;
      v20 = v22;
      v22 = (_QWORD *)(v7 + 8LL * v21);
    }
    if ( !v20 )
      v20 = v22;
    *a3 = v20;
    return 0;
  }
}
