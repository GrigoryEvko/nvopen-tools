// Function: sub_AFEAC0
// Address: 0xafeac0
//
__int64 __fastcall sub_AFEAC0(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // r14d
  __int64 v6; // rbx
  __int64 v7; // r15
  unsigned __int16 v9; // ax
  __int64 v10; // rdx
  unsigned __int8 v11; // al
  unsigned __int8 v12; // al
  __int64 v13; // rcx
  unsigned __int8 v14; // al
  __int64 v15; // rdx
  int v16; // r14d
  int v17; // eax
  __int64 v18; // rsi
  int v19; // r8d
  _QWORD *v20; // rdi
  unsigned int v21; // eax
  _QWORD *v22; // rcx
  __int64 v23; // rdx
  int v24; // [rsp+0h] [rbp-60h] BYREF
  __int64 v25; // [rsp+8h] [rbp-58h] BYREF
  __int64 v26; // [rsp+10h] [rbp-50h] BYREF
  __int8 v27[8]; // [rsp+18h] [rbp-48h] BYREF
  __int64 v28[8]; // [rsp+20h] [rbp-40h] BYREF

  v4 = *(_DWORD *)(a1 + 24);
  if ( !v4 )
  {
    *a3 = 0;
    return 0;
  }
  v6 = *a2;
  v7 = *(_QWORD *)(a1 + 8);
  v9 = sub_AF18C0(*a2);
  v10 = v6 - 16;
  v24 = v9;
  v11 = *(_BYTE *)(v6 - 16);
  if ( (v11 & 2) != 0 )
  {
    v25 = **(_QWORD **)(v6 - 32);
    v12 = *(_BYTE *)(v6 - 16);
    if ( (v12 & 2) != 0 )
    {
LABEL_5:
      v13 = *(_QWORD *)(v6 - 32);
      goto LABEL_6;
    }
  }
  else
  {
    v25 = *(_QWORD *)(v10 - 8LL * ((v11 >> 2) & 0xF));
    v12 = *(_BYTE *)(v6 - 16);
    if ( (v12 & 2) != 0 )
      goto LABEL_5;
  }
  v13 = v10 - 8LL * ((v12 >> 2) & 0xF);
LABEL_6:
  v26 = *(_QWORD *)(v13 + 8);
  v27[0] = *(_BYTE *)(v6 + 1) >> 7;
  v14 = *(_BYTE *)(v6 - 16);
  if ( (v14 & 2) != 0 )
    v15 = *(_QWORD *)(v6 - 32);
  else
    v15 = v10 - 8LL * ((v14 >> 2) & 0xF);
  v16 = v4 - 1;
  v28[0] = *(_QWORD *)(v15 + 16);
  v17 = sub_AF9230(&v24, &v25, &v26, v27, v28);
  v18 = *a2;
  v19 = 1;
  v20 = 0;
  v21 = v16 & v17;
  v22 = (_QWORD *)(v7 + 8LL * v21);
  v23 = *v22;
  if ( *a2 == *v22 )
  {
LABEL_18:
    *a3 = v22;
    return 1;
  }
  else
  {
    while ( v23 != -4096 )
    {
      if ( v23 != -8192 || v20 )
        v22 = v20;
      v21 = v16 & (v19 + v21);
      v23 = *(_QWORD *)(v7 + 8LL * v21);
      if ( v23 == v18 )
      {
        v22 = (_QWORD *)(v7 + 8LL * v21);
        goto LABEL_18;
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
