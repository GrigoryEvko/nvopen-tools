// Function: sub_AFF130
// Address: 0xaff130
//
__int64 __fastcall sub_AFF130(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // r13d
  __int64 v6; // rax
  __int64 v7; // r14
  unsigned __int8 v9; // dl
  __int64 v10; // rcx
  unsigned __int8 v11; // dl
  __int64 v12; // rsi
  unsigned __int8 v13; // dl
  __int64 v14; // rcx
  int v15; // r13d
  int v16; // eax
  int v17; // r8d
  _QWORD *v18; // rdi
  unsigned int v19; // eax
  _QWORD *v20; // rcx
  __int64 v21; // rdx
  __int64 v22; // [rsp+0h] [rbp-40h] BYREF
  __int64 v23[2]; // [rsp+8h] [rbp-38h] BYREF
  int v24[10]; // [rsp+18h] [rbp-28h] BYREF

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
    v22 = **(_QWORD **)(v6 - 32);
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
    v22 = *(_QWORD *)(v10 - 8LL * ((v9 >> 2) & 0xF));
    v11 = *(_BYTE *)(v6 - 16);
    if ( (v11 & 2) != 0 )
      goto LABEL_5;
  }
  v12 = v10 - 8LL * ((v11 >> 2) & 0xF);
LABEL_6:
  v23[0] = *(_QWORD *)(v12 + 8);
  v13 = *(_BYTE *)(v6 - 16);
  if ( (v13 & 2) != 0 )
    v14 = *(_QWORD *)(v6 - 32);
  else
    v14 = v10 - 8LL * ((v13 >> 2) & 0xF);
  v15 = v4 - 1;
  v23[1] = *(_QWORD *)(v14 + 16);
  v24[0] = *(_DWORD *)(v6 + 4);
  v16 = sub_AF8830(&v22, v23, v24);
  v17 = 1;
  v18 = 0;
  v19 = v15 & v16;
  v20 = (_QWORD *)(v7 + 8LL * v19);
  v21 = *v20;
  if ( *v20 == *a2 )
  {
LABEL_18:
    *a3 = v20;
    return 1;
  }
  else
  {
    while ( v21 != -4096 )
    {
      if ( v21 != -8192 || v18 )
        v20 = v18;
      v19 = v15 & (v17 + v19);
      v21 = *(_QWORD *)(v7 + 8LL * v19);
      if ( v21 == *a2 )
      {
        v20 = (_QWORD *)(v7 + 8LL * v19);
        goto LABEL_18;
      }
      ++v17;
      v18 = v20;
      v20 = (_QWORD *)(v7 + 8LL * v19);
    }
    if ( !v18 )
      v18 = v20;
    *a3 = v18;
    return 0;
  }
}
