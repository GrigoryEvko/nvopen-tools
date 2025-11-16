// Function: sub_AFF510
// Address: 0xaff510
//
__int64 __fastcall sub_AFF510(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // r14d
  __int64 v6; // rbx
  __int64 v7; // r15
  unsigned __int8 v9; // al
  __int64 v10; // rdx
  unsigned __int8 v11; // al
  __int64 v12; // rcx
  unsigned __int8 v13; // al
  __int64 v14; // rdx
  int v15; // r14d
  int v16; // eax
  __int64 v17; // rsi
  _QWORD *v18; // rdi
  unsigned int v19; // eax
  int v20; // r8d
  _QWORD *v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // [rsp+8h] [rbp-78h]
  __int64 v24; // [rsp+10h] [rbp-70h] BYREF
  __int64 v25; // [rsp+18h] [rbp-68h] BYREF
  int v26; // [rsp+20h] [rbp-60h] BYREF
  __int64 v27; // [rsp+28h] [rbp-58h] BYREF
  __int64 v28; // [rsp+30h] [rbp-50h] BYREF
  int v29; // [rsp+38h] [rbp-48h] BYREF
  __int64 v30[8]; // [rsp+40h] [rbp-40h] BYREF

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
  v23 = v10;
  v25 = *(_QWORD *)(v12 + 8);
  v26 = *(_DWORD *)(v6 + 16);
  v27 = sub_AF5140(v6, 2u);
  v28 = sub_AF5140(v6, 3u);
  v29 = *(_DWORD *)(v6 + 20);
  v13 = *(_BYTE *)(v6 - 16);
  if ( (v13 & 2) != 0 )
    v14 = *(_QWORD *)(v6 - 32);
  else
    v14 = v23 - 8LL * ((v13 >> 2) & 0xF);
  v15 = v4 - 1;
  v30[0] = *(_QWORD *)(v14 + 32);
  v16 = sub_AF9E80(&v24, &v25, &v26, &v27, &v28, &v29, v30);
  v17 = *a2;
  v18 = 0;
  v19 = v15 & v16;
  v20 = 1;
  v21 = (_QWORD *)(v7 + 8LL * v19);
  v22 = *v21;
  if ( *a2 == *v21 )
  {
LABEL_18:
    *a3 = v21;
    return 1;
  }
  else
  {
    while ( v22 != -4096 )
    {
      if ( v22 != -8192 || v18 )
        v21 = v18;
      v19 = v15 & (v20 + v19);
      v22 = *(_QWORD *)(v7 + 8LL * v19);
      if ( v22 == v17 )
      {
        v21 = (_QWORD *)(v7 + 8LL * v19);
        goto LABEL_18;
      }
      ++v20;
      v18 = v21;
      v21 = (_QWORD *)(v7 + 8LL * v19);
    }
    if ( !v18 )
      v18 = v21;
    *a3 = v18;
    return 0;
  }
}
