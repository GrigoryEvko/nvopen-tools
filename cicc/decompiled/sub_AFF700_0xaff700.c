// Function: sub_AFF700
// Address: 0xaff700
//
__int64 __fastcall sub_AFF700(__int64 a1, __int64 *a2, _QWORD *a3)
{
  int v4; // r14d
  __int64 v6; // rbx
  __int64 v7; // r15
  unsigned __int16 v9; // ax
  __int64 v10; // rdx
  unsigned __int8 v11; // al
  unsigned __int8 v12; // al
  unsigned __int8 v13; // al
  __int64 v14; // rcx
  unsigned __int8 v15; // al
  __int64 v16; // rdx
  int v17; // r14d
  int v18; // eax
  __int64 v19; // rsi
  _QWORD *v20; // rdi
  unsigned int v21; // eax
  int v22; // r8d
  _QWORD *v23; // rcx
  __int64 v24; // rdx
  int v25; // [rsp+10h] [rbp-70h] BYREF
  __int64 v26; // [rsp+18h] [rbp-68h] BYREF
  __int64 v27; // [rsp+20h] [rbp-60h] BYREF
  __int64 v28; // [rsp+28h] [rbp-58h] BYREF
  int v29; // [rsp+30h] [rbp-50h] BYREF
  __int64 v30; // [rsp+38h] [rbp-48h] BYREF
  __int64 v31[8]; // [rsp+40h] [rbp-40h] BYREF

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
  v25 = v9;
  v11 = *(_BYTE *)(v6 - 16);
  if ( (v11 & 2) != 0 )
  {
    v26 = **(_QWORD **)(v6 - 32);
    v12 = *(_BYTE *)(v6 - 16);
    if ( (v12 & 2) != 0 )
      goto LABEL_5;
LABEL_16:
    v27 = *(_QWORD *)(v10 - 8LL * ((v12 >> 2) & 0xF) + 8);
    v13 = *(_BYTE *)(v6 - 16);
    if ( (v13 & 2) != 0 )
      goto LABEL_6;
    goto LABEL_17;
  }
  v26 = *(_QWORD *)(v10 - 8LL * ((v11 >> 2) & 0xF));
  v12 = *(_BYTE *)(v6 - 16);
  if ( (v12 & 2) == 0 )
    goto LABEL_16;
LABEL_5:
  v27 = *(_QWORD *)(*(_QWORD *)(v6 - 32) + 8LL);
  v13 = *(_BYTE *)(v6 - 16);
  if ( (v13 & 2) != 0 )
  {
LABEL_6:
    v14 = *(_QWORD *)(v6 - 32);
    goto LABEL_7;
  }
LABEL_17:
  v14 = v10 - 8LL * ((v13 >> 2) & 0xF);
LABEL_7:
  v28 = *(_QWORD *)(v14 + 24);
  v29 = *(_DWORD *)(v6 + 4);
  v30 = sub_AF5140(v6, 2u);
  v15 = *(_BYTE *)(v6 - 16);
  if ( (v15 & 2) != 0 )
    v16 = *(_QWORD *)(v6 - 32);
  else
    v16 = v6 - 16 - 8LL * ((v15 >> 2) & 0xF);
  v17 = v4 - 1;
  v31[0] = *(_QWORD *)(v16 + 32);
  v18 = sub_AFB320(&v25, &v26, &v27, &v28, &v29, &v30, v31);
  v19 = *a2;
  v20 = 0;
  v21 = v17 & v18;
  v22 = 1;
  v23 = (_QWORD *)(v7 + 8LL * v21);
  v24 = *v23;
  if ( *a2 == *v23 )
  {
LABEL_20:
    *a3 = v23;
    return 1;
  }
  else
  {
    while ( v24 != -4096 )
    {
      if ( v24 != -8192 || v20 )
        v23 = v20;
      v21 = v17 & (v22 + v21);
      v24 = *(_QWORD *)(v7 + 8LL * v21);
      if ( v24 == v19 )
      {
        v23 = (_QWORD *)(v7 + 8LL * v21);
        goto LABEL_20;
      }
      ++v22;
      v20 = v23;
      v23 = (_QWORD *)(v7 + 8LL * v21);
    }
    if ( !v20 )
      v20 = v23;
    *a3 = v20;
    return 0;
  }
}
