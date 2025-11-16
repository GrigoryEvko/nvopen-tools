// Function: sub_2508250
// Address: 0x2508250
//
__int64 __fastcall sub_2508250(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned int v6; // r13d
  __int64 v8; // rdi
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // r9
  __int64 v12; // r8
  _QWORD *v13; // rdi
  _QWORD *v14; // rsi
  __int64 v16; // rax
  __int64 v17; // rcx
  int v18; // eax
  int v19; // edx
  unsigned int v20; // eax
  __int64 v21; // rsi
  __int64 v22; // rdi
  _QWORD *v23; // rax
  _QWORD *v24; // rdx
  int v25; // eax
  __int64 v26; // rcx
  int v27; // edx
  unsigned int v28; // eax
  __int64 v29; // rsi
  int v30; // edi
  int v31; // edi
  __int64 v32; // [rsp+8h] [rbp-48h] BYREF
  __int64 v33; // [rsp+10h] [rbp-40h]
  unsigned __int64 v34[2]; // [rsp+18h] [rbp-38h] BYREF
  _BYTE v35[40]; // [rsp+28h] [rbp-28h] BYREF

  v8 = *(_QWORD *)a2;
  v9 = *(unsigned int *)(a2 + 16);
  v34[0] = (unsigned __int64)v35;
  v34[1] = 0;
  v33 = v8;
  if ( (_DWORD)v9 )
  {
    sub_2506900((__int64)v34, (char **)(a2 + 8), v9, a4, a5, a6);
    v8 = v33;
  }
  v10 = sub_B43CB0(v8);
  v11 = *a1;
  v32 = v10;
  v12 = v10;
  if ( *(_DWORD *)(v11 + 3672) )
  {
    v25 = *(_DWORD *)(v11 + 3680);
    v26 = *(_QWORD *)(v11 + 3664);
    if ( v25 )
    {
      v27 = v25 - 1;
      v28 = (v25 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v29 = *(_QWORD *)(v26 + 8LL * v28);
      if ( v12 == v29 )
        goto LABEL_5;
      v30 = 1;
      while ( v29 != -4096 )
      {
        v28 = v27 & (v30 + v28);
        v29 = *(_QWORD *)(v26 + 8LL * v28);
        if ( v12 == v29 )
          goto LABEL_5;
        ++v30;
      }
    }
  }
  else
  {
    v13 = *(_QWORD **)(v11 + 3688);
    v14 = &v13[*(unsigned int *)(v11 + 3696)];
    if ( v14 != sub_2506780(v13, (__int64)v14, &v32) )
    {
LABEL_5:
      v6 = 1;
      goto LABEL_6;
    }
  }
  v16 = *(_QWORD *)(v11 + 200);
  v17 = *(_QWORD *)(v16 + 8);
  v18 = *(_DWORD *)(v16 + 24);
  if ( !v18 )
    goto LABEL_17;
  v19 = v18 - 1;
  v20 = (v18 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
  v21 = *(_QWORD *)(v17 + 8LL * v20);
  if ( v12 != v21 )
  {
    v31 = 1;
    while ( v21 != -4096 )
    {
      v20 = v19 & (v31 + v20);
      v21 = *(_QWORD *)(v17 + 8LL * v20);
      if ( v12 == v21 )
        goto LABEL_11;
      ++v31;
    }
    goto LABEL_17;
  }
LABEL_11:
  if ( (*(_BYTE *)(v12 + 32) & 0xFu) - 7 > 1 )
  {
LABEL_17:
    v6 = 0;
    goto LABEL_6;
  }
  v22 = a1[1];
  if ( *(_BYTE *)(v22 + 28) )
  {
    v23 = *(_QWORD **)(v22 + 8);
    v24 = &v23[*(unsigned int *)(v22 + 20)];
    if ( v23 == v24 )
      goto LABEL_5;
    while ( v12 != *v23 )
    {
      if ( v24 == ++v23 )
        goto LABEL_5;
    }
    goto LABEL_17;
  }
  LOBYTE(v6) = sub_C8CA60(v22, v12) == 0;
LABEL_6:
  if ( (_BYTE *)v34[0] != v35 )
    _libc_free(v34[0]);
  return v6;
}
