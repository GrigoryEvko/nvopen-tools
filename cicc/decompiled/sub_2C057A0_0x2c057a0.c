// Function: sub_2C057A0
// Address: 0x2c057a0
//
_QWORD *__fastcall sub_2C057A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rbx
  _QWORD *v7; // rax
  __int64 v8; // rdi
  __int64 v9; // r15
  __int64 v10; // rax
  unsigned int v11; // ecx
  __int64 *v12; // rdx
  __int64 v13; // r9
  __int64 v14; // rax
  _QWORD *v15; // rdx
  int v17; // edx
  _QWORD *v18; // rax
  __int64 v19; // rdi
  __int64 v20; // rsi
  __int64 v21; // rax
  unsigned int v22; // ecx
  __int64 *v23; // rdx
  __int64 v24; // r9
  __int64 v25; // rax
  __int64 v26; // rdx
  int v27; // r10d
  int v28; // edx
  int v29; // r10d
  __int64 v30[7]; // [rsp+18h] [rbp-38h] BYREF

  v6 = a1 + 528;
  v30[0] = a2;
  v7 = sub_2C047F0(a1 + 528, v30);
  v8 = *(_QWORD *)(a3 + 96);
  v9 = v7[2];
  v10 = *(unsigned int *)(a3 + 112);
  if ( !(_DWORD)v10 )
    goto LABEL_9;
  v11 = (v10 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
  v12 = (__int64 *)(v8 + 16LL * v11);
  v13 = *v12;
  if ( v9 != *v12 )
  {
    v17 = 1;
    while ( v13 != -4096 )
    {
      v27 = v17 + 1;
      v11 = (v10 - 1) & (v17 + v11);
      v12 = (__int64 *)(v8 + 16LL * v11);
      v13 = *v12;
      if ( v9 == *v12 )
        goto LABEL_3;
      v17 = v27;
    }
LABEL_9:
    v30[0] = v9;
    v18 = sub_2C047F0(v6, v30);
    v19 = *(_QWORD *)(a3 + 96);
    v20 = v18[2];
    v21 = *(unsigned int *)(a3 + 112);
    if ( (_DWORD)v21 )
    {
      v22 = (v21 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v23 = (__int64 *)(v19 + 16LL * v22);
      v24 = *v23;
      if ( v20 == *v23 )
      {
LABEL_11:
        if ( v23 != (__int64 *)(v19 + 16 * v21) )
        {
          v25 = *((unsigned int *)v23 + 2);
          if ( *(_DWORD *)(a3 + 32) > (unsigned int)v25 )
          {
            v26 = *(_QWORD *)(*(_QWORD *)(a3 + 24) + 8 * v25);
            if ( v26 )
              goto LABEL_14;
          }
        }
      }
      else
      {
        v28 = 1;
        while ( v24 != -4096 )
        {
          v29 = v28 + 1;
          v22 = (v21 - 1) & (v28 + v22);
          v23 = (__int64 *)(v19 + 16LL * v22);
          v24 = *v23;
          if ( v20 == *v23 )
            goto LABEL_11;
          v28 = v29;
        }
      }
    }
    v26 = sub_2C057A0(a1, v20, a3);
LABEL_14:
    v15 = sub_2C05360(a3, v9, v26);
    return sub_2C05360(a3, a2, (__int64)v15);
  }
LABEL_3:
  if ( v12 == (__int64 *)(v8 + 16 * v10) )
    goto LABEL_9;
  v14 = *((unsigned int *)v12 + 2);
  if ( *(_DWORD *)(a3 + 32) <= (unsigned int)v14 )
    goto LABEL_9;
  v15 = *(_QWORD **)(*(_QWORD *)(a3 + 24) + 8 * v14);
  if ( !v15 )
    goto LABEL_9;
  return sub_2C05360(a3, a2, (__int64)v15);
}
