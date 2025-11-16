// Function: sub_15F1BE0
// Address: 0x15f1be0
//
_BOOL8 __fastcall sub_15F1BE0(__int64 a1, _BYTE *a2, __int64 a3)
{
  int v5; // esi
  char *v6; // r13
  __int64 v7; // rdx
  unsigned __int64 v8; // r12
  unsigned __int64 v9; // rbx
  __int64 v10; // rax
  unsigned __int64 v11; // r15
  _QWORD *v12; // r14
  unsigned __int64 v13; // r12
  _QWORD *v14; // rbx
  int v15; // r8d
  int v16; // r9d
  char *v17; // rax
  int v18; // ecx
  int v19; // edx
  __int64 v20; // rax
  bool v21; // [rsp+7h] [rbp-349h]
  char *v22; // [rsp+8h] [rbp-348h]
  char *v23; // [rsp+10h] [rbp-340h] BYREF
  int v24; // [rsp+18h] [rbp-338h]
  char v25; // [rsp+20h] [rbp-330h] BYREF

  v21 = 0;
  if ( *(_DWORD *)(a1 + 8) >> 8 )
    return v21;
  sub_15F1410(&v23, a2, a3);
  if ( !v24 )
  {
    if ( a3 )
      goto LABEL_29;
    v22 = v23;
    if ( !*(_BYTE *)(**(_QWORD **)(a1 + 16) + 8LL) )
    {
      v5 = 0;
      goto LABEL_7;
    }
    goto LABEL_30;
  }
  v15 = 0;
  v5 = 0;
  v16 = 0;
  v22 = v23;
  v17 = v23;
  v18 = 0;
  do
  {
    v19 = *(_DWORD *)v17;
    if ( *(_DWORD *)v17 == 1 )
    {
      if ( v15 != v5 || v18 )
        goto LABEL_46;
      if ( !v17[10] )
      {
        ++v16;
        goto LABEL_34;
      }
      ++v15;
    }
    else
    {
      if ( v19 == 2 )
      {
        ++v18;
        goto LABEL_34;
      }
      if ( v19 )
        goto LABEL_34;
      if ( v18 )
        goto LABEL_46;
    }
    ++v5;
    v18 = 0;
LABEL_34:
    v17 += 192;
  }
  while ( v17 != &v23[192 * (v24 - 1) + 192] );
  v20 = **(_QWORD **)(a1 + 16);
  if ( !v16 )
  {
    if ( !*(_BYTE *)(v20 + 8) )
      goto LABEL_7;
LABEL_46:
    v21 = 0;
    goto LABEL_8;
  }
  if ( v16 == 1 )
  {
    v21 = 0;
    if ( *(_BYTE *)(v20 + 8) != 13 )
      goto LABEL_7;
  }
  else
  {
    v21 = 0;
    if ( *(_BYTE *)(v20 + 8) == 13 && v16 == *(_DWORD *)(v20 + 12) )
LABEL_7:
      v21 = *(_DWORD *)(a1 + 12) - 1 == v5;
  }
LABEL_8:
  v6 = &v22[192 * v24];
  if ( v6 != v22 )
  {
    do
    {
      v7 = *((unsigned int *)v6 - 30);
      v8 = *((_QWORD *)v6 - 16);
      v6 -= 192;
      v9 = v8 + 56 * v7;
      if ( v8 != v9 )
      {
        do
        {
          v10 = *(unsigned int *)(v9 - 40);
          v11 = *(_QWORD *)(v9 - 48);
          v9 -= 56LL;
          v10 *= 32;
          v12 = (_QWORD *)(v11 + v10);
          if ( v11 != v11 + v10 )
          {
            do
            {
              v12 -= 4;
              if ( (_QWORD *)*v12 != v12 + 2 )
                j_j___libc_free_0(*v12, v12[2] + 1LL);
            }
            while ( (_QWORD *)v11 != v12 );
            v11 = *(_QWORD *)(v9 + 8);
          }
          if ( v11 != v9 + 24 )
            _libc_free(v11);
        }
        while ( v8 != v9 );
        v8 = *((_QWORD *)v6 + 8);
      }
      if ( (char *)v8 != v6 + 80 )
        _libc_free(v8);
      v13 = *((_QWORD *)v6 + 2);
      v14 = (_QWORD *)(v13 + 32LL * *((unsigned int *)v6 + 6));
      if ( (_QWORD *)v13 != v14 )
      {
        do
        {
          v14 -= 4;
          if ( (_QWORD *)*v14 != v14 + 2 )
            j_j___libc_free_0(*v14, v14[2] + 1LL);
        }
        while ( (_QWORD *)v13 != v14 );
        v13 = *((_QWORD *)v6 + 2);
      }
      if ( (char *)v13 != v6 + 32 )
        _libc_free(v13);
    }
    while ( v6 != v22 );
LABEL_29:
    v22 = v23;
  }
LABEL_30:
  if ( v22 != &v25 )
    _libc_free((unsigned __int64)v22);
  return v21;
}
