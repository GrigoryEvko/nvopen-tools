// Function: sub_1A504C0
// Address: 0x1a504c0
//
_QWORD *__fastcall sub_1A504C0(_QWORD *a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v4; // rsi
  _QWORD *v5; // r8
  __int64 v6; // rax
  __int64 v7; // r12
  int v8; // ebx
  __int64 v9; // r13
  int v10; // r14d
  __int64 v11; // r15
  __int64 *v12; // r9
  __int64 v13; // rsi
  __int64 *v14; // r10
  __int64 v15; // rdi
  unsigned int v16; // ecx
  __int64 *v17; // rdx
  __int64 v18; // r11
  _QWORD **v19; // rdx
  _QWORD *v20; // rcx
  unsigned int i; // edx
  __int64 v22; // rdi
  __int64 *v23; // rcx
  _QWORD **v24; // rcx
  _QWORD *v25; // rcx
  unsigned int j; // edi
  unsigned int v28; // r11d
  int v29; // ecx
  int v30; // edx
  __int64 v31; // [rsp-8h] [rbp-8h]

  v4 = a2 - (_QWORD)a1;
  v5 = a1;
  v6 = v4 >> 3;
  if ( v4 > 0 )
  {
    v7 = *a3;
    v8 = *(_DWORD *)(a4 + 24);
    v9 = *(_QWORD *)(a4 + 8);
    v10 = v8 - 1;
    v11 = (v8 - 1) & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
    v12 = (__int64 *)(v9 + 16 * v11);
    while ( 1 )
    {
      v13 = v6 >> 1;
      if ( !v8 )
        goto LABEL_18;
      v14 = &v5[v13];
      v15 = *v14;
      v16 = v10 & (((unsigned int)*v14 >> 9) ^ ((unsigned int)*v14 >> 4));
      v17 = (__int64 *)(v9 + 16LL * v16);
      v18 = *v17;
      if ( *v17 != *v14 )
        break;
LABEL_5:
      v19 = (_QWORD **)v17[1];
      if ( !v19 )
        goto LABEL_21;
      v20 = *v19;
      for ( i = 1; v20; ++i )
        v20 = (_QWORD *)*v20;
LABEL_8:
      v22 = *v12;
      v23 = (__int64 *)(v9 + 16 * v11);
      if ( v7 != *v12 )
      {
        v28 = v11;
        v29 = 1;
        while ( v22 != -8 )
        {
          v28 = v10 & (v29 + v28);
          *((_DWORD *)&v31 - 11) = v29 + 1;
          v23 = (__int64 *)(v9 + 16LL * v28);
          v22 = *v23;
          if ( v7 == *v23 )
            goto LABEL_9;
          v29 = *((_DWORD *)&v31 - 11);
        }
LABEL_18:
        v6 >>= 1;
        goto LABEL_14;
      }
LABEL_9:
      v24 = (_QWORD **)v23[1];
      if ( !v24 )
        goto LABEL_18;
      v25 = *v24;
      for ( j = 1; v25; ++j )
        v25 = (_QWORD *)*v25;
      if ( j <= i )
        goto LABEL_18;
      v5 = v14 + 1;
      v6 = v6 - v13 - 1;
LABEL_14:
      if ( v6 <= 0 )
        return v5;
    }
    v30 = 1;
    while ( v18 != -8 )
    {
      v16 = v10 & (v30 + v16);
      *((_DWORD *)&v31 - 11) = v30 + 1;
      v17 = (__int64 *)(v9 + 16LL * v16);
      v18 = *v17;
      if ( v15 == *v17 )
        goto LABEL_5;
      v30 = *((_DWORD *)&v31 - 11);
    }
LABEL_21:
    i = 0;
    goto LABEL_8;
  }
  return a1;
}
