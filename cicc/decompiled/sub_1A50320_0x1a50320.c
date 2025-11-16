// Function: sub_1A50320
// Address: 0x1a50320
//
_QWORD *__fastcall sub_1A50320(_QWORD *a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v4; // rsi
  _QWORD *v6; // r9
  __int64 v7; // rcx
  __int64 v8; // r13
  __int64 v9; // r14
  int v10; // r12d
  int v11; // r15d
  unsigned int v12; // eax
  __int64 *v13; // r11
  __int64 *v14; // r10
  __int64 v15; // r8
  __int64 *v16; // rax
  _QWORD **v17; // rax
  _QWORD *v18; // rdx
  unsigned int i; // eax
  __int64 v20; // r8
  unsigned int v21; // esi
  __int64 *v22; // rdx
  __int64 v23; // rbx
  _QWORD **v24; // rdx
  _QWORD *v25; // rdx
  unsigned int j; // esi
  int v28; // edx
  unsigned int v29; // edx
  int v30; // eax
  int v31; // esi
  __int64 v32; // [rsp-8h] [rbp-8h]

  v4 = a2 - (_QWORD)a1;
  v6 = a1;
  v7 = v4 >> 3;
  if ( v4 > 0 )
  {
    v8 = *a3;
    v9 = *(_QWORD *)(a4 + 8);
    v10 = *(_DWORD *)(a4 + 24);
    v11 = v10 - 1;
    v12 = (v10 - 1) & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
    *((_DWORD *)&v32 - 11) = v12;
    v13 = (__int64 *)(v9 + 16LL * v12);
    while ( 1 )
    {
      v14 = &v6[v7 >> 1];
      if ( !v10 )
        goto LABEL_18;
      v15 = *v13;
      v16 = v13;
      if ( *v13 != v8 )
        break;
LABEL_5:
      v17 = (_QWORD **)v16[1];
      if ( !v17 )
        goto LABEL_21;
      v18 = *v17;
      for ( i = 1; v18; ++i )
        v18 = (_QWORD *)*v18;
LABEL_8:
      v20 = *v14;
      v21 = v11 & (((unsigned int)*v14 >> 9) ^ ((unsigned int)*v14 >> 4));
      v22 = (__int64 *)(v9 + 16LL * v21);
      v23 = *v22;
      if ( *v14 != *v22 )
      {
        v28 = 1;
        while ( v23 != -8 )
        {
          v21 = v11 & (v28 + v21);
          *((_DWORD *)&v32 - 12) = v28 + 1;
          v22 = (__int64 *)(v9 + 16LL * v21);
          v23 = *v22;
          if ( v20 == *v22 )
            goto LABEL_9;
          v28 = *((_DWORD *)&v32 - 12);
        }
LABEL_18:
        v6 = v14 + 1;
        v7 = v7 - (v7 >> 1) - 1;
        goto LABEL_14;
      }
LABEL_9:
      v24 = (_QWORD **)v22[1];
      if ( !v24 )
        goto LABEL_18;
      v25 = *v24;
      for ( j = 1; v25; ++j )
        v25 = (_QWORD *)*v25;
      if ( j <= i )
        goto LABEL_18;
      v7 >>= 1;
LABEL_14:
      if ( v7 <= 0 )
        return v6;
    }
    v29 = *((_DWORD *)&v32 - 11);
    v30 = 1;
    while ( v15 != -8 )
    {
      v31 = v30 + 1;
      v29 = v11 & (v30 + v29);
      v16 = (__int64 *)(v9 + 16LL * v29);
      v15 = *v16;
      if ( v8 == *v16 )
        goto LABEL_5;
      v30 = v31;
    }
LABEL_21:
    i = 0;
    goto LABEL_8;
  }
  return a1;
}
