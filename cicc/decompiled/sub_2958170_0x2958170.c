// Function: sub_2958170
// Address: 0x2958170
//
_QWORD *__fastcall sub_2958170(_QWORD *a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v4; // rsi
  _QWORD *v6; // r9
  __int64 v7; // rcx
  __int64 v8; // r12
  __int64 v9; // r13
  int v10; // ebx
  int v11; // r14d
  __int64 v12; // r15
  __int64 *v13; // r10
  __int64 *v14; // r8
  __int64 v15; // rsi
  unsigned int v16; // edx
  __int64 *v17; // rax
  __int64 v18; // r11
  _QWORD **v19; // rax
  _QWORD *v20; // rdx
  __int64 v21; // rdx
  __int64 *v22; // rsi
  _QWORD **v23; // rdx
  _QWORD *v24; // rdx
  unsigned int i; // esi
  unsigned int v27; // r11d
  int v28; // esi
  int v29; // eax
  __int64 v30; // [rsp-8h] [rbp-8h]

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
    v13 = (__int64 *)(v9 + 16 * v12);
    while ( 1 )
    {
      v14 = &v6[v7 >> 1];
      v15 = *v14;
      if ( !v10 )
        goto LABEL_18;
      v16 = v11 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v17 = (__int64 *)(v9 + 16LL * v16);
      v18 = *v17;
      if ( v15 == *v17 )
      {
LABEL_5:
        v19 = (_QWORD **)v17[1];
        if ( v19 )
        {
          v20 = *v19;
          for ( LODWORD(v19) = 1; v20; LODWORD(v19) = (_DWORD)v19 + 1 )
            v20 = (_QWORD *)*v20;
        }
      }
      else
      {
        v29 = 1;
        while ( v18 != -4096 )
        {
          v16 = v11 & (v29 + v16);
          *((_DWORD *)&v30 - 11) = v29 + 1;
          v17 = (__int64 *)(v9 + 16LL * v16);
          v18 = *v17;
          if ( v15 == *v17 )
            goto LABEL_5;
          v29 = *((_DWORD *)&v30 - 11);
        }
        LODWORD(v19) = 0;
      }
      v21 = *v13;
      v22 = (__int64 *)(v9 + 16 * v12);
      if ( v8 != *v13 )
        break;
LABEL_9:
      v23 = (_QWORD **)v22[1];
      if ( !v23 )
        goto LABEL_18;
      v24 = *v23;
      for ( i = 1; v24; ++i )
        v24 = (_QWORD *)*v24;
      if ( (unsigned int)v19 >= i )
        goto LABEL_18;
      v6 = v14 + 1;
      v7 = v7 - (v7 >> 1) - 1;
LABEL_14:
      if ( v7 <= 0 )
        return v6;
    }
    v27 = v12;
    v28 = 1;
    while ( v21 != -4096 )
    {
      v27 = v11 & (v28 + v27);
      *((_DWORD *)&v30 - 11) = v28 + 1;
      v22 = (__int64 *)(v9 + 16LL * v27);
      v21 = *v22;
      if ( v8 == *v22 )
        goto LABEL_9;
      v28 = *((_DWORD *)&v30 - 11);
    }
LABEL_18:
    v7 >>= 1;
    goto LABEL_14;
  }
  return a1;
}
