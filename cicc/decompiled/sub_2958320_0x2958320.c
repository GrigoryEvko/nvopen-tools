// Function: sub_2958320
// Address: 0x2958320
//
_QWORD *__fastcall sub_2958320(_QWORD *a1, __int64 a2, __int64 *a3, __int64 a4)
{
  _QWORD *v4; // r10
  __int64 v5; // rax
  __int64 v6; // rsi
  __int64 v7; // r13
  int v8; // r12d
  __int64 v9; // r14
  int v10; // r15d
  unsigned int v11; // eax
  __int64 *v12; // r11
  __int64 *v13; // r9
  __int64 v14; // rdi
  __int64 v15; // rax
  __int64 *v16; // rdx
  _QWORD **v17; // rax
  _QWORD *v18; // rdx
  unsigned int v19; // ecx
  __int64 *v20; // rdx
  __int64 v21; // rbx
  _QWORD **v22; // rdx
  _QWORD *v23; // rdx
  unsigned int i; // ecx
  int v26; // edx
  unsigned int v27; // ecx
  int v28; // edx
  int v29; // ebx
  __int64 v30; // [rsp-8h] [rbp-8h]

  v4 = a1;
  v5 = a2 - (_QWORD)a1;
  v6 = (a2 - (__int64)a1) >> 3;
  if ( v5 > 0 )
  {
    v7 = *a3;
    v8 = *(_DWORD *)(a4 + 24);
    v9 = *(_QWORD *)(a4 + 8);
    v10 = v8 - 1;
    v11 = (v8 - 1) & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
    *((_DWORD *)&v30 - 11) = v11;
    v12 = (__int64 *)(v9 + 16LL * v11);
    while ( 1 )
    {
      v13 = &v4[v6 >> 1];
      v14 = *v13;
      if ( !v8 )
        goto LABEL_18;
      v15 = *v12;
      v16 = v12;
      if ( v7 == *v12 )
      {
LABEL_5:
        v17 = (_QWORD **)v16[1];
        if ( v17 )
        {
          v18 = *v17;
          for ( LODWORD(v17) = 1; v18; LODWORD(v17) = (_DWORD)v17 + 1 )
            v18 = (_QWORD *)*v18;
        }
      }
      else
      {
        v27 = *((_DWORD *)&v30 - 11);
        v28 = 1;
        while ( v15 != -4096 )
        {
          v29 = v28 + 1;
          v27 = v10 & (v28 + v27);
          v16 = (__int64 *)(v9 + 16LL * v27);
          v15 = *v16;
          if ( v7 == *v16 )
            goto LABEL_5;
          v28 = v29;
        }
        LODWORD(v17) = 0;
      }
      v19 = v10 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v20 = (__int64 *)(v9 + 16LL * v19);
      v21 = *v20;
      if ( v14 != *v20 )
        break;
LABEL_9:
      v22 = (_QWORD **)v20[1];
      if ( !v22 )
        goto LABEL_18;
      v23 = *v22;
      for ( i = 1; v23; ++i )
        v23 = (_QWORD *)*v23;
      if ( (unsigned int)v17 >= i )
        goto LABEL_18;
      v6 >>= 1;
LABEL_14:
      if ( v6 <= 0 )
        return v4;
    }
    v26 = 1;
    while ( v21 != -4096 )
    {
      v19 = v10 & (v26 + v19);
      *((_DWORD *)&v30 - 12) = v26 + 1;
      v20 = (__int64 *)(v9 + 16LL * v19);
      v21 = *v20;
      if ( v14 == *v20 )
        goto LABEL_9;
      v26 = *((_DWORD *)&v30 - 12);
    }
LABEL_18:
    v4 = v13 + 1;
    v6 = v6 - (v6 >> 1) - 1;
    goto LABEL_14;
  }
  return a1;
}
