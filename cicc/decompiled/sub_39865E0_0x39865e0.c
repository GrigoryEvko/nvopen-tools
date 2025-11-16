// Function: sub_39865E0
// Address: 0x39865e0
//
_QWORD *__fastcall sub_39865E0(_QWORD *a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v4; // rsi
  _QWORD *v5; // r8
  __int64 v6; // rax
  __int64 v8; // r10
  unsigned int v9; // r14d
  __int64 v10; // r9
  int v11; // ecx
  int v12; // ecx
  __int64 v13; // rbx
  unsigned int v14; // r11d
  __int64 *v15; // r9
  __int64 v16; // r15
  unsigned int v17; // r11d
  __int64 v18; // rsi
  __int64 *v19; // rdx
  __int64 v20; // rdi
  __int64 v21; // rcx
  int v22; // ecx
  unsigned int v23; // r9d
  __int64 v24; // r13
  int v25; // r15d
  __int64 *v26; // r15
  unsigned int v27; // ecx
  int v29; // r9d
  int v30; // r13d
  __int64 v31; // [rsp-8h] [rbp-8h]

  v4 = a2 - (_QWORD)a1;
  v5 = a1;
  v6 = v4 >> 4;
  if ( v4 > 0 )
  {
    v8 = *a3;
    v9 = ((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4);
    while ( 1 )
    {
      while ( 1 )
      {
        v18 = v6 >> 1;
        v19 = &v5[2 * (v6 >> 1)];
        v20 = *v19;
        if ( !v8 )
        {
          if ( !v20 )
            goto LABEL_19;
          v21 = *(_QWORD *)(*(_QWORD *)(a4 + 8) + 256LL);
          v13 = *(_QWORD *)(v21 + 88);
          v22 = *(_DWORD *)(v21 + 104);
          if ( !v22 )
            goto LABEL_19;
          v12 = v22 - 1;
LABEL_12:
          v23 = v12 & (((unsigned int)v20 >> 4) ^ ((unsigned int)v20 >> 9));
          v24 = *(_QWORD *)(v13 + 16LL * v23);
          v17 = 0;
          if ( v20 == v24 )
            goto LABEL_19;
          goto LABEL_13;
        }
        v10 = *(_QWORD *)(*(_QWORD *)(a4 + 8) + 256LL);
        v11 = *(_DWORD *)(v10 + 104);
        if ( v11 )
          break;
LABEL_19:
        v5 = v19 + 2;
        v6 = v6 - v18 - 1;
        if ( v6 <= 0 )
          return v5;
      }
      v12 = v11 - 1;
      v13 = *(_QWORD *)(v10 + 88);
      v14 = v12 & v9;
      v15 = (__int64 *)(v13 + 16LL * (v12 & v9));
      v16 = *v15;
      if ( v8 != *v15 )
      {
        v29 = 1;
        while ( v16 != -8 )
        {
          v30 = v29 + 1;
          v14 = v12 & (v29 + v14);
          v15 = (__int64 *)(v13 + 16LL * v14);
          v16 = *v15;
          if ( v8 == *v15 )
            goto LABEL_5;
          v29 = v30;
        }
        if ( !v20 )
          goto LABEL_19;
        goto LABEL_12;
      }
LABEL_5:
      v17 = *((_DWORD *)v15 + 2);
      if ( !v20 )
        goto LABEL_6;
      v23 = v12 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v26 = (__int64 *)(v13 + 16LL * v23);
      v24 = *v26;
      if ( *v26 == v20 )
      {
LABEL_16:
        v27 = *((_DWORD *)v26 + 2);
        if ( v17 && (!v27 || v27 > v17) )
          goto LABEL_7;
        goto LABEL_19;
      }
LABEL_13:
      v25 = 1;
      while ( v24 != -8 )
      {
        v23 = v12 & (v25 + v23);
        *((_DWORD *)&v31 - 11) = v25 + 1;
        v26 = (__int64 *)(v13 + 16LL * v23);
        v24 = *v26;
        if ( *v26 == v20 )
          goto LABEL_16;
        v25 = *((_DWORD *)&v31 - 11);
      }
LABEL_6:
      if ( !v17 )
        goto LABEL_19;
LABEL_7:
      v6 >>= 1;
      if ( v18 <= 0 )
        return v5;
    }
  }
  return a1;
}
