// Function: sub_DAEE00
// Address: 0xdaee00
//
_QWORD *__fastcall sub_DAEE00(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  _QWORD *result; // rax
  __int64 *v5; // r15
  __int64 *v8; // rbx
  __int64 v9; // r12
  unsigned int v10; // esi
  int v11; // r11d
  __int64 v12; // r9
  __int64 v13; // rdx
  __int64 v14; // r8
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v17; // rdi
  __int64 v18; // rdi
  int v19; // edi
  int v20; // edi
  int v21; // eax
  __int64 v22; // rsi
  unsigned int v23; // edx
  __int64 v24; // rcx
  int v25; // r10d
  int v26; // eax
  int v27; // ecx
  int v28; // r10d
  unsigned int v29; // edx
  __int64 v30; // rsi
  __int64 v31; // [rsp+8h] [rbp-38h]

  result = (_QWORD *)(a1 + 936);
  v5 = &a3[a4];
  v31 = a1 + 936;
  if ( a3 != v5 )
  {
    v8 = a3;
    while ( 1 )
    {
      v9 = *v8;
      if ( !*(_WORD *)(*v8 + 24) )
        goto LABEL_3;
      v10 = *(_DWORD *)(a1 + 960);
      if ( !v10 )
        break;
      v11 = 1;
      v12 = *(_QWORD *)(a1 + 944);
      v13 = ((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4);
      v14 = (v10 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v15 = v12 + 104 * v14;
      v16 = 0;
      v17 = *(_QWORD *)v15;
      if ( v9 == *(_QWORD *)v15 )
      {
LABEL_7:
        v18 = v15 + 8;
        if ( *(_BYTE *)(v15 + 36) )
          goto LABEL_23;
LABEL_8:
        ++v8;
        result = sub_C8CC70(v18, a2, v13, v15, v14, v12);
        if ( v5 == v8 )
          return result;
      }
      else
      {
        while ( v17 != -4096 )
        {
          if ( v17 == -8192 && !v16 )
            v16 = v15;
          v14 = (v10 - 1) & (v11 + (_DWORD)v14);
          v15 = v12 + 104LL * (unsigned int)v14;
          v17 = *(_QWORD *)v15;
          if ( v9 == *(_QWORD *)v15 )
            goto LABEL_7;
          ++v11;
        }
        v19 = *(_DWORD *)(a1 + 952);
        if ( !v16 )
          v16 = v15;
        ++*(_QWORD *)(a1 + 936);
        v20 = v19 + 1;
        if ( 4 * v20 < 3 * v10 )
        {
          v14 = v10 >> 3;
          if ( v10 - *(_DWORD *)(a1 + 956) - v20 > (unsigned int)v14 )
            goto LABEL_20;
          sub_DAEBC0(v31, v10);
          v26 = *(_DWORD *)(a1 + 960);
          if ( !v26 )
          {
LABEL_49:
            ++*(_DWORD *)(a1 + 952);
            BUG();
          }
          v27 = v26 - 1;
          v14 = *(_QWORD *)(a1 + 944);
          v12 = 0;
          v28 = 1;
          v29 = (v26 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
          v20 = *(_DWORD *)(a1 + 952) + 1;
          v16 = v14 + 104LL * v29;
          v30 = *(_QWORD *)v16;
          if ( v9 == *(_QWORD *)v16 )
            goto LABEL_20;
          while ( v30 != -4096 )
          {
            if ( v30 == -8192 && !v12 )
              v12 = v16;
            v29 = v27 & (v28 + v29);
            v16 = v14 + 104LL * v29;
            v30 = *(_QWORD *)v16;
            if ( v9 == *(_QWORD *)v16 )
              goto LABEL_20;
            ++v28;
          }
          goto LABEL_41;
        }
LABEL_29:
        sub_DAEBC0(v31, 2 * v10);
        v21 = *(_DWORD *)(a1 + 960);
        if ( !v21 )
          goto LABEL_49;
        v14 = (unsigned int)(v21 - 1);
        v22 = *(_QWORD *)(a1 + 944);
        v23 = v14 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v20 = *(_DWORD *)(a1 + 952) + 1;
        v16 = v22 + 104LL * v23;
        v24 = *(_QWORD *)v16;
        if ( v9 == *(_QWORD *)v16 )
          goto LABEL_20;
        v25 = 1;
        v12 = 0;
        while ( v24 != -4096 )
        {
          if ( !v12 && v24 == -8192 )
            v12 = v16;
          v23 = v14 & (v25 + v23);
          v16 = v22 + 104LL * v23;
          v24 = *(_QWORD *)v16;
          if ( v9 == *(_QWORD *)v16 )
            goto LABEL_20;
          ++v25;
        }
LABEL_41:
        if ( v12 )
          v16 = v12;
LABEL_20:
        *(_DWORD *)(a1 + 952) = v20;
        if ( *(_QWORD *)v16 != -4096 )
          --*(_DWORD *)(a1 + 956);
        *(_QWORD *)v16 = v9;
        v18 = v16 + 8;
        *(_QWORD *)(v16 + 8) = 0;
        *(_QWORD *)(v16 + 16) = v16 + 40;
        *(_QWORD *)(v16 + 24) = 8;
        *(_DWORD *)(v16 + 32) = 0;
        *(_BYTE *)(v16 + 36) = 1;
LABEL_23:
        result = *(_QWORD **)(v18 + 8);
        v15 = *(unsigned int *)(v18 + 20);
        v13 = (__int64)&result[v15];
        if ( result == (_QWORD *)v13 )
        {
LABEL_26:
          if ( (unsigned int)v15 >= *(_DWORD *)(v18 + 16) )
            goto LABEL_8;
          *(_DWORD *)(v18 + 20) = v15 + 1;
          *(_QWORD *)v13 = a2;
          ++*(_QWORD *)v18;
        }
        else
        {
          while ( a2 != *result )
          {
            if ( (_QWORD *)v13 == ++result )
              goto LABEL_26;
          }
        }
LABEL_3:
        if ( v5 == ++v8 )
          return result;
      }
    }
    ++*(_QWORD *)(a1 + 936);
    goto LABEL_29;
  }
  return result;
}
