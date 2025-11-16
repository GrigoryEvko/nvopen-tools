// Function: sub_C18300
// Address: 0xc18300
//
__int64 __fastcall sub_C18300(__int64 a1, __int64 a2)
{
  __int64 v3; // r14
  __int64 v4; // r15
  __int64 v5; // rbx
  __int64 v6; // r9
  int v7; // r11d
  unsigned int v8; // edi
  int *v9; // rcx
  _DWORD *v10; // rax
  int v11; // r8d
  __int64 v12; // rdi
  unsigned int v13; // esi
  int *v14; // r13
  int v15; // ecx
  int v16; // ecx
  __int64 v17; // r9
  int v18; // edx
  unsigned int v19; // edi
  int v20; // r8d
  int v21; // edx
  _QWORD *v22; // rax
  __int64 v23; // rcx
  int v25; // edx
  int v26; // ecx
  int v27; // ecx
  __int64 v28; // r9
  _DWORD *v29; // r10
  int v30; // r11d
  unsigned int v31; // edi
  int v32; // r8d
  int v33; // r11d
  __int64 v34; // [rsp+8h] [rbp-38h]

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  v3 = *(_QWORD *)(a2 + 32);
  v34 = v3 + 72LL * *(unsigned int *)(a2 + 40);
  if ( v34 != v3 )
  {
    while ( 1 )
    {
      v4 = *(unsigned int *)(v3 + 16);
      v5 = 0;
      if ( (_DWORD)v4 )
        break;
LABEL_13:
      v3 += 72;
      if ( v3 == v34 )
        return a1;
    }
    while ( 1 )
    {
      v13 = *(_DWORD *)(a1 + 24);
      v14 = (int *)(*(_QWORD *)(v3 + 8) + 4 * v5);
      if ( !v13 )
        break;
      v6 = *(_QWORD *)(a1 + 8);
      v7 = 1;
      v8 = (v13 - 1) & (37 * *v14);
      v9 = (int *)(v6 + 24LL * v8);
      v10 = 0;
      v11 = *v9;
      if ( *v9 == *v14 )
      {
LABEL_5:
        v12 = v5 + *((_QWORD *)v9 + 2);
        ++v5;
        ++*((_QWORD *)v9 + 1);
        *((_QWORD *)v9 + 2) = v12;
        if ( v4 == v5 )
          goto LABEL_13;
      }
      else
      {
        while ( v11 != -1 )
        {
          if ( !v10 && v11 == -2 )
            v10 = v9;
          v8 = (v13 - 1) & (v7 + v8);
          v9 = (int *)(v6 + 24LL * v8);
          v11 = *v9;
          if ( *v14 == *v9 )
            goto LABEL_5;
          ++v7;
        }
        v25 = *(_DWORD *)(a1 + 16);
        if ( !v10 )
          v10 = v9;
        ++*(_QWORD *)a1;
        v18 = v25 + 1;
        if ( 4 * v18 < 3 * v13 )
        {
          if ( v13 - *(_DWORD *)(a1 + 20) - v18 > v13 >> 3 )
            goto LABEL_10;
          sub_C18120(a1, v13);
          v26 = *(_DWORD *)(a1 + 24);
          if ( !v26 )
          {
LABEL_43:
            ++*(_DWORD *)(a1 + 16);
            BUG();
          }
          v27 = v26 - 1;
          v28 = *(_QWORD *)(a1 + 8);
          v29 = 0;
          v30 = 1;
          v18 = *(_DWORD *)(a1 + 16) + 1;
          v31 = v27 & (37 * *v14);
          v10 = (_DWORD *)(v28 + 24LL * v31);
          v32 = *v10;
          if ( *v14 == *v10 )
            goto LABEL_10;
          while ( v32 != -1 )
          {
            if ( v32 == -2 && !v29 )
              v29 = v10;
            v31 = v27 & (v30 + v31);
            v10 = (_DWORD *)(v28 + 24LL * v31);
            v32 = *v10;
            if ( *v14 == *v10 )
              goto LABEL_10;
            ++v30;
          }
          goto LABEL_28;
        }
LABEL_8:
        sub_C18120(a1, 2 * v13);
        v15 = *(_DWORD *)(a1 + 24);
        if ( !v15 )
          goto LABEL_43;
        v16 = v15 - 1;
        v17 = *(_QWORD *)(a1 + 8);
        v18 = *(_DWORD *)(a1 + 16) + 1;
        v19 = v16 & (37 * *v14);
        v10 = (_DWORD *)(v17 + 24LL * v19);
        v20 = *v10;
        if ( *v14 == *v10 )
          goto LABEL_10;
        v33 = 1;
        v29 = 0;
        while ( v20 != -1 )
        {
          if ( !v29 && v20 == -2 )
            v29 = v10;
          v19 = v16 & (v33 + v19);
          v10 = (_DWORD *)(v17 + 24LL * v19);
          v20 = *v10;
          if ( *v14 == *v10 )
            goto LABEL_10;
          ++v33;
        }
LABEL_28:
        if ( v29 )
          v10 = v29;
LABEL_10:
        *(_DWORD *)(a1 + 16) = v18;
        if ( *v10 != -1 )
          --*(_DWORD *)(a1 + 20);
        v21 = *v14;
        v22 = v10 + 2;
        v23 = v5;
        *v22 = 0;
        v22[1] = 0;
        ++v5;
        *((_DWORD *)v22 - 2) = v21;
        *v22 = 1;
        v22[1] = v23;
        if ( v4 == v5 )
          goto LABEL_13;
      }
    }
    ++*(_QWORD *)a1;
    goto LABEL_8;
  }
  return a1;
}
