// Function: sub_25E1800
// Address: 0x25e1800
//
__int64 __fastcall sub_25E1800(__int64 a1, __int64 a2)
{
  __int64 *v4; // rax
  __int64 *v5; // r12
  __int64 *v6; // rbx
  __int64 v7; // rax
  unsigned __int64 v8; // r14
  unsigned int v9; // esi
  __int64 v10; // rdi
  _QWORD *v11; // r9
  int v12; // r15d
  unsigned int v13; // edx
  _QWORD *v14; // rax
  __int64 v15; // r11
  unsigned __int64 *v16; // rax
  int v17; // eax
  int v18; // edx
  __int64 v19; // rax
  __int64 v20; // rax
  int v21; // eax
  int v22; // eax
  __int64 v23; // r8
  unsigned int v24; // ecx
  __int64 v25; // rsi
  int v26; // r11d
  _QWORD *v27; // r10
  int v28; // eax
  int v29; // esi
  __int64 v30; // r8
  int v31; // r11d
  unsigned int v32; // ecx
  __int64 v33; // rax

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  if ( *(_DWORD *)(a2 + 144) )
  {
    v4 = *(__int64 **)(a2 + 136);
    v5 = &v4[2 * *(unsigned int *)(a2 + 152)];
    if ( v4 != v5 )
    {
      while ( 1 )
      {
        v6 = v4;
        if ( *v4 != -4096 && *v4 != -8192 )
          break;
        v4 += 2;
        if ( v5 == v4 )
          return a1;
      }
      while ( 1 )
      {
        if ( v5 == v6 )
          return a1;
        v7 = v6[1];
        v8 = v7 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v7 && (v7 & 4) == 0 && (v6[1] & 0xFFFFFFFFFFFFFFF8LL) != 0 )
        {
          v9 = *(_DWORD *)(a1 + 24);
          if ( !v9 )
            goto LABEL_34;
        }
        else
        {
          v20 = sub_29CF9B0();
          v9 = *(_DWORD *)(a1 + 24);
          v8 = v20;
          if ( !v9 )
          {
LABEL_34:
            ++*(_QWORD *)a1;
            goto LABEL_35;
          }
        }
        v10 = *(_QWORD *)(a1 + 8);
        v11 = 0;
        v12 = 1;
        v13 = (v9 - 1) & (((unsigned int)*v6 >> 9) ^ ((unsigned int)*v6 >> 4));
        v14 = (_QWORD *)(v10 + 16LL * v13);
        v15 = *v14;
        if ( *v6 == *v14 )
        {
LABEL_14:
          v16 = v14 + 1;
          goto LABEL_15;
        }
        while ( v15 != -4096 )
        {
          if ( !v11 && v15 == -8192 )
            v11 = v14;
          v13 = (v9 - 1) & (v12 + v13);
          v14 = (_QWORD *)(v10 + 16LL * v13);
          v15 = *v14;
          if ( *v6 == *v14 )
            goto LABEL_14;
          ++v12;
        }
        if ( !v11 )
          v11 = v14;
        v17 = *(_DWORD *)(a1 + 16);
        ++*(_QWORD *)a1;
        v18 = v17 + 1;
        if ( 4 * (v17 + 1) < 3 * v9 )
        {
          if ( v9 - *(_DWORD *)(a1 + 20) - v18 > v9 >> 3 )
            goto LABEL_30;
          sub_25E1620(a1, v9);
          v28 = *(_DWORD *)(a1 + 24);
          if ( !v28 )
          {
LABEL_55:
            ++*(_DWORD *)(a1 + 16);
            BUG();
          }
          v29 = v28 - 1;
          v30 = *(_QWORD *)(a1 + 8);
          v31 = 1;
          v27 = 0;
          v18 = *(_DWORD *)(a1 + 16) + 1;
          v32 = (v28 - 1) & (((unsigned int)*v6 >> 9) ^ ((unsigned int)*v6 >> 4));
          v11 = (_QWORD *)(v30 + 16LL * v32);
          v33 = *v11;
          if ( *v11 == *v6 )
            goto LABEL_30;
          while ( v33 != -4096 )
          {
            if ( v33 == -8192 && !v27 )
              v27 = v11;
            v32 = v29 & (v31 + v32);
            v11 = (_QWORD *)(v30 + 16LL * v32);
            v33 = *v11;
            if ( *v6 == *v11 )
              goto LABEL_30;
            ++v31;
          }
          goto LABEL_39;
        }
LABEL_35:
        sub_25E1620(a1, 2 * v9);
        v21 = *(_DWORD *)(a1 + 24);
        if ( !v21 )
          goto LABEL_55;
        v22 = v21 - 1;
        v23 = *(_QWORD *)(a1 + 8);
        v24 = v22 & (((unsigned int)*v6 >> 9) ^ ((unsigned int)*v6 >> 4));
        v18 = *(_DWORD *)(a1 + 16) + 1;
        v11 = (_QWORD *)(v23 + 16LL * v24);
        v25 = *v11;
        if ( *v11 == *v6 )
          goto LABEL_30;
        v26 = 1;
        v27 = 0;
        while ( v25 != -4096 )
        {
          if ( v25 == -8192 && !v27 )
            v27 = v11;
          v24 = v22 & (v26 + v24);
          v11 = (_QWORD *)(v23 + 16LL * v24);
          v25 = *v11;
          if ( *v6 == *v11 )
            goto LABEL_30;
          ++v26;
        }
LABEL_39:
        if ( v27 )
          v11 = v27;
LABEL_30:
        *(_DWORD *)(a1 + 16) = v18;
        if ( *v11 != -4096 )
          --*(_DWORD *)(a1 + 20);
        v19 = *v6;
        v11[1] = 0;
        *v11 = v19;
        v16 = v11 + 1;
LABEL_15:
        v6 += 2;
        *v16 = v8;
        if ( v6 == v5 )
          return a1;
        while ( *v6 == -8192 || *v6 == -4096 )
        {
          v6 += 2;
          if ( v5 == v6 )
            return a1;
        }
      }
    }
  }
  return a1;
}
