// Function: sub_2EF97A0
// Address: 0x2ef97a0
//
__int64 __fastcall sub_2EF97A0(__int64 a1, __int64 a2)
{
  _DWORD *v2; // rbx
  __int64 result; // rax
  __int64 v4; // r13
  __int64 v6; // r8
  unsigned int v7; // edx
  _DWORD *v8; // rdi
  int v9; // ecx
  unsigned int v10; // esi
  int v11; // eax
  int v12; // ecx
  __int64 v13; // r8
  unsigned int v14; // eax
  _DWORD *v15; // r10
  int v16; // edi
  int v17; // edx
  int v18; // r11d
  _DWORD *v19; // r9
  int v20; // r11d
  int v21; // eax
  int v22; // eax
  int v23; // eax
  __int64 v24; // r8
  int v25; // r11d
  unsigned int v26; // edi
  int v27; // esi

  v2 = *(_DWORD **)a2;
  result = *(unsigned int *)(a2 + 8);
  v4 = *(_QWORD *)a2 + 4 * result;
  if ( *(_QWORD *)a2 != v4 )
  {
    while ( 1 )
    {
      v10 = *(_DWORD *)(a1 + 24);
      if ( !v10 )
        break;
      result = (unsigned int)*v2;
      v6 = *(_QWORD *)(a1 + 8);
      v7 = (v10 - 1) & (37 * result);
      v8 = (_DWORD *)(v6 + 4LL * v7);
      v9 = *v8;
      if ( *v8 != (_DWORD)result )
      {
        v20 = 1;
        v15 = 0;
        while ( v9 != -1 )
        {
          if ( v15 || v9 != -2 )
            v8 = v15;
          v7 = (v10 - 1) & (v20 + v7);
          v9 = *(_DWORD *)(v6 + 4LL * v7);
          if ( (_DWORD)result == v9 )
            goto LABEL_4;
          ++v20;
          v15 = v8;
          v8 = (_DWORD *)(v6 + 4LL * v7);
        }
        v21 = *(_DWORD *)(a1 + 16);
        if ( !v15 )
          v15 = v8;
        ++*(_QWORD *)a1;
        v17 = v21 + 1;
        if ( 4 * (v21 + 1) < 3 * v10 )
        {
          if ( v10 - *(_DWORD *)(a1 + 20) - v17 <= v10 >> 3 )
          {
            sub_2E29BA0(a1, v10);
            v22 = *(_DWORD *)(a1 + 24);
            if ( !v22 )
            {
LABEL_42:
              ++*(_DWORD *)(a1 + 16);
              BUG();
            }
            v23 = v22 - 1;
            v24 = *(_QWORD *)(a1 + 8);
            v25 = 1;
            v19 = 0;
            v26 = v23 & (37 * *v2);
            v15 = (_DWORD *)(v24 + 4LL * v26);
            v27 = *v15;
            v17 = *(_DWORD *)(a1 + 16) + 1;
            if ( *v15 != *v2 )
            {
              while ( v27 != -1 )
              {
                if ( v27 == -2 && !v19 )
                  v19 = v15;
                v26 = v23 & (v25 + v26);
                v15 = (_DWORD *)(v24 + 4LL * v26);
                v27 = *v15;
                if ( *v2 == *v15 )
                  goto LABEL_21;
                ++v25;
              }
LABEL_11:
              if ( v19 )
                v15 = v19;
            }
          }
LABEL_21:
          *(_DWORD *)(a1 + 16) = v17;
          if ( *v15 != -1 )
            --*(_DWORD *)(a1 + 20);
          result = (unsigned int)*v2;
          *v15 = result;
          goto LABEL_4;
        }
LABEL_7:
        sub_2E29BA0(a1, 2 * v10);
        v11 = *(_DWORD *)(a1 + 24);
        if ( !v11 )
          goto LABEL_42;
        v12 = v11 - 1;
        v13 = *(_QWORD *)(a1 + 8);
        v14 = (v11 - 1) & (37 * *v2);
        v15 = (_DWORD *)(v13 + 4LL * (v12 & (unsigned int)(37 * *v2)));
        v16 = *v15;
        v17 = *(_DWORD *)(a1 + 16) + 1;
        if ( *v15 != *v2 )
        {
          v18 = 1;
          v19 = 0;
          while ( v16 != -1 )
          {
            if ( v16 == -2 && !v19 )
              v19 = v15;
            v14 = v12 & (v18 + v14);
            v15 = (_DWORD *)(v13 + 4LL * v14);
            v16 = *v15;
            if ( *v2 == *v15 )
              goto LABEL_21;
            ++v18;
          }
          goto LABEL_11;
        }
        goto LABEL_21;
      }
LABEL_4:
      if ( (_DWORD *)v4 == ++v2 )
        return result;
    }
    ++*(_QWORD *)a1;
    goto LABEL_7;
  }
  return result;
}
