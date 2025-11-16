// Function: sub_2871A60
// Address: 0x2871a60
//
__int64 __fastcall sub_2871A60(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 result; // rax
  __int64 *v9; // r12
  __int64 *v10; // r13
  __int64 v11; // r8
  __int64 *v12; // rdi
  __int64 v13; // rcx
  unsigned int v14; // esi
  int v15; // eax
  int v16; // ecx
  __int64 v17; // r8
  unsigned int v18; // eax
  __int64 *v19; // r10
  __int64 v20; // rdi
  int v21; // edx
  int v22; // r11d
  int v23; // eax
  int v24; // eax
  int v25; // ecx
  __int64 v26; // r8
  __int64 *v27; // r9
  int v28; // r11d
  unsigned int v29; // eax
  __int64 v30; // rdi
  int v31; // r11d

  v6 = *(unsigned int *)(a1 + 40);
  if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v6 + 1, 8u, a5, a6);
    v6 = *(unsigned int *)(a1 + 40);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v6) = a2;
  result = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
  *(_DWORD *)(a1 + 40) = result;
  if ( (unsigned int)result > 4 )
  {
    v9 = *(__int64 **)(a1 + 32);
    v10 = &v9[result];
    while ( 1 )
    {
      v14 = *(_DWORD *)(a1 + 24);
      if ( !v14 )
        break;
      v11 = *(_QWORD *)(a1 + 8);
      result = (v14 - 1) & (((unsigned int)*v9 >> 9) ^ ((unsigned int)*v9 >> 4));
      v12 = (__int64 *)(v11 + 8 * result);
      v13 = *v12;
      if ( *v9 != *v12 )
      {
        v22 = 1;
        v19 = 0;
        while ( v13 != -4096 )
        {
          if ( v19 || v13 != -8192 )
            v12 = v19;
          result = (v14 - 1) & (v22 + (_DWORD)result);
          v13 = *(_QWORD *)(v11 + 8LL * (unsigned int)result);
          if ( *v9 == v13 )
            goto LABEL_7;
          ++v22;
          v19 = v12;
          v12 = (__int64 *)(v11 + 8LL * (unsigned int)result);
        }
        v23 = *(_DWORD *)(a1 + 16);
        if ( !v19 )
          v19 = v12;
        ++*(_QWORD *)a1;
        v21 = v23 + 1;
        if ( 4 * (v23 + 1) < 3 * v14 )
        {
          if ( v14 - *(_DWORD *)(a1 + 20) - v21 <= v14 >> 3 )
          {
            sub_2871610(a1, v14);
            v24 = *(_DWORD *)(a1 + 24);
            if ( !v24 )
            {
LABEL_44:
              ++*(_DWORD *)(a1 + 16);
              BUG();
            }
            v25 = v24 - 1;
            v26 = *(_QWORD *)(a1 + 8);
            v27 = 0;
            v28 = 1;
            v29 = (v24 - 1) & (((unsigned int)*v9 >> 9) ^ ((unsigned int)*v9 >> 4));
            v19 = (__int64 *)(v26 + 8LL * v29);
            v30 = *v19;
            v21 = *(_DWORD *)(a1 + 16) + 1;
            if ( *v19 != *v9 )
            {
              while ( v30 != -4096 )
              {
                if ( v30 == -8192 && !v27 )
                  v27 = v19;
                v29 = v25 & (v28 + v29);
                v19 = (__int64 *)(v26 + 8LL * v29);
                v30 = *v19;
                if ( *v9 == *v19 )
                  goto LABEL_12;
                ++v28;
              }
LABEL_24:
              if ( v27 )
                v19 = v27;
            }
          }
LABEL_12:
          *(_DWORD *)(a1 + 16) = v21;
          if ( *v19 != -4096 )
            --*(_DWORD *)(a1 + 20);
          result = *v9;
          *v19 = *v9;
          goto LABEL_7;
        }
LABEL_10:
        sub_2871610(a1, 2 * v14);
        v15 = *(_DWORD *)(a1 + 24);
        if ( !v15 )
          goto LABEL_44;
        v16 = v15 - 1;
        v17 = *(_QWORD *)(a1 + 8);
        v18 = (v15 - 1) & (((unsigned int)*v9 >> 9) ^ ((unsigned int)*v9 >> 4));
        v19 = (__int64 *)(v17 + 8LL * v18);
        v20 = *v19;
        v21 = *(_DWORD *)(a1 + 16) + 1;
        if ( *v19 != *v9 )
        {
          v31 = 1;
          v27 = 0;
          while ( v20 != -4096 )
          {
            if ( v20 == -8192 && !v27 )
              v27 = v19;
            v18 = v16 & (v31 + v18);
            v19 = (__int64 *)(v17 + 8LL * v18);
            v20 = *v19;
            if ( *v9 == *v19 )
              goto LABEL_12;
            ++v31;
          }
          goto LABEL_24;
        }
        goto LABEL_12;
      }
LABEL_7:
      if ( v10 == ++v9 )
        return result;
    }
    ++*(_QWORD *)a1;
    goto LABEL_10;
  }
  return result;
}
