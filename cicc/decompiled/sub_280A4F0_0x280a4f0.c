// Function: sub_280A4F0
// Address: 0x280a4f0
//
__int64 __fastcall sub_280A4F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v8; // rax
  _QWORD *v10; // r12
  _QWORD *v11; // r13
  __int64 v12; // r8
  unsigned int v13; // eax
  _QWORD *v14; // rdi
  __int64 v15; // rcx
  unsigned int v16; // esi
  int v17; // eax
  int v18; // ecx
  __int64 v19; // r8
  unsigned int v20; // eax
  _QWORD *v21; // r10
  __int64 v22; // rdi
  int v23; // edx
  int v24; // r11d
  int v25; // eax
  int v26; // eax
  int v27; // ecx
  __int64 v28; // r8
  _QWORD *v29; // r9
  int v30; // r11d
  unsigned int v31; // eax
  __int64 v32; // rdi
  int v33; // r11d

  v6 = *(unsigned int *)(a1 + 40);
  if ( v6 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 44) )
  {
    sub_C8D5F0(a1 + 32, (const void *)(a1 + 48), v6 + 1, 8u, a5, a6);
    v6 = *(unsigned int *)(a1 + 40);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v6) = a2;
  v8 = (unsigned int)(*(_DWORD *)(a1 + 40) + 1);
  *(_DWORD *)(a1 + 40) = v8;
  if ( (unsigned int)v8 > 8 )
  {
    v10 = *(_QWORD **)(a1 + 32);
    v11 = &v10[v8];
    while ( 1 )
    {
      v16 = *(_DWORD *)(a1 + 24);
      if ( !v16 )
        break;
      v12 = *(_QWORD *)(a1 + 8);
      v13 = (v16 - 1) & (((unsigned int)*v10 >> 9) ^ ((unsigned int)*v10 >> 4));
      v14 = (_QWORD *)(v12 + 8LL * v13);
      v15 = *v14;
      if ( *v10 != *v14 )
      {
        v24 = 1;
        v21 = 0;
        while ( v15 != -4096 )
        {
          if ( v21 || v15 != -8192 )
            v14 = v21;
          v13 = (v16 - 1) & (v24 + v13);
          v15 = *(_QWORD *)(v12 + 8LL * v13);
          if ( *v10 == v15 )
            goto LABEL_7;
          ++v24;
          v21 = v14;
          v14 = (_QWORD *)(v12 + 8LL * v13);
        }
        v25 = *(_DWORD *)(a1 + 16);
        if ( !v21 )
          v21 = v14;
        ++*(_QWORD *)a1;
        v23 = v25 + 1;
        if ( 4 * (v25 + 1) < 3 * v16 )
        {
          if ( v16 - *(_DWORD *)(a1 + 20) - v23 <= v16 >> 3 )
          {
            sub_CF4090(a1, v16);
            v26 = *(_DWORD *)(a1 + 24);
            if ( !v26 )
            {
LABEL_44:
              ++*(_DWORD *)(a1 + 16);
              BUG();
            }
            v27 = v26 - 1;
            v28 = *(_QWORD *)(a1 + 8);
            v29 = 0;
            v30 = 1;
            v31 = (v26 - 1) & (((unsigned int)*v10 >> 9) ^ ((unsigned int)*v10 >> 4));
            v21 = (_QWORD *)(v28 + 8LL * v31);
            v32 = *v21;
            v23 = *(_DWORD *)(a1 + 16) + 1;
            if ( *v21 != *v10 )
            {
              while ( v32 != -4096 )
              {
                if ( v32 == -8192 && !v29 )
                  v29 = v21;
                v31 = v27 & (v30 + v31);
                v21 = (_QWORD *)(v28 + 8LL * v31);
                v32 = *v21;
                if ( *v10 == *v21 )
                  goto LABEL_12;
                ++v30;
              }
LABEL_24:
              if ( v29 )
                v21 = v29;
            }
          }
LABEL_12:
          *(_DWORD *)(a1 + 16) = v23;
          if ( *v21 != -4096 )
            --*(_DWORD *)(a1 + 20);
          *v21 = *v10;
          goto LABEL_7;
        }
LABEL_10:
        sub_CF4090(a1, 2 * v16);
        v17 = *(_DWORD *)(a1 + 24);
        if ( !v17 )
          goto LABEL_44;
        v18 = v17 - 1;
        v19 = *(_QWORD *)(a1 + 8);
        v20 = (v17 - 1) & (((unsigned int)*v10 >> 9) ^ ((unsigned int)*v10 >> 4));
        v21 = (_QWORD *)(v19 + 8LL * v20);
        v22 = *v21;
        v23 = *(_DWORD *)(a1 + 16) + 1;
        if ( *v21 != *v10 )
        {
          v33 = 1;
          v29 = 0;
          while ( v22 != -4096 )
          {
            if ( v22 == -8192 && !v29 )
              v29 = v21;
            v20 = v18 & (v33 + v20);
            v21 = (_QWORD *)(v19 + 8LL * v20);
            v22 = *v21;
            if ( *v10 == *v21 )
              goto LABEL_12;
            ++v33;
          }
          goto LABEL_24;
        }
        goto LABEL_12;
      }
LABEL_7:
      if ( v11 == ++v10 )
        return 1;
    }
    ++*(_QWORD *)a1;
    goto LABEL_10;
  }
  return 1;
}
