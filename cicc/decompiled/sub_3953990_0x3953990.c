// Function: sub_3953990
// Address: 0x3953990
//
__int64 __fastcall sub_3953990(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rdx
  __int64 v6; // r8
  unsigned int v7; // ecx
  __int64 *v8; // rax
  __int64 v9; // r10
  __int64 v10; // r15
  __int64 result; // rax
  __int64 v12; // r9
  __int64 v13; // rdx
  __int64 v14; // r14
  __int64 v15; // rbx
  __int64 v16; // r8
  unsigned int v17; // edi
  __int64 *v18; // rax
  __int64 v19; // rcx
  unsigned int v20; // ecx
  __int64 v21; // rax
  __int64 v22; // rdx
  unsigned int v23; // esi
  int v24; // eax
  int v25; // edi
  __int64 v26; // rsi
  unsigned int v27; // eax
  int v28; // ecx
  __int64 *v29; // rdx
  __int64 v30; // r8
  int v31; // r11d
  int v32; // eax
  int v33; // eax
  int v34; // eax
  __int64 v35; // rdi
  __int64 *v36; // r8
  unsigned int v37; // r13d
  int v38; // r10d
  __int64 v39; // rsi
  int v40; // eax
  int v41; // edx
  __int64 *v42; // r11
  int v43; // r11d
  int v44; // r11d
  __int64 *v45; // r10
  __int64 v46; // [rsp+8h] [rbp-38h]
  __int64 v47; // [rsp+8h] [rbp-38h]

  v5 = *(unsigned int *)(a1 + 136);
  v6 = *(_QWORD *)(a1 + 120);
  if ( (_DWORD)v5 )
  {
    v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = (__int64 *)(v6 + 16LL * v7);
    v9 = *v8;
    if ( a2 == *v8 )
      goto LABEL_3;
    v40 = 1;
    while ( v9 != -8 )
    {
      v43 = v40 + 1;
      v7 = (v5 - 1) & (v40 + v7);
      v8 = (__int64 *)(v6 + 16LL * v7);
      v9 = *v8;
      if ( a2 == *v8 )
        goto LABEL_3;
      v40 = v43;
    }
  }
  v8 = (__int64 *)(v6 + 16 * v5);
LABEL_3:
  v10 = v8[1];
  result = sub_157F280(a3);
  v12 = a1 + 56;
  v14 = v13;
  v15 = result;
  if ( result != v13 )
  {
    while ( 1 )
    {
      v23 = *(_DWORD *)(a1 + 80);
      if ( !v23 )
        break;
      v16 = *(_QWORD *)(a1 + 64);
      v17 = (v23 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v18 = (__int64 *)(v16 + 16LL * v17);
      v19 = *v18;
      if ( v15 == *v18 )
      {
        v20 = *((_DWORD *)v18 + 2);
        goto LABEL_7;
      }
      v31 = 1;
      v29 = 0;
      while ( 1 )
      {
        if ( v19 == -8 )
        {
          if ( !v29 )
            v29 = v18;
          v32 = *(_DWORD *)(a1 + 72);
          ++*(_QWORD *)(a1 + 56);
          v28 = v32 + 1;
          if ( 4 * (v32 + 1) < 3 * v23 )
          {
            if ( v23 - *(_DWORD *)(a1 + 76) - v28 <= v23 >> 3 )
            {
              v47 = v12;
              sub_1BFE340(v12, v23);
              v33 = *(_DWORD *)(a1 + 80);
              if ( v33 )
              {
                v34 = v33 - 1;
                v35 = *(_QWORD *)(a1 + 64);
                v36 = 0;
                v37 = v34 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
                v12 = v47;
                v38 = 1;
                v28 = *(_DWORD *)(a1 + 72) + 1;
                v29 = (__int64 *)(v35 + 16LL * v37);
                v39 = *v29;
                if ( v15 != *v29 )
                {
                  while ( v39 != -8 )
                  {
                    if ( !v36 && v39 == -16 )
                      v36 = v29;
                    v37 = v34 & (v38 + v37);
                    v29 = (__int64 *)(v35 + 16LL * v37);
                    v39 = *v29;
                    if ( v15 == *v29 )
                      goto LABEL_17;
                    ++v38;
                  }
                  if ( v36 )
                    v29 = v36;
                }
                goto LABEL_17;
              }
LABEL_61:
              ++*(_DWORD *)(a1 + 72);
              BUG();
            }
LABEL_17:
            *(_DWORD *)(a1 + 72) = v28;
            if ( *v29 != -8 )
              --*(_DWORD *)(a1 + 76);
            *v29 = v15;
            v21 = 0;
            *((_DWORD *)v29 + 2) = 0;
            v22 = -2;
            goto LABEL_8;
          }
LABEL_15:
          v46 = v12;
          sub_1BFE340(v12, 2 * v23);
          v24 = *(_DWORD *)(a1 + 80);
          if ( !v24 )
            goto LABEL_61;
          v25 = v24 - 1;
          v26 = *(_QWORD *)(a1 + 64);
          v12 = v46;
          v27 = (v24 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
          v28 = *(_DWORD *)(a1 + 72) + 1;
          v29 = (__int64 *)(v26 + 16LL * v27);
          v30 = *v29;
          if ( v15 != *v29 )
          {
            v44 = 1;
            v45 = 0;
            while ( v30 != -8 )
            {
              if ( v30 == -16 && !v45 )
                v45 = v29;
              v27 = v25 & (v44 + v27);
              v29 = (__int64 *)(v26 + 16LL * v27);
              v30 = *v29;
              if ( v15 == *v29 )
                goto LABEL_17;
              ++v44;
            }
            if ( v45 )
              v29 = v45;
          }
          goto LABEL_17;
        }
        if ( v19 != -16 || v29 )
          v18 = v29;
        v41 = v31 + 1;
        v17 = (v23 - 1) & (v31 + v17);
        v42 = (__int64 *)(v16 + 16LL * v17);
        v19 = *v42;
        if ( v15 == *v42 )
          break;
        v31 = v41;
        v29 = v18;
        v18 = (__int64 *)(v16 + 16LL * v17);
      }
      v20 = *((_DWORD *)v42 + 2);
LABEL_7:
      v21 = 8LL * (v20 >> 6);
      v22 = ~(1LL << v20);
LABEL_8:
      *(_QWORD *)(*(_QWORD *)(v10 + 48) + v21) &= v22;
      if ( !v15 )
        BUG();
      result = *(_QWORD *)(v15 + 32);
      if ( !result )
        BUG();
      v15 = 0;
      if ( *(_BYTE *)(result - 8) == 77 )
        v15 = result - 24;
      if ( v14 == v15 )
        return result;
    }
    ++*(_QWORD *)(a1 + 56);
    goto LABEL_15;
  }
  return result;
}
