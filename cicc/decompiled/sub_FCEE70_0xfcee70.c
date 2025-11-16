// Function: sub_FCEE70
// Address: 0xfcee70
//
__int64 __fastcall sub_FCEE70(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned int v5; // r8d
  __int64 v6; // rcx
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // r10
  __int64 v10; // r15
  __int64 result; // rax
  __int64 v12; // r9
  __int64 v13; // rdx
  __int64 v14; // r14
  __int64 v15; // rbx
  unsigned int v16; // esi
  __int64 v17; // r8
  int v18; // r11d
  __int64 *v19; // rdx
  unsigned int v20; // edi
  __int64 *v21; // rax
  __int64 v22; // rcx
  unsigned int v23; // ecx
  __int64 v24; // rdx
  __int64 v25; // rax
  int v26; // eax
  int v27; // ecx
  int v28; // eax
  int v29; // edi
  __int64 v30; // rsi
  unsigned int v31; // eax
  __int64 v32; // r8
  int v33; // r11d
  __int64 *v34; // r10
  int v35; // eax
  int v36; // eax
  __int64 v37; // rdi
  __int64 *v38; // r8
  unsigned int v39; // r13d
  int v40; // r10d
  __int64 v41; // rsi
  int v42; // eax
  int v43; // r11d
  __int64 v44; // [rsp+8h] [rbp-38h]
  __int64 v45; // [rsp+8h] [rbp-38h]

  v5 = *(_DWORD *)(a1 + 136);
  v6 = *(_QWORD *)(a1 + 120);
  if ( !v5 )
  {
LABEL_42:
    v8 = (__int64 *)(v6 + 16LL * v5);
    goto LABEL_3;
  }
  v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v6 + 16LL * v7);
  v9 = *v8;
  if ( a2 != *v8 )
  {
    v42 = 1;
    while ( v9 != -4096 )
    {
      v43 = v42 + 1;
      v7 = (v5 - 1) & (v42 + v7);
      v8 = (__int64 *)(v6 + 16LL * v7);
      v9 = *v8;
      if ( a2 == *v8 )
        goto LABEL_3;
      v42 = v43;
    }
    goto LABEL_42;
  }
LABEL_3:
  v10 = v8[1];
  result = sub_AA5930(a3);
  v12 = a1 + 56;
  v14 = v13;
  v15 = result;
LABEL_4:
  if ( v14 != v15 )
  {
    do
    {
      v16 = *(_DWORD *)(a1 + 80);
      if ( v16 )
      {
        v17 = *(_QWORD *)(a1 + 64);
        v18 = 1;
        v19 = 0;
        v20 = (v16 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
        v21 = (__int64 *)(v17 + 16LL * v20);
        v22 = *v21;
        if ( v15 == *v21 )
        {
LABEL_7:
          v23 = *((_DWORD *)v21 + 2);
          v24 = ~(1LL << v23);
          v25 = 8LL * (v23 >> 6);
          goto LABEL_8;
        }
        while ( v22 != -4096 )
        {
          if ( v22 == -8192 && !v19 )
            v19 = v21;
          v20 = (v16 - 1) & (v18 + v20);
          v21 = (__int64 *)(v17 + 16LL * v20);
          v22 = *v21;
          if ( v15 == *v21 )
            goto LABEL_7;
          ++v18;
        }
        if ( !v19 )
          v19 = v21;
        v26 = *(_DWORD *)(a1 + 72);
        ++*(_QWORD *)(a1 + 56);
        v27 = v26 + 1;
        if ( 4 * (v26 + 1) < 3 * v16 )
        {
          if ( v16 - *(_DWORD *)(a1 + 76) - v27 <= v16 >> 3 )
          {
            v45 = v12;
            sub_CE2410(v12, v16);
            v35 = *(_DWORD *)(a1 + 80);
            if ( !v35 )
            {
LABEL_57:
              ++*(_DWORD *)(a1 + 72);
              BUG();
            }
            v36 = v35 - 1;
            v37 = *(_QWORD *)(a1 + 64);
            v38 = 0;
            v39 = v36 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
            v12 = v45;
            v40 = 1;
            v27 = *(_DWORD *)(a1 + 72) + 1;
            v19 = (__int64 *)(v37 + 16LL * v39);
            v41 = *v19;
            if ( v15 != *v19 )
            {
              while ( v41 != -4096 )
              {
                if ( !v38 && v41 == -8192 )
                  v38 = v19;
                v39 = v36 & (v40 + v39);
                v19 = (__int64 *)(v37 + 16LL * v39);
                v41 = *v19;
                if ( v15 == *v19 )
                  goto LABEL_23;
                ++v40;
              }
              if ( v38 )
                v19 = v38;
            }
          }
          goto LABEL_23;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 56);
      }
      v44 = v12;
      sub_CE2410(v12, 2 * v16);
      v28 = *(_DWORD *)(a1 + 80);
      if ( !v28 )
        goto LABEL_57;
      v29 = v28 - 1;
      v30 = *(_QWORD *)(a1 + 64);
      v12 = v44;
      v31 = (v28 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v27 = *(_DWORD *)(a1 + 72) + 1;
      v19 = (__int64 *)(v30 + 16LL * v31);
      v32 = *v19;
      if ( v15 != *v19 )
      {
        v33 = 1;
        v34 = 0;
        while ( v32 != -4096 )
        {
          if ( v32 == -8192 && !v34 )
            v34 = v19;
          v31 = v29 & (v33 + v31);
          v19 = (__int64 *)(v30 + 16LL * v31);
          v32 = *v19;
          if ( v15 == *v19 )
            goto LABEL_23;
          ++v33;
        }
        if ( v34 )
          v19 = v34;
      }
LABEL_23:
      *(_DWORD *)(a1 + 72) = v27;
      if ( *v19 != -4096 )
        --*(_DWORD *)(a1 + 76);
      *v19 = v15;
      v25 = 0;
      *((_DWORD *)v19 + 2) = 0;
      v24 = -2;
LABEL_8:
      *(_QWORD *)(*(_QWORD *)(v10 + 96) + v25) &= v24;
      if ( !v15 )
        BUG();
      result = *(_QWORD *)(v15 + 32);
      if ( !result )
        BUG();
      v15 = 0;
      if ( *(_BYTE *)(result - 24) != 84 )
        goto LABEL_4;
      v15 = result - 24;
    }
    while ( v14 != result - 24 );
  }
  return result;
}
