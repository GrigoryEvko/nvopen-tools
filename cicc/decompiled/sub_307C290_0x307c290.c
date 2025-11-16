// Function: sub_307C290
// Address: 0x307c290
//
__int64 __fastcall sub_307C290(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v5; // rcx
  unsigned int v6; // edi
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // r9
  __int64 result; // rax
  __int64 v11; // r12
  __int64 v12; // rbx
  __int64 v13; // r14
  __int64 v14; // rbx
  int v15; // r13d
  unsigned int v16; // esi
  __int64 v17; // r9
  int *v18; // rdx
  int v19; // r11d
  unsigned int v20; // r8d
  int *v21; // rax
  int v22; // edi
  unsigned int v23; // ecx
  __int64 v24; // rdx
  __int64 v25; // rax
  int v26; // eax
  int v27; // eax
  int v28; // ecx
  int v29; // ecx
  __int64 v30; // r8
  __int64 v31; // rsi
  int v32; // edi
  int v33; // r10d
  int *v34; // r9
  int v35; // esi
  int v36; // esi
  __int64 v37; // r8
  int v38; // r10d
  __int64 v39; // rcx
  int v40; // edi
  int v41; // eax
  int v42; // r10d
  int v43; // [rsp+4h] [rbp-4Ch]
  __int64 v44; // [rsp+8h] [rbp-48h]
  __int64 v45; // [rsp+10h] [rbp-40h]
  __int64 v46; // [rsp+18h] [rbp-38h]

  v5 = *(_QWORD *)(a1 + 120);
  v6 = *(_DWORD *)(a1 + 136);
  if ( v6 )
  {
    v7 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = (__int64 *)(v5 + 16LL * v7);
    v9 = *v8;
    if ( a2 == *v8 )
      goto LABEL_3;
    v41 = 1;
    while ( v9 != -4096 )
    {
      v42 = v41 + 1;
      v7 = (v6 - 1) & (v41 + v7);
      v8 = (__int64 *)(v5 + 16LL * v7);
      v9 = *v8;
      if ( a2 == *v8 )
        goto LABEL_3;
      v41 = v42;
    }
  }
  v8 = (__int64 *)(v5 + 16LL * v6);
LABEL_3:
  v46 = v8[1];
  result = sub_2E311E0(a3);
  v11 = *(_QWORD *)(a3 + 56);
  v45 = result;
  v44 = a1 + 56;
  if ( v11 != result )
  {
    while ( 1 )
    {
      v12 = *(_QWORD *)(v11 + 32);
      result = 5LL * (unsigned int)sub_2E88FE0(v11);
      v13 = v12 + 8 * result;
      v14 = *(_QWORD *)(v11 + 32);
      if ( v13 != v14 )
        break;
LABEL_13:
      if ( (*(_BYTE *)v11 & 4) == 0 )
      {
        while ( (*(_BYTE *)(v11 + 44) & 8) != 0 )
          v11 = *(_QWORD *)(v11 + 8);
      }
      v11 = *(_QWORD *)(v11 + 8);
      if ( v45 == v11 )
        return result;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        if ( !*(_BYTE *)v14 )
        {
          v15 = *(_DWORD *)(v14 + 8);
          if ( v15 < 0 )
            break;
        }
        v14 += 40;
        if ( v13 == v14 )
          goto LABEL_13;
      }
      v16 = *(_DWORD *)(a1 + 80);
      if ( !v16 )
        break;
      v17 = *(_QWORD *)(a1 + 64);
      v18 = 0;
      v19 = 1;
      v20 = (v16 - 1) & (37 * v15);
      v21 = (int *)(v17 + 8LL * v20);
      v22 = *v21;
      if ( v15 != *v21 )
      {
        while ( v22 != -1 )
        {
          if ( v22 == -2 && !v18 )
            v18 = v21;
          v20 = (v16 - 1) & (v19 + v20);
          v21 = (int *)(v17 + 8LL * v20);
          v22 = *v21;
          if ( v15 == *v21 )
            goto LABEL_11;
          ++v19;
        }
        if ( !v18 )
          v18 = v21;
        v26 = *(_DWORD *)(a1 + 72);
        ++*(_QWORD *)(a1 + 56);
        v27 = v26 + 1;
        if ( 4 * v27 < 3 * v16 )
        {
          if ( v16 - *(_DWORD *)(a1 + 76) - v27 <= v16 >> 3 )
          {
            v43 = 37 * v15;
            sub_2E518D0(v44, v16);
            v35 = *(_DWORD *)(a1 + 80);
            if ( !v35 )
            {
LABEL_58:
              ++*(_DWORD *)(a1 + 72);
              BUG();
            }
            v36 = v35 - 1;
            v37 = *(_QWORD *)(a1 + 64);
            v38 = 1;
            v34 = 0;
            LODWORD(v39) = v36 & v43;
            v18 = (int *)(v37 + 8LL * (v36 & (unsigned int)v43));
            v40 = *v18;
            v27 = *(_DWORD *)(a1 + 72) + 1;
            if ( v15 != *v18 )
            {
              while ( v40 != -1 )
              {
                if ( v40 == -2 && !v34 )
                  v34 = v18;
                v39 = v36 & (unsigned int)(v39 + v38);
                v18 = (int *)(v37 + 8 * v39);
                v40 = *v18;
                if ( v15 == *v18 )
                  goto LABEL_26;
                ++v38;
              }
              goto LABEL_37;
            }
          }
          goto LABEL_26;
        }
LABEL_33:
        sub_2E518D0(v44, 2 * v16);
        v28 = *(_DWORD *)(a1 + 80);
        if ( !v28 )
          goto LABEL_58;
        v29 = v28 - 1;
        v30 = *(_QWORD *)(a1 + 64);
        LODWORD(v31) = v29 & (37 * v15);
        v18 = (int *)(v30 + 8LL * (unsigned int)v31);
        v32 = *v18;
        v27 = *(_DWORD *)(a1 + 72) + 1;
        if ( v15 != *v18 )
        {
          v33 = 1;
          v34 = 0;
          while ( v32 != -1 )
          {
            if ( v32 == -2 && !v34 )
              v34 = v18;
            v31 = v29 & (unsigned int)(v31 + v33);
            v18 = (int *)(v30 + 8 * v31);
            v32 = *v18;
            if ( v15 == *v18 )
              goto LABEL_26;
            ++v33;
          }
LABEL_37:
          if ( v34 )
            v18 = v34;
        }
LABEL_26:
        *(_DWORD *)(a1 + 72) = v27;
        if ( *v18 != -1 )
          --*(_DWORD *)(a1 + 76);
        *v18 = v15;
        v25 = 0;
        v18[1] = 0;
        v24 = -2;
        goto LABEL_12;
      }
LABEL_11:
      v23 = v21[1];
      v24 = ~(1LL << v23);
      v25 = 8LL * (v23 >> 6);
LABEL_12:
      v14 += 40;
      result = *(_QWORD *)(v46 + 96) + v25;
      *(_QWORD *)result &= v24;
      if ( v13 == v14 )
        goto LABEL_13;
    }
    ++*(_QWORD *)(a1 + 56);
    goto LABEL_33;
  }
  return result;
}
