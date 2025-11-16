// Function: sub_39AA770
// Address: 0x39aa770
//
void __fastcall sub_39AA770(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  int v5; // r11d
  __int64 v6; // r9
  __int64 v7; // r10
  __int64 v8; // rdi
  unsigned int v9; // ecx
  __int64 *v10; // rax
  __int64 v11; // rdx
  unsigned int v12; // esi
  int v13; // r15d
  __int64 v14; // r13
  int v15; // esi
  int v16; // esi
  __int64 v17; // r8
  unsigned int v18; // ecx
  int v19; // edx
  __int64 v20; // rdi
  int v21; // edx
  int v22; // ecx
  int v23; // ecx
  __int64 v24; // rdi
  unsigned int v25; // r14d
  __int64 v26; // rsi
  __int64 *v27; // r8
  __int64 *v28; // r14
  int v29; // [rsp+0h] [rbp-60h]
  int v30; // [rsp+0h] [rbp-60h]
  __int64 v31; // [rsp+8h] [rbp-58h]
  int v32; // [rsp+8h] [rbp-58h]
  __int64 v33; // [rsp+8h] [rbp-58h]
  __int64 v34; // [rsp+10h] [rbp-50h]
  __int64 *v35; // [rsp+10h] [rbp-50h]
  __int64 v36; // [rsp+10h] [rbp-50h]
  int v37; // [rsp+10h] [rbp-50h]
  int v38; // [rsp+10h] [rbp-50h]
  __int64 v40; // [rsp+20h] [rbp-40h]
  __int64 v41; // [rsp+28h] [rbp-38h]

  v41 = 0;
  v40 = *(unsigned int *)(a2 + 8);
  if ( (_DWORD)v40 )
  {
    while ( 1 )
    {
      v4 = 0;
      v5 = v41;
      v6 = *(_QWORD *)(*(_QWORD *)a2 + 8 * v41);
      v7 = *(unsigned int *)(v6 + 16);
      if ( (_DWORD)v7 )
        break;
LABEL_13:
      if ( ++v41 == v40 )
        return;
    }
    while ( 1 )
    {
      v12 = *(_DWORD *)(a3 + 24);
      v13 = v4;
      v14 = *(_QWORD *)(*(_QWORD *)(v6 + 8) + 8 * v4);
      if ( !v12 )
        break;
      v8 = *(_QWORD *)(a3 + 8);
      v9 = (v12 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v10 = (__int64 *)(v8 + 16LL * v9);
      v11 = *v10;
      if ( v14 != *v10 )
      {
        v32 = 1;
        v35 = 0;
        while ( v11 != -8 )
        {
          if ( !v35 )
          {
            if ( v11 != -16 )
              v10 = 0;
            v35 = v10;
          }
          v9 = (v12 - 1) & (v32 + v9);
          v10 = (__int64 *)(v8 + 16LL * v9);
          v11 = *v10;
          if ( v14 == *v10 )
            goto LABEL_5;
          ++v32;
        }
        if ( v35 )
          v10 = v35;
        v21 = *(_DWORD *)(a3 + 16);
        ++*(_QWORD *)a3;
        v19 = v21 + 1;
        if ( 4 * v19 < 3 * v12 )
        {
          if ( v12 - *(_DWORD *)(a3 + 20) - v19 <= v12 >> 3 )
          {
            v30 = v5;
            v33 = v7;
            v36 = v6;
            sub_39AA5B0(a3, v12);
            v22 = *(_DWORD *)(a3 + 24);
            if ( !v22 )
            {
LABEL_49:
              ++*(_DWORD *)(a3 + 16);
              BUG();
            }
            v23 = v22 - 1;
            v24 = *(_QWORD *)(a3 + 8);
            v25 = v23 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
            v6 = v36;
            v7 = v33;
            v19 = *(_DWORD *)(a3 + 16) + 1;
            v5 = v30;
            v10 = (__int64 *)(v24 + 16LL * v25);
            v26 = *v10;
            if ( v14 != *v10 )
            {
              v37 = 1;
              v27 = 0;
              while ( v26 != -8 )
              {
                if ( !v27 && v26 == -16 )
                  v27 = v10;
                v25 = v23 & (v37 + v25);
                v10 = (__int64 *)(v24 + 16LL * v25);
                v26 = *v10;
                if ( v14 == *v10 )
                  goto LABEL_10;
                ++v37;
              }
              if ( v27 )
                v10 = v27;
            }
          }
          goto LABEL_10;
        }
LABEL_8:
        v29 = v5;
        v31 = v7;
        v34 = v6;
        sub_39AA5B0(a3, 2 * v12);
        v15 = *(_DWORD *)(a3 + 24);
        if ( !v15 )
          goto LABEL_49;
        v16 = v15 - 1;
        v17 = *(_QWORD *)(a3 + 8);
        v6 = v34;
        v7 = v31;
        v5 = v29;
        v18 = v16 & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
        v19 = *(_DWORD *)(a3 + 16) + 1;
        v10 = (__int64 *)(v17 + 16LL * v18);
        v20 = *v10;
        if ( v14 != *v10 )
        {
          v38 = 1;
          v28 = 0;
          while ( v20 != -8 )
          {
            if ( !v28 && v20 == -16 )
              v28 = v10;
            v18 = v16 & (v38 + v18);
            v10 = (__int64 *)(v17 + 16LL * v18);
            v20 = *v10;
            if ( v14 == *v10 )
              goto LABEL_10;
            ++v38;
          }
          if ( v28 )
            v10 = v28;
        }
LABEL_10:
        *(_DWORD *)(a3 + 16) = v19;
        if ( *v10 != -8 )
          --*(_DWORD *)(a3 + 20);
        *v10 = v14;
        v10[1] = 0;
      }
LABEL_5:
      ++v4;
      *((_DWORD *)v10 + 2) = v5;
      *((_DWORD *)v10 + 3) = v13;
      if ( v7 == v4 )
        goto LABEL_13;
    }
    ++*(_QWORD *)a3;
    goto LABEL_8;
  }
}
