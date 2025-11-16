// Function: sub_1FD4EF0
// Address: 0x1fd4ef0
//
void __fastcall sub_1FD4EF0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r11
  int v5; // r13d
  __int64 v7; // rbx
  __int64 v8; // rdi
  unsigned int v9; // ecx
  __int64 *v10; // rax
  __int64 v11; // rdx
  unsigned int v12; // esi
  int v13; // esi
  int v14; // esi
  __int64 v15; // r8
  unsigned int v16; // ecx
  int v17; // edx
  __int64 v18; // rdi
  int v19; // r14d
  __int64 *v20; // r9
  __int16 v21; // ax
  __int64 v22; // rax
  __int64 *v23; // r9
  int v24; // edx
  int v25; // ecx
  int v26; // ecx
  __int64 v27; // rdi
  __int64 *v28; // r8
  unsigned int v29; // r14d
  int v30; // r9d
  __int64 v31; // rsi
  __int64 v32; // [rsp+8h] [rbp-48h]
  __int64 v33; // [rsp+8h] [rbp-48h]
  __int64 v34; // [rsp+8h] [rbp-48h]
  __int64 v35; // [rsp+10h] [rbp-40h]
  int v36; // [rsp+10h] [rbp-40h]
  __int64 v37; // [rsp+10h] [rbp-40h]
  __int64 v38; // [rsp+10h] [rbp-40h]

  v3 = a2 + 24;
  v5 = 0;
  v7 = *(_QWORD *)(a2 + 32);
  if ( v7 != a2 + 24 )
  {
    while ( 1 )
    {
      if ( !*(_QWORD *)(a1 + 32) )
      {
        v21 = *(_WORD *)(v7 + 46);
        if ( (v21 & 4) == 0 && (v21 & 8) != 0 )
        {
          v33 = a3;
          v37 = v3;
          LOBYTE(v22) = sub_1E15D00(v7, 0x40u, 1);
          v3 = v37;
          a3 = v33;
        }
        else
        {
          v22 = (*(_QWORD *)(*(_QWORD *)(v7 + 16) + 8LL) >> 6) & 1LL;
        }
        if ( (_BYTE)v22 || **(_WORD **)(v7 + 16) == 3 && *(_QWORD *)(a2 + 32) != v7 )
        {
          *(_QWORD *)(a1 + 32) = v7;
          *(_DWORD *)(a1 + 40) = v5;
        }
      }
      v12 = *(_DWORD *)(a1 + 24);
      if ( v12 )
      {
        v8 = *(_QWORD *)(a1 + 8);
        v9 = (v12 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
        v10 = (__int64 *)(v8 + 16LL * v9);
        v11 = *v10;
        if ( v7 == *v10 )
          goto LABEL_4;
        v36 = 1;
        v23 = 0;
        while ( v11 != -8 )
        {
          if ( !v23 && v11 == -16 )
            v23 = v10;
          v9 = (v12 - 1) & (v36 + v9);
          v10 = (__int64 *)(v8 + 16LL * v9);
          v11 = *v10;
          if ( *v10 == v7 )
            goto LABEL_4;
          ++v36;
        }
        v24 = *(_DWORD *)(a1 + 16);
        if ( v23 )
          v10 = v23;
        ++*(_QWORD *)a1;
        v17 = v24 + 1;
        if ( 4 * v17 < 3 * v12 )
        {
          if ( v12 - *(_DWORD *)(a1 + 20) - v17 <= v12 >> 3 )
          {
            v34 = a3;
            v38 = v3;
            sub_1DC6D40(a1, v12);
            v25 = *(_DWORD *)(a1 + 24);
            if ( !v25 )
            {
LABEL_62:
              ++*(_DWORD *)(a1 + 16);
              BUG();
            }
            v26 = v25 - 1;
            v27 = *(_QWORD *)(a1 + 8);
            v28 = 0;
            v29 = v26 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
            v3 = v38;
            a3 = v34;
            v30 = 1;
            v17 = *(_DWORD *)(a1 + 16) + 1;
            v10 = (__int64 *)(v27 + 16LL * v29);
            v31 = *v10;
            if ( v7 != *v10 )
            {
              while ( v31 != -8 )
              {
                if ( v31 == -16 && !v28 )
                  v28 = v10;
                v29 = v26 & (v30 + v29);
                v10 = (__int64 *)(v27 + 16LL * v29);
                v31 = *v10;
                if ( v7 == *v10 )
                  goto LABEL_36;
                ++v30;
              }
              if ( v28 )
                v10 = v28;
            }
          }
          goto LABEL_36;
        }
      }
      else
      {
        ++*(_QWORD *)a1;
      }
      v32 = a3;
      v35 = v3;
      sub_1DC6D40(a1, 2 * v12);
      v13 = *(_DWORD *)(a1 + 24);
      if ( !v13 )
        goto LABEL_62;
      v14 = v13 - 1;
      v15 = *(_QWORD *)(a1 + 8);
      v3 = v35;
      a3 = v32;
      v16 = v14 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v17 = *(_DWORD *)(a1 + 16) + 1;
      v10 = (__int64 *)(v15 + 16LL * v16);
      v18 = *v10;
      if ( *v10 != v7 )
      {
        v19 = 1;
        v20 = 0;
        while ( v18 != -8 )
        {
          if ( v18 == -16 && !v20 )
            v20 = v10;
          v16 = v14 & (v19 + v16);
          v10 = (__int64 *)(v15 + 16LL * v16);
          v18 = *v10;
          if ( v7 == *v10 )
            goto LABEL_36;
          ++v19;
        }
        if ( v20 )
          v10 = v20;
      }
LABEL_36:
      *(_DWORD *)(a1 + 16) = v17;
      if ( *v10 != -8 )
        --*(_DWORD *)(a1 + 20);
      *v10 = v7;
      *((_DWORD *)v10 + 2) = 0;
LABEL_4:
      *((_DWORD *)v10 + 2) = v5;
      if ( v7 == a3 )
        return;
      if ( !v7 )
        BUG();
      if ( (*(_BYTE *)v7 & 4) != 0 )
      {
        v7 = *(_QWORD *)(v7 + 8);
        if ( v3 == v7 )
          return;
      }
      else
      {
        while ( (*(_BYTE *)(v7 + 46) & 8) != 0 )
          v7 = *(_QWORD *)(v7 + 8);
        v7 = *(_QWORD *)(v7 + 8);
        if ( v3 == v7 )
          return;
      }
      ++v5;
    }
  }
}
