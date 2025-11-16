// Function: sub_3994010
// Address: 0x3994010
//
void __fastcall sub_3994010(__int64 a1)
{
  _DWORD *v1; // r15
  _DWORD *v2; // rbx
  __int64 v4; // r9
  unsigned int v5; // r8d
  _QWORD *v6; // rax
  __int64 v7; // rcx
  __int64 v8; // r12
  __int64 v9; // r14
  unsigned int v10; // esi
  __int64 v11; // rdi
  int v12; // esi
  int v13; // esi
  __int64 v14; // r8
  unsigned int v15; // edx
  int v16; // ecx
  __int64 v17; // rdi
  int v18; // ecx
  int v19; // esi
  int v20; // esi
  __int64 v21; // r8
  _QWORD *v22; // r9
  int v23; // r10d
  unsigned int v24; // edx
  __int64 v25; // rdi
  int v26; // r10d
  int v27; // [rsp+14h] [rbp-3Ch]
  _QWORD *v28; // [rsp+18h] [rbp-38h]
  unsigned int v29; // [rsp+18h] [rbp-38h]

  v1 = *(_DWORD **)(a1 + 336);
  v2 = *(_DWORD **)(a1 + 328);
  if ( v2 != v1 )
  {
    while ( 1 )
    {
      if ( !v2[6] )
        goto LABEL_5;
      v8 = *(_QWORD *)(*(_QWORD *)v2 + 8 * (3LL - *(unsigned int *)(*(_QWORD *)v2 + 8LL)));
      if ( *(_BYTE *)v8 != 32 )
        goto LABEL_5;
      v9 = *(_QWORD *)(v8 + 8 * (3LL - *(unsigned int *)(v8 + 8)));
      if ( !v9 )
        goto LABEL_5;
      v10 = *(_DWORD *)(a1 + 5504);
      v11 = a1 + 5480;
      if ( !v10 )
        break;
      v4 = *(_QWORD *)(a1 + 5488);
      v5 = (v10 - 1) & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v6 = (_QWORD *)(v4 + 16LL * v5);
      v7 = *v6;
      if ( v9 != *v6 )
      {
        v27 = 1;
        v28 = 0;
        while ( v7 != -8 )
        {
          if ( !v28 )
          {
            if ( v7 != -16 )
              v6 = 0;
            v28 = v6;
          }
          v5 = (v10 - 1) & (v27 + v5);
          v6 = (_QWORD *)(v4 + 16LL * v5);
          v7 = *v6;
          if ( v9 == *v6 )
            goto LABEL_4;
          ++v27;
        }
        if ( v28 )
          v6 = v28;
        v18 = *(_DWORD *)(a1 + 5496);
        ++*(_QWORD *)(a1 + 5480);
        v16 = v18 + 1;
        if ( 4 * v16 < 3 * v10 )
        {
          if ( v10 - *(_DWORD *)(a1 + 5500) - v16 <= v10 >> 3 )
          {
            v29 = ((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4);
            sub_3993E50(v11, v10);
            v19 = *(_DWORD *)(a1 + 5504);
            if ( !v19 )
            {
LABEL_47:
              ++*(_DWORD *)(a1 + 5496);
              BUG();
            }
            v20 = v19 - 1;
            v21 = *(_QWORD *)(a1 + 5488);
            v22 = 0;
            v23 = 1;
            v24 = v20 & v29;
            v16 = *(_DWORD *)(a1 + 5496) + 1;
            v6 = (_QWORD *)(v21 + 16LL * (v20 & v29));
            v25 = *v6;
            if ( v9 != *v6 )
            {
              while ( v25 != -8 )
              {
                if ( !v22 && v25 == -16 )
                  v22 = v6;
                v24 = v20 & (v23 + v24);
                v6 = (_QWORD *)(v21 + 16LL * v24);
                v25 = *v6;
                if ( v9 == *v6 )
                  goto LABEL_13;
                ++v23;
              }
LABEL_33:
              if ( v22 )
                v6 = v22;
            }
          }
LABEL_13:
          *(_DWORD *)(a1 + 5496) = v16;
          if ( *v6 != -8 )
            --*(_DWORD *)(a1 + 5500);
          *v6 = v9;
          v6[1] = 0;
          goto LABEL_4;
        }
LABEL_11:
        sub_3993E50(v11, 2 * v10);
        v12 = *(_DWORD *)(a1 + 5504);
        if ( !v12 )
          goto LABEL_47;
        v13 = v12 - 1;
        v14 = *(_QWORD *)(a1 + 5488);
        v15 = v13 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v16 = *(_DWORD *)(a1 + 5496) + 1;
        v6 = (_QWORD *)(v14 + 16LL * v15);
        v17 = *v6;
        if ( v9 != *v6 )
        {
          v26 = 1;
          v22 = 0;
          while ( v17 != -8 )
          {
            if ( v17 == -16 && !v22 )
              v22 = v6;
            v15 = v13 & (v26 + v15);
            v6 = (_QWORD *)(v14 + 16LL * v15);
            v17 = *v6;
            if ( v9 == *v6 )
              goto LABEL_13;
            ++v26;
          }
          goto LABEL_33;
        }
        goto LABEL_13;
      }
LABEL_4:
      v6[1] = v8;
LABEL_5:
      v2 += 24;
      if ( v1 == v2 )
        return;
    }
    ++*(_QWORD *)(a1 + 5480);
    goto LABEL_11;
  }
}
