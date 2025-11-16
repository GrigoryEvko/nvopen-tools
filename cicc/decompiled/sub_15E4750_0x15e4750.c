// Function: sub_15E4750
// Address: 0x15e4750
//
__int64 __fastcall sub_15E4750(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rbx
  _BYTE *v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rdx
  unsigned int v9; // eax
  unsigned int v10; // r14d
  __int64 v11; // r8
  unsigned int v12; // edi
  _QWORD *v13; // rcx
  _QWORD *v14; // rdx
  __int64 v15; // rdx
  _QWORD *v16; // r13
  unsigned int v17; // esi
  int v18; // eax
  int v19; // ecx
  __int64 v20; // rdi
  unsigned int v21; // eax
  _QWORD *v22; // r10
  __int64 v23; // rsi
  int v24; // edx
  int v25; // r11d
  int v26; // eax
  int v27; // eax
  int v28; // eax
  __int64 v29; // rsi
  int v30; // r8d
  unsigned int v31; // r15d
  _QWORD *v32; // rdi
  __int64 v33; // rcx
  int v34; // r9d
  _QWORD *v35; // r8

  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_DWORD *)(a1 + 24) = 0;
  v3 = sub_1626AA0(a2, 2);
  if ( v3 )
  {
    v4 = v3;
    v5 = *(_BYTE **)(v3 - 8LL * *(unsigned int *)(v3 + 8));
    if ( !*v5 )
    {
      v6 = sub_161E970(v5);
      if ( v7 == 20
        && !(*(_QWORD *)v6 ^ 0x6E6F6974636E7566LL | *(_QWORD *)(v6 + 8) ^ 0x635F7972746E655FLL)
        && *(_DWORD *)(v6 + 16) == 1953396079 )
      {
        v9 = *(_DWORD *)(v4 + 8);
        if ( v9 > 2 )
        {
          v10 = 2;
          while ( 1 )
          {
            v15 = *(_QWORD *)(*(_QWORD *)(v4 + 8 * (v10 - (unsigned __int64)v9)) + 136LL);
            v16 = *(_QWORD **)(v15 + 24);
            if ( *(_DWORD *)(v15 + 32) > 0x40u )
              v16 = (_QWORD *)*v16;
            v17 = *(_DWORD *)(a1 + 24);
            if ( !v17 )
              break;
            v11 = *(_QWORD *)(a1 + 8);
            v12 = (v17 - 1) & (37 * (_DWORD)v16);
            v13 = (_QWORD *)(v11 + 8LL * v12);
            v14 = (_QWORD *)*v13;
            if ( v16 != (_QWORD *)*v13 )
            {
              v25 = 1;
              v22 = 0;
              while ( v14 != (_QWORD *)-1LL )
              {
                if ( v22 || v14 != (_QWORD *)-2LL )
                  v13 = v22;
                v12 = (v17 - 1) & (v25 + v12);
                v14 = *(_QWORD **)(v11 + 8LL * v12);
                if ( v16 == v14 )
                  goto LABEL_10;
                ++v25;
                v22 = v13;
                v13 = (_QWORD *)(v11 + 8LL * v12);
              }
              v26 = *(_DWORD *)(a1 + 16);
              if ( !v22 )
                v22 = v13;
              ++*(_QWORD *)a1;
              v24 = v26 + 1;
              if ( 4 * (v26 + 1) < 3 * v17 )
              {
                if ( v17 - *(_DWORD *)(a1 + 20) - v24 <= v17 >> 3 )
                {
                  sub_142F750(a1, v17);
                  v27 = *(_DWORD *)(a1 + 24);
                  if ( !v27 )
                  {
LABEL_52:
                    ++*(_DWORD *)(a1 + 16);
                    BUG();
                  }
                  v28 = v27 - 1;
                  v29 = *(_QWORD *)(a1 + 8);
                  v30 = 1;
                  v31 = v28 & (37 * (_DWORD)v16);
                  v22 = (_QWORD *)(v29 + 8LL * v31);
                  v24 = *(_DWORD *)(a1 + 16) + 1;
                  v32 = 0;
                  v33 = *v22;
                  if ( v16 != (_QWORD *)*v22 )
                  {
                    while ( v33 != -1 )
                    {
                      if ( v33 == -2 && !v32 )
                        v32 = v22;
                      v31 = v28 & (v30 + v31);
                      v22 = (_QWORD *)(v29 + 8LL * v31);
                      v33 = *v22;
                      if ( v16 == (_QWORD *)*v22 )
                        goto LABEL_17;
                      ++v30;
                    }
                    if ( v32 )
                      v22 = v32;
                  }
                }
                goto LABEL_17;
              }
LABEL_15:
              sub_142F750(a1, 2 * v17);
              v18 = *(_DWORD *)(a1 + 24);
              if ( !v18 )
                goto LABEL_52;
              v19 = v18 - 1;
              v20 = *(_QWORD *)(a1 + 8);
              v21 = (v18 - 1) & (37 * (_DWORD)v16);
              v22 = (_QWORD *)(v20 + 8LL * v21);
              v23 = *v22;
              v24 = *(_DWORD *)(a1 + 16) + 1;
              if ( v16 != (_QWORD *)*v22 )
              {
                v34 = 1;
                v35 = 0;
                while ( v23 != -1 )
                {
                  if ( v23 == -2 && !v35 )
                    v35 = v22;
                  v21 = v19 & (v34 + v21);
                  v22 = (_QWORD *)(v20 + 8LL * v21);
                  v23 = *v22;
                  if ( v16 == (_QWORD *)*v22 )
                    goto LABEL_17;
                  ++v34;
                }
                if ( v35 )
                  v22 = v35;
              }
LABEL_17:
              *(_DWORD *)(a1 + 16) = v24;
              if ( *v22 != -1 )
                --*(_DWORD *)(a1 + 20);
              *v22 = v16;
              v9 = *(_DWORD *)(v4 + 8);
            }
LABEL_10:
            if ( ++v10 >= v9 )
              return a1;
          }
          ++*(_QWORD *)a1;
          goto LABEL_15;
        }
      }
    }
  }
  return a1;
}
