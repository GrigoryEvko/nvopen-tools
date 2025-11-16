// Function: sub_2F6A220
// Address: 0x2f6a220
//
__int64 __fastcall sub_2F6A220(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rcx
  __int64 v5; // r8
  unsigned int v7; // esi
  __int64 *v8; // rax
  __int64 v9; // r9
  __int64 v10; // r12
  int v11; // edi
  unsigned int v12; // esi
  int v13; // eax
  __int64 v14; // r11
  unsigned int v15; // r13d
  unsigned int v16; // r10d
  _QWORD *v17; // rcx
  __int64 v18; // r9
  int v20; // eax
  int v21; // eax
  int v22; // ecx
  __int64 v23; // r9
  unsigned int v24; // eax
  _QWORD *v25; // r8
  __int64 v26; // rsi
  int v27; // r11d
  int v28; // eax
  int v29; // eax
  __int64 v30; // rsi
  _QWORD *v31; // r9
  unsigned int v32; // r13d
  int v33; // r10d
  __int64 v34; // rcx
  int v35; // r11d
  _QWORD *v36; // r10
  __int64 v37; // [rsp+8h] [rbp-38h]
  int v38; // [rsp+8h] [rbp-38h]
  __int64 v39; // [rsp+8h] [rbp-38h]

  v4 = *(unsigned int *)(a1 + 144);
  v5 = *(_QWORD *)(a1 + 128);
  if ( (_DWORD)v4 )
  {
    v7 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v8 = (__int64 *)(v5 + 16LL * v7);
    v9 = *v8;
    if ( a2 == *v8 )
    {
LABEL_3:
      if ( v8 != (__int64 *)(v5 + 16 * v4) )
      {
        v10 = v8[1];
        *(_QWORD *)((v10 & 0xFFFFFFFFFFFFFFF8LL) + 16) = a3;
        *v8 = -8192;
        v11 = *(_DWORD *)(a1 + 136);
        v12 = *(_DWORD *)(a1 + 144);
        *(_DWORD *)(a1 + 136) = v11 - 1;
        v13 = *(_DWORD *)(a1 + 140) + 1;
        *(_DWORD *)(a1 + 140) = v13;
        if ( v12 )
        {
          v14 = *(_QWORD *)(a1 + 128);
          v15 = ((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4);
          v16 = (v12 - 1) & v15;
          v17 = (_QWORD *)(v14 + 16LL * v16);
          v18 = *v17;
          if ( a3 == *v17 )
            return v10;
          v38 = 1;
          v25 = 0;
          while ( v18 != -4096 )
          {
            if ( v18 != -8192 || v25 )
              v17 = v25;
            v16 = (v12 - 1) & (v38 + v16);
            v18 = *(_QWORD *)(v14 + 16LL * v16);
            if ( a3 == v18 )
              return v10;
            ++v38;
            v25 = v17;
            v17 = (_QWORD *)(v14 + 16LL * v16);
          }
          if ( !v25 )
            v25 = v17;
          ++*(_QWORD *)(a1 + 120);
          if ( 4 * v11 < 3 * v12 )
          {
            if ( v12 - v11 - v13 > v12 >> 3 )
            {
LABEL_13:
              *(_DWORD *)(a1 + 136) = v11;
              if ( *v25 != -4096 )
                --*(_DWORD *)(a1 + 140);
              *v25 = a3;
              v25[1] = v10;
              return v10;
            }
            v39 = a3;
            sub_2E190F0(a1 + 120, v12);
            v28 = *(_DWORD *)(a1 + 144);
            if ( v28 )
            {
              v29 = v28 - 1;
              v30 = *(_QWORD *)(a1 + 128);
              v31 = 0;
              v32 = v29 & v15;
              v33 = 1;
              v11 = *(_DWORD *)(a1 + 136) + 1;
              a3 = v39;
              v25 = (_QWORD *)(v30 + 16LL * v32);
              v34 = *v25;
              if ( v39 != *v25 )
              {
                while ( v34 != -4096 )
                {
                  if ( v34 == -8192 && !v31 )
                    v31 = v25;
                  v32 = v29 & (v33 + v32);
                  v25 = (_QWORD *)(v30 + 16LL * v32);
                  v34 = *v25;
                  if ( v39 == *v25 )
                    goto LABEL_13;
                  ++v33;
                }
                if ( v31 )
                  v25 = v31;
              }
              goto LABEL_13;
            }
LABEL_50:
            ++*(_DWORD *)(a1 + 136);
            BUG();
          }
        }
        else
        {
          ++*(_QWORD *)(a1 + 120);
        }
        v37 = a3;
        sub_2E190F0(a1 + 120, 2 * v12);
        v21 = *(_DWORD *)(a1 + 144);
        if ( v21 )
        {
          a3 = v37;
          v22 = v21 - 1;
          v23 = *(_QWORD *)(a1 + 128);
          v11 = *(_DWORD *)(a1 + 136) + 1;
          v24 = (v21 - 1) & (((unsigned int)v37 >> 9) ^ ((unsigned int)v37 >> 4));
          v25 = (_QWORD *)(v23 + 16LL * v24);
          v26 = *v25;
          if ( v37 != *v25 )
          {
            v35 = 1;
            v36 = 0;
            while ( v26 != -4096 )
            {
              if ( v26 == -8192 && !v36 )
                v36 = v25;
              v24 = v22 & (v35 + v24);
              v25 = (_QWORD *)(v23 + 16LL * v24);
              v26 = *v25;
              if ( v37 == *v25 )
                goto LABEL_13;
              ++v35;
            }
            if ( v36 )
              v25 = v36;
          }
          goto LABEL_13;
        }
        goto LABEL_50;
      }
    }
    else
    {
      v20 = 1;
      while ( v9 != -4096 )
      {
        v27 = v20 + 1;
        v7 = (v4 - 1) & (v20 + v7);
        v8 = (__int64 *)(v5 + 16LL * v7);
        v9 = *v8;
        if ( a2 == *v8 )
          goto LABEL_3;
        v20 = v27;
      }
    }
  }
  return 0;
}
