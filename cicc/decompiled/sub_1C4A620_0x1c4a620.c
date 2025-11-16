// Function: sub_1C4A620
// Address: 0x1c4a620
//
__int64 __fastcall sub_1C4A620(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // r12
  __int64 v6; // rdx
  unsigned int v7; // esi
  __int64 v8; // r13
  __int64 v9; // r8
  unsigned int v10; // edi
  _QWORD *v11; // rcx
  __int64 v12; // r10
  __int64 v13; // rsi
  __int64 v14; // rcx
  __int64 v15; // r9
  int v16; // r11d
  _QWORD *v17; // r10
  int v18; // eax
  int v19; // ecx
  int v20; // eax
  int v21; // edi
  __int64 v22; // r8
  __int64 v23; // rax
  __int64 v24; // rsi
  int v25; // r11d
  _QWORD *v26; // r9
  int v27; // eax
  int v28; // esi
  __int64 v29; // r8
  _QWORD *v30; // rdi
  __int64 v31; // r14
  int v32; // r9d
  __int64 v33; // rax
  __int64 v34; // [rsp-48h] [rbp-48h]
  __int64 v35; // [rsp-48h] [rbp-48h]
  __int64 v36; // [rsp-40h] [rbp-40h]

  result = *(unsigned int *)(a2 + 8);
  if ( (_DWORD)result )
  {
    v4 = 0;
    v36 = a1 + 8;
    v6 = 8 * result;
    do
    {
      v7 = *(_DWORD *)(a1 + 32);
      v8 = *(_QWORD *)(*(_QWORD *)a2 + v4);
      if ( v7 )
      {
        v9 = *(_QWORD *)(a1 + 16);
        v10 = (v7 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
        v11 = (_QWORD *)(v9 + 8LL * v10);
        result = *v11;
        if ( v8 == *v11 )
          goto LABEL_5;
        v16 = 1;
        v17 = 0;
        while ( result != -8 )
        {
          if ( v17 || result != -16 )
            v11 = v17;
          v10 = (v7 - 1) & (v16 + v10);
          result = *(_QWORD *)(v9 + 8LL * v10);
          if ( v8 == result )
            goto LABEL_5;
          ++v16;
          v17 = v11;
          v11 = (_QWORD *)(v9 + 8LL * v10);
        }
        v18 = *(_DWORD *)(a1 + 24);
        if ( !v17 )
          v17 = v11;
        ++*(_QWORD *)(a1 + 8);
        v19 = v18 + 1;
        if ( 4 * (v18 + 1) < 3 * v7 )
        {
          if ( v7 - *(_DWORD *)(a1 + 28) - v19 <= v7 >> 3 )
          {
            v35 = v6;
            sub_1C4A480(v36, v7);
            v27 = *(_DWORD *)(a1 + 32);
            if ( !v27 )
            {
LABEL_53:
              ++*(_DWORD *)(a1 + 24);
              BUG();
            }
            v28 = v27 - 1;
            v29 = *(_QWORD *)(a1 + 16);
            v30 = 0;
            LODWORD(v31) = (v27 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
            v32 = 1;
            v19 = *(_DWORD *)(a1 + 24) + 1;
            v6 = v35;
            v17 = (_QWORD *)(v29 + 8LL * (unsigned int)v31);
            v33 = *v17;
            if ( v8 != *v17 )
            {
              while ( v33 != -8 )
              {
                if ( !v30 && v33 == -16 )
                  v30 = v17;
                v31 = v28 & (unsigned int)(v31 + v32);
                v17 = (_QWORD *)(v29 + 8 * v31);
                v33 = *v17;
                if ( v8 == *v17 )
                  goto LABEL_20;
                ++v32;
              }
              if ( v30 )
                v17 = v30;
            }
          }
          goto LABEL_20;
        }
      }
      else
      {
        ++*(_QWORD *)(a1 + 8);
      }
      v34 = v6;
      sub_1C4A480(v36, 2 * v7);
      v20 = *(_DWORD *)(a1 + 32);
      if ( !v20 )
        goto LABEL_53;
      v21 = v20 - 1;
      v22 = *(_QWORD *)(a1 + 16);
      LODWORD(v23) = (v20 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v17 = (_QWORD *)(v22 + 8LL * (unsigned int)v23);
      v19 = *(_DWORD *)(a1 + 24) + 1;
      v6 = v34;
      v24 = *v17;
      if ( v8 != *v17 )
      {
        v25 = 1;
        v26 = 0;
        while ( v24 != -8 )
        {
          if ( v24 == -16 && !v26 )
            v26 = v17;
          v23 = v21 & (unsigned int)(v23 + v25);
          v17 = (_QWORD *)(v22 + 8 * v23);
          v24 = *v17;
          if ( v8 == *v17 )
            goto LABEL_20;
          ++v25;
        }
        if ( v26 )
          v17 = v26;
      }
LABEL_20:
      *(_DWORD *)(a1 + 24) = v19;
      if ( *v17 != -8 )
        --*(_DWORD *)(a1 + 28);
      *v17 = v8;
      result = *(_QWORD *)(*(_QWORD *)a2 + v4);
LABEL_5:
      v12 = ***(_QWORD ***)(a1 + 48);
      v13 = ***(_QWORD ***)(result + 48);
      v14 = 0;
      v15 = 8LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF);
      if ( (*(_DWORD *)(v13 + 20) & 0xFFFFFFF) != 0 )
      {
        while ( 1 )
        {
          if ( (*(_BYTE *)(v13 + 23) & 0x40) != 0 )
          {
            if ( v12 != *(_QWORD *)(*(_QWORD *)(v13 - 8) + 3 * v14) )
              goto LABEL_8;
LABEL_11:
            *(_QWORD *)(*(_QWORD *)(result + 56) + v14) = a1;
            v14 += 8;
            if ( v15 == v14 )
              break;
          }
          else
          {
            if ( v12 == *(_QWORD *)(v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF) + 3 * v14) )
              goto LABEL_11;
LABEL_8:
            v14 += 8;
            if ( v15 == v14 )
              break;
          }
        }
      }
      v4 += 8;
    }
    while ( v6 != v4 );
  }
  return result;
}
