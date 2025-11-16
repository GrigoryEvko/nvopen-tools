// Function: sub_1630C80
// Address: 0x1630c80
//
void __fastcall sub_1630C80(__int64 a1, __int64 *a2, __int64 *a3)
{
  __int64 v4; // r13
  __int64 *v5; // r12
  __int64 v7; // rcx
  int v8; // esi
  unsigned int v9; // eax
  _QWORD *v10; // r9
  __int64 v11; // r8
  __int64 v12; // r15
  char v13; // dl
  unsigned int v14; // esi
  unsigned int v15; // eax
  _QWORD *v16; // rdi
  int v17; // ecx
  __int64 v18; // r15
  __int64 v19; // rax
  int v20; // r10d
  int v21; // eax
  __int64 v22; // rcx
  int v23; // edx
  unsigned int v24; // eax
  __int64 v25; // rsi
  int v26; // edx
  __int64 v27; // rcx
  int v28; // edx
  unsigned int v29; // eax
  __int64 v30; // rsi
  int v31; // r9d
  _QWORD *v32; // r8
  int v33; // r9d

  if ( a2 != a3 )
  {
    v4 = a1 + 64;
    v5 = a2;
    while ( 1 )
    {
      v12 = *v5;
      v13 = *(_BYTE *)(a1 + 8) & 1;
      if ( v13 )
      {
        v7 = a1 + 16;
        v8 = 3;
      }
      else
      {
        v14 = *(_DWORD *)(a1 + 24);
        v7 = *(_QWORD *)(a1 + 16);
        if ( !v14 )
        {
          v15 = *(_DWORD *)(a1 + 8);
          ++*(_QWORD *)a1;
          v16 = 0;
          v17 = (v15 >> 1) + 1;
          goto LABEL_10;
        }
        v8 = v14 - 1;
      }
      v9 = v8 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v10 = (_QWORD *)(v7 + 8LL * v9);
      v11 = *v10;
      if ( v12 == *v10 )
      {
LABEL_5:
        if ( a3 == ++v5 )
          return;
      }
      else
      {
        v20 = 1;
        v16 = 0;
        while ( v11 != -4 )
        {
          if ( v16 || v11 != -8 )
            v10 = v16;
          v9 = v8 & (v20 + v9);
          v11 = *(_QWORD *)(v7 + 8LL * v9);
          if ( v12 == v11 )
            goto LABEL_5;
          ++v20;
          v16 = v10;
          v10 = (_QWORD *)(v7 + 8LL * v9);
        }
        v15 = *(_DWORD *)(a1 + 8);
        if ( !v16 )
          v16 = v10;
        ++*(_QWORD *)a1;
        v17 = (v15 >> 1) + 1;
        if ( !v13 )
        {
          v14 = *(_DWORD *)(a1 + 24);
LABEL_10:
          if ( 3 * v14 <= 4 * v17 )
            goto LABEL_24;
          goto LABEL_11;
        }
        v14 = 4;
        if ( (unsigned int)(4 * v17) >= 0xC )
        {
LABEL_24:
          sub_16308D0(a1, 2 * v14);
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v22 = a1 + 16;
            v23 = 3;
          }
          else
          {
            v21 = *(_DWORD *)(a1 + 24);
            v22 = *(_QWORD *)(a1 + 16);
            if ( !v21 )
              goto LABEL_58;
            v23 = v21 - 1;
          }
          v24 = v23 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
          v16 = (_QWORD *)(v22 + 8LL * v24);
          v25 = *v16;
          if ( v12 == *v16 )
            goto LABEL_28;
          v33 = 1;
          v32 = 0;
          while ( v25 != -4 )
          {
            if ( v25 == -8 && !v32 )
              v32 = v16;
            v24 = v23 & (v33 + v24);
            v16 = (_QWORD *)(v22 + 8LL * v24);
            v25 = *v16;
            if ( v12 == *v16 )
              goto LABEL_28;
            ++v33;
          }
LABEL_35:
          if ( v32 )
            v16 = v32;
LABEL_28:
          v15 = *(_DWORD *)(a1 + 8);
          goto LABEL_12;
        }
LABEL_11:
        if ( v14 - *(_DWORD *)(a1 + 12) - v17 <= v14 >> 3 )
        {
          sub_16308D0(a1, v14);
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v27 = a1 + 16;
            v28 = 3;
          }
          else
          {
            v26 = *(_DWORD *)(a1 + 24);
            v27 = *(_QWORD *)(a1 + 16);
            if ( !v26 )
            {
LABEL_58:
              *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
              BUG();
            }
            v28 = v26 - 1;
          }
          v29 = v28 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
          v16 = (_QWORD *)(v27 + 8LL * v29);
          v30 = *v16;
          if ( v12 == *v16 )
            goto LABEL_28;
          v31 = 1;
          v32 = 0;
          while ( v30 != -4 )
          {
            if ( v30 == -8 && !v32 )
              v32 = v16;
            v29 = v28 & (v31 + v29);
            v16 = (_QWORD *)(v27 + 8LL * v29);
            v30 = *v16;
            if ( v12 == *v16 )
              goto LABEL_28;
            ++v31;
          }
          goto LABEL_35;
        }
LABEL_12:
        *(_DWORD *)(a1 + 8) = (2 * (v15 >> 1) + 2) | v15 & 1;
        if ( *v16 != -4 )
          --*(_DWORD *)(a1 + 12);
        *v16 = v12;
        v18 = *v5;
        v19 = *(unsigned int *)(a1 + 56);
        if ( (unsigned int)v19 >= *(_DWORD *)(a1 + 60) )
        {
          sub_16CD150(a1 + 48, v4, 0, 8);
          v19 = *(unsigned int *)(a1 + 56);
        }
        ++v5;
        *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8 * v19) = v18;
        ++*(_DWORD *)(a1 + 56);
        if ( a3 == v5 )
          return;
      }
    }
  }
}
