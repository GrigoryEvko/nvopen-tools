// Function: sub_1963390
// Address: 0x1963390
//
void __fastcall sub_1963390(__int64 a1, __int64 a2, __int64 a3)
{
  const void *v3; // r15
  __int64 v5; // r13
  __int64 v7; // rcx
  int v8; // esi
  unsigned int v9; // eax
  _QWORD *v10; // r9
  _QWORD *v11; // r8
  _QWORD *v12; // rax
  _QWORD *v13; // r12
  char v14; // dl
  unsigned int v15; // esi
  unsigned int v16; // eax
  _QWORD *v17; // rdi
  int v18; // ecx
  int v19; // r8d
  int v20; // r9d
  _QWORD *v21; // r12
  __int64 v22; // rax
  int v23; // r10d
  int v24; // eax
  __int64 v25; // rcx
  int v26; // edx
  unsigned int v27; // eax
  __int64 v28; // rsi
  int v29; // edx
  __int64 v30; // rcx
  int v31; // edx
  unsigned int v32; // eax
  __int64 v33; // rsi
  int v34; // r9d
  _QWORD *v35; // r8
  int v36; // r9d

  if ( a3 != a2 )
  {
    v3 = (const void *)(a1 + 96);
    v5 = a2;
    while ( 1 )
    {
      v12 = sub_1648700(v5);
      v13 = v12;
      v14 = *(_BYTE *)(a1 + 8) & 1;
      if ( v14 )
      {
        v7 = a1 + 16;
        v8 = 7;
      }
      else
      {
        v15 = *(_DWORD *)(a1 + 24);
        v7 = *(_QWORD *)(a1 + 16);
        if ( !v15 )
        {
          v16 = *(_DWORD *)(a1 + 8);
          ++*(_QWORD *)a1;
          v17 = 0;
          v18 = (v16 >> 1) + 1;
          goto LABEL_10;
        }
        v8 = v15 - 1;
      }
      v9 = v8 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
      v10 = (_QWORD *)(v7 + 8LL * (v8 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4))));
      v11 = (_QWORD *)*v10;
      if ( v13 == (_QWORD *)*v10 )
      {
LABEL_5:
        v5 = *(_QWORD *)(v5 + 8);
        if ( a3 == v5 )
          return;
      }
      else
      {
        v23 = 1;
        v17 = 0;
        while ( v11 != (_QWORD *)-8LL )
        {
          if ( v17 || v11 != (_QWORD *)-16LL )
            v10 = v17;
          v9 = v8 & (v23 + v9);
          v11 = *(_QWORD **)(v7 + 8LL * v9);
          if ( v13 == v11 )
            goto LABEL_5;
          ++v23;
          v17 = v10;
          v10 = (_QWORD *)(v7 + 8LL * v9);
        }
        v16 = *(_DWORD *)(a1 + 8);
        if ( !v17 )
          v17 = v10;
        ++*(_QWORD *)a1;
        v18 = (v16 >> 1) + 1;
        if ( !v14 )
        {
          v15 = *(_DWORD *)(a1 + 24);
LABEL_10:
          if ( 4 * v18 >= 3 * v15 )
            goto LABEL_24;
          goto LABEL_11;
        }
        v15 = 8;
        if ( (unsigned int)(4 * v18) >= 0x18 )
        {
LABEL_24:
          sub_1962FE0(a1, 2 * v15);
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v25 = a1 + 16;
            v26 = 7;
          }
          else
          {
            v24 = *(_DWORD *)(a1 + 24);
            v25 = *(_QWORD *)(a1 + 16);
            if ( !v24 )
              goto LABEL_58;
            v26 = v24 - 1;
          }
          v27 = v26 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v17 = (_QWORD *)(v25 + 8LL * v27);
          v28 = *v17;
          if ( v13 == (_QWORD *)*v17 )
            goto LABEL_28;
          v36 = 1;
          v35 = 0;
          while ( v28 != -8 )
          {
            if ( v28 == -16 && !v35 )
              v35 = v17;
            v27 = v26 & (v36 + v27);
            v17 = (_QWORD *)(v25 + 8LL * v27);
            v28 = *v17;
            if ( v13 == (_QWORD *)*v17 )
              goto LABEL_28;
            ++v36;
          }
LABEL_35:
          if ( v35 )
            v17 = v35;
LABEL_28:
          v16 = *(_DWORD *)(a1 + 8);
          goto LABEL_12;
        }
LABEL_11:
        if ( v15 - *(_DWORD *)(a1 + 12) - v18 <= v15 >> 3 )
        {
          sub_1962FE0(a1, v15);
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v30 = a1 + 16;
            v31 = 7;
          }
          else
          {
            v29 = *(_DWORD *)(a1 + 24);
            v30 = *(_QWORD *)(a1 + 16);
            if ( !v29 )
            {
LABEL_58:
              *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
              BUG();
            }
            v31 = v29 - 1;
          }
          v32 = v31 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
          v17 = (_QWORD *)(v30 + 8LL * v32);
          v33 = *v17;
          if ( v13 == (_QWORD *)*v17 )
            goto LABEL_28;
          v34 = 1;
          v35 = 0;
          while ( v33 != -8 )
          {
            if ( v33 == -16 && !v35 )
              v35 = v17;
            v32 = v31 & (v34 + v32);
            v17 = (_QWORD *)(v30 + 8LL * v32);
            v33 = *v17;
            if ( v13 == (_QWORD *)*v17 )
              goto LABEL_28;
            ++v34;
          }
          goto LABEL_35;
        }
LABEL_12:
        *(_DWORD *)(a1 + 8) = (2 * (v16 >> 1) + 2) | v16 & 1;
        if ( *v17 != -8 )
          --*(_DWORD *)(a1 + 12);
        *v17 = v13;
        v21 = sub_1648700(v5);
        v22 = *(unsigned int *)(a1 + 88);
        if ( (unsigned int)v22 >= *(_DWORD *)(a1 + 92) )
        {
          sub_16CD150(a1 + 80, v3, 0, 8, v19, v20);
          v22 = *(unsigned int *)(a1 + 88);
        }
        *(_QWORD *)(*(_QWORD *)(a1 + 80) + 8 * v22) = v21;
        ++*(_DWORD *)(a1 + 88);
        v5 = *(_QWORD *)(v5 + 8);
        if ( a3 == v5 )
          return;
      }
    }
  }
}
