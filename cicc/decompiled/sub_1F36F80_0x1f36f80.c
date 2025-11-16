// Function: sub_1F36F80
// Address: 0x1f36f80
//
void __fastcall sub_1F36F80(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4, __int64 a5, __int64 *a6)
{
  const void *v7; // r13
  __int64 *v8; // r12
  __int64 v10; // rdi
  int v11; // esi
  unsigned int v12; // eax
  __int64 *v13; // r10
  char v14; // cl
  unsigned int v15; // esi
  unsigned int v16; // eax
  __int64 *v17; // r8
  int v18; // edx
  __int64 v19; // rax
  __int64 v20; // rcx
  int v21; // r11d
  int v22; // eax
  __int64 v23; // rsi
  int v24; // edx
  unsigned int v25; // eax
  __int64 v26; // rdi
  int v27; // edx
  __int64 v28; // rsi
  int v29; // edx
  unsigned int v30; // eax
  __int64 v31; // rdi
  int v32; // r10d
  int v33; // r10d

  if ( a3 != a2 )
  {
    v7 = (const void *)(a1 + 96);
    v8 = a2;
    while ( 1 )
    {
      v14 = *(_BYTE *)(a1 + 8) & 1;
      if ( v14 )
      {
        v10 = a1 + 16;
        v11 = 7;
      }
      else
      {
        v15 = *(_DWORD *)(a1 + 24);
        v10 = *(_QWORD *)(a1 + 16);
        if ( !v15 )
        {
          v16 = *(_DWORD *)(a1 + 8);
          ++*(_QWORD *)a1;
          v17 = 0;
          v18 = (v16 >> 1) + 1;
          goto LABEL_10;
        }
        v11 = v15 - 1;
      }
      v12 = v11 & (((unsigned int)*v8 >> 9) ^ ((unsigned int)*v8 >> 4));
      v13 = (__int64 *)(v10 + 8LL * v12);
      a6 = (__int64 *)*v13;
      if ( *v8 == *v13 )
      {
LABEL_5:
        if ( a3 == ++v8 )
          return;
      }
      else
      {
        v21 = 1;
        v17 = 0;
        while ( a6 != (__int64 *)-8LL )
        {
          if ( v17 || a6 != (__int64 *)-16LL )
            v13 = v17;
          v12 = v11 & (v21 + v12);
          a6 = *(__int64 **)(v10 + 8LL * v12);
          if ( (__int64 *)*v8 == a6 )
            goto LABEL_5;
          ++v21;
          v17 = v13;
          v13 = (__int64 *)(v10 + 8LL * v12);
        }
        v16 = *(_DWORD *)(a1 + 8);
        if ( !v17 )
          v17 = v13;
        ++*(_QWORD *)a1;
        v18 = (v16 >> 1) + 1;
        if ( !v14 )
        {
          v15 = *(_DWORD *)(a1 + 24);
LABEL_10:
          if ( 3 * v15 <= 4 * v18 )
            goto LABEL_24;
          goto LABEL_11;
        }
        v15 = 8;
        if ( (unsigned int)(4 * v18) >= 0x18 )
        {
LABEL_24:
          sub_1F36BD0(a1, 2 * v15);
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v23 = a1 + 16;
            v24 = 7;
          }
          else
          {
            v22 = *(_DWORD *)(a1 + 24);
            v23 = *(_QWORD *)(a1 + 16);
            if ( !v22 )
              goto LABEL_58;
            v24 = v22 - 1;
          }
          v25 = v24 & (((unsigned int)*v8 >> 9) ^ ((unsigned int)*v8 >> 4));
          v17 = (__int64 *)(v23 + 8LL * v25);
          v26 = *v17;
          if ( *v8 == *v17 )
            goto LABEL_28;
          v33 = 1;
          a6 = 0;
          while ( v26 != -8 )
          {
            if ( v26 == -16 && !a6 )
              a6 = v17;
            v25 = v24 & (v33 + v25);
            v17 = (__int64 *)(v23 + 8LL * v25);
            v26 = *v17;
            if ( *v8 == *v17 )
              goto LABEL_28;
            ++v33;
          }
LABEL_35:
          if ( a6 )
            v17 = a6;
LABEL_28:
          v16 = *(_DWORD *)(a1 + 8);
          goto LABEL_12;
        }
LABEL_11:
        if ( v15 - *(_DWORD *)(a1 + 12) - v18 <= v15 >> 3 )
        {
          sub_1F36BD0(a1, v15);
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v28 = a1 + 16;
            v29 = 7;
          }
          else
          {
            v27 = *(_DWORD *)(a1 + 24);
            v28 = *(_QWORD *)(a1 + 16);
            if ( !v27 )
            {
LABEL_58:
              *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
              BUG();
            }
            v29 = v27 - 1;
          }
          v30 = v29 & (((unsigned int)*v8 >> 9) ^ ((unsigned int)*v8 >> 4));
          v17 = (__int64 *)(v28 + 8LL * v30);
          v31 = *v17;
          if ( *v17 == *v8 )
            goto LABEL_28;
          v32 = 1;
          a6 = 0;
          while ( v31 != -8 )
          {
            if ( v31 == -16 && !a6 )
              a6 = v17;
            v30 = v29 & (v32 + v30);
            v17 = (__int64 *)(v28 + 8LL * v30);
            v31 = *v17;
            if ( *v8 == *v17 )
              goto LABEL_28;
            ++v32;
          }
          goto LABEL_35;
        }
LABEL_12:
        *(_DWORD *)(a1 + 8) = (2 * (v16 >> 1) + 2) | v16 & 1;
        if ( *v17 != -8 )
          --*(_DWORD *)(a1 + 12);
        *v17 = *v8;
        v19 = *(unsigned int *)(a1 + 88);
        if ( (unsigned int)v19 >= *(_DWORD *)(a1 + 92) )
        {
          sub_16CD150(a1 + 80, v7, 0, 8, (int)v17, (int)a6);
          v19 = *(unsigned int *)(a1 + 88);
        }
        v20 = *v8++;
        *(_QWORD *)(*(_QWORD *)(a1 + 80) + 8 * v19) = v20;
        ++*(_DWORD *)(a1 + 88);
        if ( a3 == v8 )
          return;
      }
    }
  }
}
