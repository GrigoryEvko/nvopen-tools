// Function: sub_2BB32D0
// Address: 0x2bb32d0
//
void __fastcall sub_2BB32D0(__int64 a1, int *a2, int *a3)
{
  int *v4; // r12
  __int64 v6; // r10
  int v7; // esi
  unsigned int v8; // edx
  _DWORD *v9; // rdi
  int v10; // ecx
  char v11; // r8
  unsigned int v12; // esi
  unsigned int v13; // eax
  _DWORD *v14; // r9
  int v15; // edx
  int v16; // eax
  int v17; // r11d
  int v18; // eax
  __int64 v19; // rsi
  int v20; // edx
  unsigned int v21; // eax
  int v22; // edi
  int v23; // edx
  __int64 v24; // rsi
  int v25; // edx
  unsigned int v26; // eax
  int v27; // edi
  int v28; // r10d
  _DWORD *v29; // r8
  int v30; // r10d

  if ( a2 != a3 )
  {
    v4 = a2;
    while ( 1 )
    {
      v11 = *(_BYTE *)(a1 + 8) & 1;
      if ( v11 )
      {
        v6 = a1 + 16;
        v7 = 7;
      }
      else
      {
        v12 = *(_DWORD *)(a1 + 24);
        v6 = *(_QWORD *)(a1 + 16);
        if ( !v12 )
        {
          v13 = *(_DWORD *)(a1 + 8);
          ++*(_QWORD *)a1;
          v14 = 0;
          v15 = (v13 >> 1) + 1;
          goto LABEL_10;
        }
        v7 = v12 - 1;
      }
      v8 = v7 & (37 * *v4);
      v9 = (_DWORD *)(v6 + 4LL * v8);
      v10 = *v9;
      if ( *v4 == *v9 )
      {
LABEL_5:
        if ( a3 == ++v4 )
          return;
      }
      else
      {
        v17 = 1;
        v14 = 0;
        while ( v10 != -1 )
        {
          if ( v10 != -2 || v14 )
            v9 = v14;
          v8 = v7 & (v17 + v8);
          v10 = *(_DWORD *)(v6 + 4LL * v8);
          if ( *v4 == v10 )
            goto LABEL_5;
          ++v17;
          v14 = v9;
          v9 = (_DWORD *)(v6 + 4LL * v8);
        }
        v13 = *(_DWORD *)(a1 + 8);
        if ( !v14 )
          v14 = v9;
        ++*(_QWORD *)a1;
        v15 = (v13 >> 1) + 1;
        if ( !v11 )
        {
          v12 = *(_DWORD *)(a1 + 24);
LABEL_10:
          if ( 4 * v15 >= 3 * v12 )
            goto LABEL_22;
          goto LABEL_11;
        }
        v12 = 8;
        if ( (unsigned int)(4 * v15) >= 0x18 )
        {
LABEL_22:
          sub_2BB2EE0(a1, 2 * v12);
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v19 = a1 + 16;
            v20 = 7;
          }
          else
          {
            v18 = *(_DWORD *)(a1 + 24);
            v19 = *(_QWORD *)(a1 + 16);
            if ( !v18 )
              goto LABEL_56;
            v20 = v18 - 1;
          }
          v21 = v20 & (37 * *v4);
          v14 = (_DWORD *)(v19 + 4LL * v21);
          v22 = *v14;
          if ( *v4 == *v14 )
            goto LABEL_26;
          v30 = 1;
          v29 = 0;
          while ( v22 != -1 )
          {
            if ( v22 == -2 && !v29 )
              v29 = v14;
            v21 = v20 & (v30 + v21);
            v14 = (_DWORD *)(v19 + 4LL * v21);
            v22 = *v14;
            if ( *v4 == *v14 )
              goto LABEL_26;
            ++v30;
          }
LABEL_33:
          if ( v29 )
            v14 = v29;
LABEL_26:
          v13 = *(_DWORD *)(a1 + 8);
          goto LABEL_12;
        }
LABEL_11:
        if ( v12 - *(_DWORD *)(a1 + 12) - v15 <= v12 >> 3 )
        {
          sub_2BB2EE0(a1, v12);
          if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
          {
            v24 = a1 + 16;
            v25 = 7;
          }
          else
          {
            v23 = *(_DWORD *)(a1 + 24);
            v24 = *(_QWORD *)(a1 + 16);
            if ( !v23 )
            {
LABEL_56:
              *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
              BUG();
            }
            v25 = v23 - 1;
          }
          v26 = v25 & (37 * *v4);
          v14 = (_DWORD *)(v24 + 4LL * v26);
          v27 = *v14;
          if ( *v14 == *v4 )
            goto LABEL_26;
          v28 = 1;
          v29 = 0;
          while ( v27 != -1 )
          {
            if ( !v29 && v27 == -2 )
              v29 = v14;
            v26 = v25 & (v28 + v26);
            v14 = (_DWORD *)(v24 + 4LL * v26);
            v27 = *v14;
            if ( *v4 == *v14 )
              goto LABEL_26;
            ++v28;
          }
          goto LABEL_33;
        }
LABEL_12:
        *(_DWORD *)(a1 + 8) = (2 * (v13 >> 1) + 2) | v13 & 1;
        if ( *v14 != -1 )
          --*(_DWORD *)(a1 + 12);
        v16 = *v4++;
        *v14 = v16;
        if ( a3 == v4 )
          return;
      }
    }
  }
}
