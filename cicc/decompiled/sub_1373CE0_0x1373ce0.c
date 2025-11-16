// Function: sub_1373CE0
// Address: 0x1373ce0
//
void __fastcall sub_1373CE0(__int64 a1)
{
  int *v1; // r13
  int *v2; // r12
  __int64 v4; // r14
  __int64 v5; // r8
  int v6; // esi
  unsigned int v7; // ecx
  int *v8; // rax
  int v9; // r9d
  char v10; // di
  unsigned int v11; // esi
  unsigned int v12; // edx
  int v13; // ecx
  unsigned int v14; // r8d
  int v15; // edx
  int v16; // r11d
  int *v17; // r10
  int v18; // eax
  __int64 v19; // r8
  int v20; // ecx
  unsigned int v21; // edx
  int v22; // edi
  int v23; // ecx
  __int64 v24; // r8
  int v25; // ecx
  unsigned int v26; // edi
  int v27; // esi
  int v28; // r10d
  int *v29; // r9
  int v30; // r10d

  v1 = *(int **)(a1 + 32);
  v2 = *(int **)(a1 + 24);
  if ( v2 != v1 )
  {
    v4 = a1 + 48;
    while ( 1 )
    {
      v10 = *(_BYTE *)(a1 + 56) & 1;
      if ( v10 )
      {
        v5 = a1 + 64;
        v6 = 3;
      }
      else
      {
        v11 = *(_DWORD *)(a1 + 72);
        v5 = *(_QWORD *)(a1 + 64);
        if ( !v11 )
        {
          v12 = *(_DWORD *)(a1 + 56);
          ++*(_QWORD *)(a1 + 48);
          v8 = 0;
          v13 = (v12 >> 1) + 1;
LABEL_10:
          v14 = 3 * v11;
          goto LABEL_11;
        }
        v6 = v11 - 1;
      }
      v7 = v6 & (37 * *v2);
      v8 = (int *)(v5 + 16LL * v7);
      v9 = *v8;
      if ( *v2 == *v8 )
      {
LABEL_5:
        *((_QWORD *)v8 + 1) = v2;
        v2 += 22;
        if ( v1 == v2 )
          return;
      }
      else
      {
        v16 = 1;
        v17 = 0;
        while ( v9 != -1 )
        {
          if ( !v17 && v9 == -2 )
            v17 = v8;
          v7 = v6 & (v16 + v7);
          v8 = (int *)(v5 + 16LL * v7);
          v9 = *v8;
          if ( *v2 == *v8 )
            goto LABEL_5;
          ++v16;
        }
        v12 = *(_DWORD *)(a1 + 56);
        v14 = 12;
        v11 = 4;
        if ( v17 )
          v8 = v17;
        ++*(_QWORD *)(a1 + 48);
        v13 = (v12 >> 1) + 1;
        if ( !v10 )
        {
          v11 = *(_DWORD *)(a1 + 72);
          goto LABEL_10;
        }
LABEL_11:
        if ( v14 <= 4 * v13 )
        {
          sub_136EE70(v4, 2 * v11);
          if ( (*(_BYTE *)(a1 + 56) & 1) != 0 )
          {
            v19 = a1 + 64;
            v20 = 3;
          }
          else
          {
            v18 = *(_DWORD *)(a1 + 72);
            v19 = *(_QWORD *)(a1 + 64);
            if ( !v18 )
              goto LABEL_55;
            v20 = v18 - 1;
          }
          v21 = v20 & (37 * *v2);
          v8 = (int *)(v19 + 16LL * v21);
          v22 = *v8;
          if ( *v8 != *v2 )
          {
            v30 = 1;
            v29 = 0;
            while ( v22 != -1 )
            {
              if ( !v29 && v22 == -2 )
                v29 = v8;
              v21 = v20 & (v30 + v21);
              v8 = (int *)(v19 + 16LL * v21);
              v22 = *v8;
              if ( *v2 == *v8 )
                goto LABEL_27;
              ++v30;
            }
LABEL_34:
            if ( v29 )
              v8 = v29;
          }
LABEL_27:
          v12 = *(_DWORD *)(a1 + 56);
          goto LABEL_13;
        }
        if ( v11 - *(_DWORD *)(a1 + 60) - v13 <= v11 >> 3 )
        {
          sub_136EE70(v4, v11);
          if ( (*(_BYTE *)(a1 + 56) & 1) != 0 )
          {
            v24 = a1 + 64;
            v25 = 3;
          }
          else
          {
            v23 = *(_DWORD *)(a1 + 72);
            v24 = *(_QWORD *)(a1 + 64);
            if ( !v23 )
            {
LABEL_55:
              *(_DWORD *)(a1 + 56) = (2 * (*(_DWORD *)(a1 + 56) >> 1) + 2) | *(_DWORD *)(a1 + 56) & 1;
              BUG();
            }
            v25 = v23 - 1;
          }
          v26 = v25 & (37 * *v2);
          v8 = (int *)(v24 + 16LL * v26);
          v27 = *v8;
          if ( *v8 != *v2 )
          {
            v28 = 1;
            v29 = 0;
            while ( v27 != -1 )
            {
              if ( v27 == -2 && !v29 )
                v29 = v8;
              v26 = v25 & (v28 + v26);
              v8 = (int *)(v24 + 16LL * v26);
              v27 = *v8;
              if ( *v2 == *v8 )
                goto LABEL_27;
              ++v28;
            }
            goto LABEL_34;
          }
          goto LABEL_27;
        }
LABEL_13:
        *(_DWORD *)(a1 + 56) = (2 * (v12 >> 1) + 2) | v12 & 1;
        if ( *v8 != -1 )
          --*(_DWORD *)(a1 + 60);
        v15 = *v2;
        *((_QWORD *)v8 + 1) = 0;
        *((_QWORD *)v8 + 1) = v2;
        v2 += 22;
        *v8 = v15;
        if ( v1 == v2 )
          return;
      }
    }
  }
}
