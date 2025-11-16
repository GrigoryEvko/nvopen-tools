// Function: sub_190ACD0
// Address: 0x190acd0
//
void __fastcall sub_190ACD0(__int64 a1, __int64 a2)
{
  int v2; // ecx
  int v3; // r10d
  __int64 v4; // r9
  unsigned int v5; // edx
  __int64 *v6; // rax
  __int64 v7; // r8
  int v8; // eax
  int v9; // esi
  __int64 v10; // r9
  unsigned int v11; // edx
  int *v12; // rax
  int v13; // r8d
  __int64 v14; // rax
  unsigned int v15; // ecx
  int i; // r11d
  int v17; // eax
  int v18; // r11d
  int v19; // eax
  int v20; // r10d
  int v21; // ebx
  __int64 *v22; // r11

  v2 = *(_DWORD *)(a1 + 24);
  if ( v2 )
  {
    v3 = v2 - 1;
    v4 = *(_QWORD *)(a1 + 8);
    v5 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = (__int64 *)(v4 + 16LL * v5);
    v7 = *v6;
    if ( *v6 == a2 )
    {
      v2 = *((_DWORD *)v6 + 2);
LABEL_4:
      *v6 = -16;
      --*(_DWORD *)(a1 + 16);
      ++*(_DWORD *)(a1 + 20);
    }
    else
    {
      v14 = *v6;
      v15 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      for ( i = 1; ; i = v21 )
      {
        if ( v14 == -8 )
        {
          v2 = 0;
          goto LABEL_13;
        }
        v21 = i + 1;
        v15 = v3 & (i + v15);
        v22 = (__int64 *)(v4 + 16LL * v15);
        v14 = *v22;
        if ( *v22 == a2 )
          break;
      }
      v2 = *((_DWORD *)v22 + 2);
LABEL_13:
      v17 = 1;
      while ( v7 != -8 )
      {
        v18 = v17 + 1;
        v5 = v3 & (v17 + v5);
        v6 = (__int64 *)(v4 + 16LL * v5);
        v7 = *v6;
        if ( *v6 == a2 )
          goto LABEL_4;
        v17 = v18;
      }
    }
  }
  if ( *(_BYTE *)(a2 + 16) == 77 )
  {
    v8 = *(_DWORD *)(a1 + 144);
    if ( v8 )
    {
      v9 = v8 - 1;
      v10 = *(_QWORD *)(a1 + 128);
      v11 = (v8 - 1) & (37 * v2);
      v12 = (int *)(v10 + 16LL * v11);
      v13 = *v12;
      if ( v2 == *v12 )
      {
LABEL_9:
        *v12 = -2;
        --*(_DWORD *)(a1 + 136);
        ++*(_DWORD *)(a1 + 140);
      }
      else
      {
        v19 = 1;
        while ( v13 != -1 )
        {
          v20 = v19 + 1;
          v11 = v9 & (v19 + v11);
          v12 = (int *)(v10 + 16LL * v11);
          v13 = *v12;
          if ( v2 == *v12 )
            goto LABEL_9;
          v19 = v20;
        }
      }
    }
  }
}
