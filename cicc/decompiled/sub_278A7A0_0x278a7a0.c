// Function: sub_278A7A0
// Address: 0x278a7a0
//
void __fastcall sub_278A7A0(__int64 a1, _BYTE *a2)
{
  int v2; // ecx
  __int64 v3; // r8
  int v4; // r10d
  unsigned int v5; // edx
  __int64 v6; // rax
  _BYTE *v7; // r9
  int v8; // eax
  __int64 v9; // r8
  int v10; // esi
  unsigned int v11; // edx
  int *v12; // rax
  int v13; // r9d
  _BYTE *v14; // rax
  unsigned int v15; // ecx
  int i; // r11d
  int v17; // eax
  int v18; // r11d
  int v19; // eax
  int v20; // r10d
  int v21; // ebx
  __int64 v22; // r11

  v2 = *(_DWORD *)(a1 + 24);
  v3 = *(_QWORD *)(a1 + 8);
  if ( v2 )
  {
    v4 = v2 - 1;
    v5 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v6 = v3 + 16LL * v5;
    v7 = *(_BYTE **)v6;
    if ( *(_BYTE **)v6 == a2 )
    {
      v2 = *(_DWORD *)(v6 + 8);
LABEL_4:
      *(_QWORD *)v6 = -8192;
      --*(_DWORD *)(a1 + 16);
      ++*(_DWORD *)(a1 + 20);
    }
    else
    {
      v14 = *(_BYTE **)v6;
      v15 = (v2 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      for ( i = 1; ; i = v21 )
      {
        if ( v14 == (_BYTE *)-4096LL )
        {
          v2 = 0;
          goto LABEL_13;
        }
        v21 = i + 1;
        v15 = v4 & (i + v15);
        v22 = v3 + 16LL * v15;
        v14 = *(_BYTE **)v22;
        if ( *(_BYTE **)v22 == a2 )
          break;
      }
      v2 = *(_DWORD *)(v22 + 8);
LABEL_13:
      v17 = 1;
      while ( v7 != (_BYTE *)-4096LL )
      {
        v18 = v17 + 1;
        v5 = v4 & (v17 + v5);
        v6 = v3 + 16LL * v5;
        v7 = *(_BYTE **)v6;
        if ( *(_BYTE **)v6 == a2 )
          goto LABEL_4;
        v17 = v18;
      }
    }
  }
  if ( *a2 == 84 )
  {
    v8 = *(_DWORD *)(a1 + 144);
    v9 = *(_QWORD *)(a1 + 128);
    if ( v8 )
    {
      v10 = v8 - 1;
      v11 = (v8 - 1) & (37 * v2);
      v12 = (int *)(v9 + 16LL * v11);
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
          v11 = v10 & (v19 + v11);
          v12 = (int *)(v9 + 16LL * v11);
          v13 = *v12;
          if ( v2 == *v12 )
            goto LABEL_9;
          v19 = v20;
        }
      }
    }
  }
}
