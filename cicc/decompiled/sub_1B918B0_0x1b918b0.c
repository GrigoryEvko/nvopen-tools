// Function: sub_1B918B0
// Address: 0x1b918b0
//
__int64 __fastcall sub_1B918B0(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // rax
  __int64 v4; // r8
  unsigned int v5; // edi
  int *v6; // rcx
  int v7; // r9d
  int v8; // eax
  unsigned int v9; // r8d
  __int64 v10; // rdi
  int v11; // edx
  unsigned int v12; // eax
  __int64 v13; // rcx
  int v15; // ecx
  int i; // r8d
  int v17; // r11d

  v3 = *(unsigned int *)(a1 + 160);
  v4 = *(_QWORD *)(a1 + 144);
  if ( !(_DWORD)v3 )
  {
LABEL_8:
    v6 = (int *)(v4 + 40 * v3);
    goto LABEL_3;
  }
  v5 = (v3 - 1) & (37 * a3);
  v6 = (int *)(v4 + 40LL * v5);
  v7 = *v6;
  if ( *v6 != a3 )
  {
    v15 = 1;
    while ( v7 != -1 )
    {
      v17 = v15 + 1;
      v5 = (v3 - 1) & (v15 + v5);
      v6 = (int *)(v4 + 40LL * v5);
      v7 = *v6;
      if ( *v6 == a3 )
        goto LABEL_3;
      v15 = v17;
    }
    goto LABEL_8;
  }
LABEL_3:
  v8 = v6[8];
  v9 = 0;
  if ( v8 )
  {
    v10 = *((_QWORD *)v6 + 2);
    v11 = v8 - 1;
    v9 = 1;
    v12 = (v8 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v13 = *(_QWORD *)(v10 + 16LL * v12);
    if ( a2 != v13 )
    {
      for ( i = 1; ; ++i )
      {
        if ( v13 == -8 )
          return 0;
        v12 = v11 & (i + v12);
        v13 = *(_QWORD *)(v10 + 16LL * v12);
        if ( a2 == v13 )
          break;
      }
      return 1;
    }
  }
  return v9;
}
