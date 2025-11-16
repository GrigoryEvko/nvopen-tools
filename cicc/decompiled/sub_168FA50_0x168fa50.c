// Function: sub_168FA50
// Address: 0x168fa50
//
__int64 __fastcall sub_168FA50(__int64 a1, _DWORD *a2, __int64 a3)
{
  _DWORD *v3; // r10
  __int64 v4; // r8
  unsigned int v5; // ecx
  __int64 v6; // r11
  __int64 v7; // r9
  int v8; // r12d
  unsigned int v9; // edi
  int *v10; // rax
  int v11; // r13d
  unsigned int v12; // edx
  unsigned int v13; // eax
  int v15; // eax
  int v16; // r14d

  v3 = &a2[a3];
  if ( a2 == v3 )
    return 0;
  v4 = 0;
  v5 = -1;
  v6 = *(_QWORD *)(a1 + 160);
  v7 = *(unsigned int *)(a1 + 176);
  v8 = v7 - 1;
  do
  {
    if ( (_DWORD)v7 )
    {
      v9 = v8 & (37 * *a2);
      v10 = (int *)(v6 + 12LL * v9);
      v11 = *v10;
      if ( *v10 == *a2 )
      {
LABEL_5:
        if ( (int *)(v6 + 12 * v7) != v10 )
        {
          v12 = v10[1];
          v13 = v10[2];
          if ( v5 > v12 )
            v5 = v12;
          if ( (unsigned int)v4 < v13 )
            v4 = v13;
        }
      }
      else
      {
        v15 = 1;
        while ( v11 != -1 )
        {
          v16 = v15 + 1;
          v9 = v8 & (v15 + v9);
          v10 = (int *)(v6 + 12LL * v9);
          v11 = *v10;
          if ( *a2 == *v10 )
            goto LABEL_5;
          v15 = v16;
        }
      }
    }
    ++a2;
  }
  while ( a2 != v3 );
  if ( v5 == -1 )
    v5 = 0;
  return (v4 << 32) | v5;
}
