// Function: sub_22F3490
// Address: 0x22f3490
//
__int64 __fastcall sub_22F3490(__int64 a1, int *a2, __int64 a3)
{
  int *v3; // r10
  __int64 v4; // r8
  __int64 v5; // rax
  __int64 v6; // r11
  unsigned int v7; // edi
  int v8; // r9d
  int *v9; // r12
  int v10; // ebx
  int v11; // eax
  unsigned int v12; // ecx
  int *v13; // rdx
  int v14; // r13d
  unsigned int v15; // eax
  int v17; // edx
  int v18; // r14d

  v3 = &a2[a3];
  if ( a2 == v3 )
    return 0;
  v4 = 0;
  v5 = *(unsigned int *)(a1 + 176);
  v6 = *(_QWORD *)(a1 + 160);
  v7 = -1;
  v8 = v5;
  v9 = (int *)(v6 + 12 * v5);
  v10 = v5 - 1;
  do
  {
    v11 = *a2;
    if ( v8 )
    {
      v12 = v10 & (37 * v11);
      v13 = (int *)(v6 + 12LL * v12);
      v14 = *v13;
      if ( v11 == *v13 )
      {
LABEL_5:
        if ( v9 != v13 )
        {
          if ( v7 > v13[1] )
            v7 = v13[1];
          v15 = v13[2];
          if ( (unsigned int)v4 < v15 )
            v4 = v15;
        }
      }
      else
      {
        v17 = 1;
        while ( v14 != -1 )
        {
          v18 = v17 + 1;
          v12 = v10 & (v17 + v12);
          v13 = (int *)(v6 + 12LL * v12);
          v14 = *v13;
          if ( v11 == *v13 )
            goto LABEL_5;
          v17 = v18;
        }
      }
    }
    ++a2;
  }
  while ( a2 != v3 );
  if ( v7 == -1 )
    v7 = 0;
  return (v4 << 32) | v7;
}
