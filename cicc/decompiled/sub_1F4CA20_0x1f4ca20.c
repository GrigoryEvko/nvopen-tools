// Function: sub_1F4CA20
// Address: 0x1f4ca20
//
__int64 __fastcall sub_1F4CA20(int a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // r9
  int v4; // r10d
  int v5; // esi
  int *v6; // r8
  unsigned int v7; // edx
  int *v8; // rax
  int v9; // ecx
  int v10; // eax
  int v12; // r11d

  if ( a1 >= 0 )
    return (unsigned int)a1;
  v2 = *(unsigned int *)(a2 + 24);
  v3 = *(_QWORD *)(a2 + 8);
  v4 = v2 - 1;
  v5 = *(_DWORD *)(a2 + 24);
  v6 = (int *)(v3 + 8 * v2);
  while ( v5 )
  {
    v7 = v4 & (37 * a1);
    v8 = (int *)(v3 + 8LL * v7);
    v9 = *v8;
    if ( a1 != *v8 )
    {
      v10 = 1;
      while ( v9 != -1 )
      {
        v12 = v10 + 1;
        v7 = v4 & (v10 + v7);
        v8 = (int *)(v3 + 8LL * v7);
        v9 = *v8;
        if ( *v8 == a1 )
          goto LABEL_4;
        v10 = v12;
      }
      return 0;
    }
LABEL_4:
    if ( v6 == v8 )
      return 0;
    a1 = v8[1];
    if ( a1 >= 0 )
      return (unsigned int)a1;
  }
  return 0;
}
