// Function: sub_278B930
// Address: 0x278b930
//
__int64 __fastcall sub_278B930(__int64 a1, int a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rdi
  __int64 v5; // rcx
  unsigned int v7; // esi
  int *v8; // rax
  int v9; // r9d
  _DWORD *v10; // rax
  int v12; // eax
  int v13; // r11d

  v4 = *(_QWORD *)(a4 + 360);
  v5 = *(unsigned int *)(a4 + 376);
  if ( !(_DWORD)v5 )
    return 1;
  v7 = (v5 - 1) & (37 * a2);
  v8 = (int *)(v4 + 40LL * v7);
  v9 = *v8;
  if ( a2 != *v8 )
  {
    v12 = 1;
    while ( v9 != -1 )
    {
      v13 = v12 + 1;
      v7 = (v5 - 1) & (v12 + v7);
      v8 = (int *)(v4 + 40LL * v7);
      v9 = *v8;
      if ( a2 == *v8 )
        goto LABEL_3;
      v12 = v13;
    }
    return 1;
  }
LABEL_3:
  if ( v8 == (int *)(v4 + 40 * v5) )
    return 1;
  v10 = v8 + 2;
  while ( a3 == *((_QWORD *)v10 + 1) )
  {
    v10 = (_DWORD *)*((_QWORD *)v10 + 3);
    if ( !v10 )
      return 1;
  }
  return 0;
}
