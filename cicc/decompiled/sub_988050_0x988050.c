// Function: sub_988050
// Address: 0x988050
//
__int64 __fastcall sub_988050(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rsi
  unsigned int v5; // edx
  __int64 v6; // rcx
  __int64 v7; // rdi
  int v9; // ecx
  int v10; // r9d

  if ( *(_BYTE *)(a1 + 192) )
  {
    v3 = *(unsigned int *)(a1 + 184);
    if ( !(_DWORD)v3 )
      return 0;
  }
  else
  {
    sub_CFDFC0();
    v3 = *(unsigned int *)(a1 + 184);
    if ( !(_DWORD)v3 )
      return 0;
  }
  v4 = *(_QWORD *)(a1 + 168);
  v5 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v6 = v4 + 88LL * v5;
  v7 = *(_QWORD *)(v6 + 24);
  if ( a2 == v7 )
  {
LABEL_4:
    if ( v6 != v4 + 88 * v3 )
      return *(_QWORD *)(v6 + 40);
  }
  else
  {
    v9 = 1;
    while ( v7 != -4096 )
    {
      v10 = v9 + 1;
      v5 = (v3 - 1) & (v9 + v5);
      v6 = v4 + 88LL * v5;
      v7 = *(_QWORD *)(v6 + 24);
      if ( a2 == v7 )
        goto LABEL_4;
      v9 = v10;
    }
  }
  return 0;
}
