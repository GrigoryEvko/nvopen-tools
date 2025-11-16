// Function: sub_2508DC0
// Address: 0x2508dc0
//
__int64 __fastcall sub_2508DC0(__int64 a1, _QWORD *a2)
{
  __int64 v3; // rdi
  unsigned __int8 v4; // al
  __int64 v5; // rax
  __int64 v6; // rcx
  int v7; // eax
  int v8; // edx
  unsigned int v9; // eax
  __int64 v10; // rsi
  int v12; // r8d

  v3 = *a2 & 0xFFFFFFFFFFFFFFFCLL;
  if ( (*a2 & 3LL) == 3 )
    v3 = *(_QWORD *)(v3 + 24);
  v4 = *(_BYTE *)v3;
  if ( *(_BYTE *)v3 )
  {
    if ( v4 == 22 )
    {
      v3 = *(_QWORD *)(v3 + 24);
    }
    else if ( v4 <= 0x1Cu )
    {
      v3 = 0;
    }
    else
    {
      v3 = sub_B43CB0(v3);
    }
  }
  v5 = *(_QWORD *)(a1 + 200);
  if ( !*(_DWORD *)(v5 + 40) )
    return 1;
  v6 = *(_QWORD *)(v5 + 8);
  v7 = *(_DWORD *)(v5 + 24);
  if ( v7 )
  {
    v8 = v7 - 1;
    v9 = (v7 - 1) & (((unsigned int)v3 >> 9) ^ ((unsigned int)v3 >> 4));
    v10 = *(_QWORD *)(v6 + 8LL * v9);
    if ( v3 == v10 )
      return 1;
    v12 = 1;
    while ( v10 != -4096 )
    {
      v9 = v8 & (v12 + v9);
      v10 = *(_QWORD *)(v6 + 8LL * v9);
      if ( v3 == v10 )
        return 1;
      ++v12;
    }
  }
  return 0;
}
