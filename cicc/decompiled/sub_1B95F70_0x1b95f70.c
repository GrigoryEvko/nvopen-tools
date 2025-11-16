// Function: sub_1B95F70
// Address: 0x1b95f70
//
char __fastcall sub_1B95F70(__int64 a1, __int64 a2, int a3)
{
  __int64 v3; // rax
  __int64 v4; // r8
  unsigned int v5; // ecx
  int *v6; // rdi
  int v7; // r9d
  int v9; // edi
  int v10; // r11d

  if ( a3 == 1 )
    return 1;
  v3 = *(unsigned int *)(a1 + 224);
  v4 = *(_QWORD *)(a1 + 208);
  if ( (_DWORD)v3 )
  {
    v5 = (v3 - 1) & (37 * a3);
    v6 = (int *)(v4 + 80LL * v5);
    v7 = *v6;
    if ( *v6 == a3 )
      return sub_13A0E30((__int64)(v6 + 2), a2);
    v9 = 1;
    while ( v7 != -1 )
    {
      v10 = v9 + 1;
      v5 = (v3 - 1) & (v9 + v5);
      v6 = (int *)(v4 + 80LL * v5);
      v7 = *v6;
      if ( *v6 == a3 )
        return sub_13A0E30((__int64)(v6 + 2), a2);
      v9 = v10;
    }
  }
  return sub_13A0E30(v4 + 80 * v3 + 8, a2);
}
