// Function: sub_1B961A0
// Address: 0x1b961a0
//
__int64 __fastcall sub_1B961A0(__int64 **a1, int *a2)
{
  int v2; // edx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // rcx
  unsigned int v7; // esi
  int *v8; // rdi
  int v9; // r9d
  unsigned int v10; // r8d
  int v12; // edi
  int v13; // r11d

  v2 = *a2;
  if ( *a2 != 1 )
  {
    v4 = (*a1)[4];
    v5 = *(_QWORD *)(v4 + 176);
    v6 = *(unsigned int *)(v4 + 192);
    if ( (_DWORD)v6 )
    {
      v7 = (v6 - 1) & (37 * v2);
      v8 = (int *)(v5 + 80LL * v7);
      v9 = *v8;
      if ( v2 == *v8 )
      {
LABEL_4:
        LOBYTE(v10) = sub_13A0E30((__int64)(v8 + 2), *a1[1]);
        return v10;
      }
      v12 = 1;
      while ( v9 != -1 )
      {
        v13 = v12 + 1;
        v7 = (v6 - 1) & (v12 + v7);
        v8 = (int *)(v5 + 80LL * v7);
        v9 = *v8;
        if ( v2 == *v8 )
          goto LABEL_4;
        v12 = v13;
      }
    }
    v8 = (int *)(v5 + 80 * v6);
    goto LABEL_4;
  }
  return 1;
}
