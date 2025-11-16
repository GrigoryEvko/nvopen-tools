// Function: sub_359BBF0
// Address: 0x359bbf0
//
int *__fastcall sub_359BBF0(__int64 a1, int *a2)
{
  int v2; // eax
  __int64 v3; // r9
  int v4; // edx
  int v5; // eax
  int v6; // edi
  unsigned int v7; // ecx
  int *v8; // r8
  int v9; // esi

  v2 = *(_DWORD *)(a1 + 24);
  v3 = *(_QWORD *)(a1 + 8);
  if ( v2 )
  {
    v4 = *a2;
    v5 = v2 - 1;
    v6 = 1;
    v7 = v5 & (37 * *a2);
    v8 = (int *)(v3 + 8LL * v7);
    v9 = *v8;
    if ( v4 == *v8 )
      return v8;
    while ( v9 != -1 )
    {
      v7 = v5 & (v6 + v7);
      v8 = (int *)(v3 + 8LL * v7);
      v9 = *v8;
      if ( *v8 == v4 )
        return v8;
      ++v6;
    }
  }
  return 0;
}
