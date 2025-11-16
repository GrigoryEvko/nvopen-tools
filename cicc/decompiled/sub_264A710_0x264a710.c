// Function: sub_264A710
// Address: 0x264a710
//
bool __fastcall sub_264A710(__int64 a1, int *a2)
{
  __int64 v2; // r8
  __int64 v3; // rax
  int *v4; // r9
  int v5; // edx
  int v6; // eax
  unsigned int v7; // ecx
  int *v8; // rsi
  int v9; // edi
  int v11; // esi
  int v12; // r10d

  v2 = *(_QWORD *)(a1 + 8);
  v3 = *(unsigned int *)(a1 + 24);
  v4 = (int *)(v2 + 4 * v3);
  if ( (_DWORD)v3 )
  {
    v5 = *a2;
    v6 = v3 - 1;
    v7 = v6 & (37 * *a2);
    v8 = (int *)(v2 + 4LL * v7);
    v9 = *v8;
    if ( v5 == *v8 )
      return v4 != v8;
    v11 = 1;
    while ( v9 != -1 )
    {
      v12 = v11 + 1;
      v7 = v6 & (v11 + v7);
      v8 = (int *)(v2 + 4LL * v7);
      v9 = *v8;
      if ( v5 == *v8 )
        return v4 != v8;
      v11 = v12;
    }
  }
  return 0;
}
