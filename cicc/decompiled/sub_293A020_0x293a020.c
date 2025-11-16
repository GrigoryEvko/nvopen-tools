// Function: sub_293A020
// Address: 0x293a020
//
bool __fastcall sub_293A020(__int64 a1, unsigned __int8 *a2)
{
  __int64 v2; // rcx
  __int64 v3; // r8
  int v4; // eax
  int *v5; // r9
  int v6; // edx
  unsigned int v7; // ecx
  int *v8; // rsi
  int v9; // edi
  int v11; // esi
  int v12; // r10d

  v2 = *(unsigned int *)(a1 + 1160);
  v3 = *(_QWORD *)(a1 + 1144);
  v4 = *a2 - 29;
  v5 = (int *)(v3 + 4 * v2);
  if ( (_DWORD)v2 )
  {
    v6 = v2 - 1;
    v7 = (v2 - 1) & (37 * v4);
    v8 = (int *)(v3 + 4LL * v7);
    v9 = *v8;
    if ( v4 == *v8 )
      return v5 != v8;
    v11 = 1;
    while ( v9 != -1 )
    {
      v12 = v11 + 1;
      v7 = v6 & (v11 + v7);
      v8 = (int *)(v3 + 4LL * v7);
      v9 = *v8;
      if ( v4 == *v8 )
        return v5 != v8;
      v11 = v12;
    }
  }
  return 0;
}
