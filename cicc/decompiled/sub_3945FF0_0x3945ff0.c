// Function: sub_3945FF0
// Address: 0x3945ff0
//
__int64 __fastcall sub_3945FF0(__int64 *a1, int a2)
{
  __int64 v4; // r8
  unsigned int v5; // eax
  __int64 v6; // rdx
  int v7; // esi
  int v8; // eax
  __int64 v9; // rdx
  int v11; // edx
  int v12; // edx

  if ( a2 )
  {
    v4 = *a1;
    v5 = a2 - 1;
    if ( a2 != 1 )
    {
      v6 = v4 + 16LL * v5;
      do
      {
        v7 = *(_DWORD *)(v6 + 12);
        if ( v7 != *(_DWORD *)(v6 + 8) - 1 )
        {
          v8 = v5 + 1;
          v9 = *(_QWORD *)(*(_QWORD *)v6 + 8LL * (unsigned int)(v7 + 1));
          if ( a2 == v8 )
            return v9;
          goto LABEL_7;
        }
        v6 -= 16;
        --v5;
      }
      while ( v5 );
      v11 = *(_DWORD *)(v4 + 12);
      if ( *(_DWORD *)(v4 + 8) - 1 == v11 )
        return 0;
      v9 = *(_QWORD *)(*(_QWORD *)v4 + 8LL * (unsigned int)(v11 + 1));
      v8 = 1;
      do
      {
LABEL_7:
        ++v8;
        v9 = *(_QWORD *)(v9 & 0xFFFFFFFFFFFFFFC0LL);
      }
      while ( a2 != v8 );
      return v9;
    }
    v12 = *(_DWORD *)(v4 + 12);
    if ( v12 != *(_DWORD *)(v4 + 8) - 1 )
      return *(_QWORD *)(*(_QWORD *)v4 + 8LL * (unsigned int)(v12 + 1));
  }
  return 0;
}
