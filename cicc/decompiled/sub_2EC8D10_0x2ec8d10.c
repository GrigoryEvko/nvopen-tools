// Function: sub_2EC8D10
// Address: 0x2ec8d10
//
__int64 __fastcall sub_2EC8D10(__int64 a1, unsigned int *a2)
{
  char v3; // al
  unsigned int v4; // r8d
  __int64 v5; // rax
  __int64 v6; // rcx
  int v7; // edi
  unsigned int v8; // edx
  unsigned int v9; // ecx

  *a2 = 0;
  v3 = sub_2FF7B70(*(_QWORD *)(a1 + 8));
  v4 = 0;
  if ( v3 )
  {
    v5 = *(_QWORD *)(a1 + 8);
    v6 = *(_QWORD *)(a1 + 16);
    v7 = *(_DWORD *)(v5 + 48);
    v4 = *(_DWORD *)(v6 + 8) + *(_DWORD *)(v5 + 288) * *(_DWORD *)(a1 + 184);
    if ( v7 != 1 )
    {
      v8 = 1;
      while ( 1 )
      {
        v9 = *(_DWORD *)(*(_QWORD *)(a1 + 192) + 4LL * v8) + *(_DWORD *)(*(_QWORD *)(v6 + 16) + 4LL * v8);
        if ( v9 > v4 )
        {
          *a2 = v8;
          v4 = v9;
        }
        if ( v7 == ++v8 )
          break;
        v6 = *(_QWORD *)(a1 + 16);
      }
    }
  }
  return v4;
}
