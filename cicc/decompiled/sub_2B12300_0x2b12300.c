// Function: sub_2B12300
// Address: 0x2b12300
//
__int64 __fastcall sub_2B12300(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rcx
  __int64 v4; // rdi
  unsigned int v5; // eax
  unsigned int v6; // esi
  unsigned int v7; // r8d
  __int64 v8; // rcx
  __int64 v9; // rdi
  __int64 v10; // rdx
  unsigned int v11; // eax
  __int64 v12; // rax

  v3 = *(_QWORD *)(a1 + 3320);
  if ( a2 )
  {
    v4 = (unsigned int)(*(_DWORD *)(a2 + 44) + 1);
    v5 = *(_DWORD *)(a2 + 44) + 1;
  }
  else
  {
    v4 = 0;
    v5 = 0;
  }
  v6 = *(_DWORD *)(v3 + 32);
  v7 = 0;
  if ( v5 < v6 )
  {
    v8 = *(_QWORD *)(v3 + 24);
    v9 = *(_QWORD *)(v8 + 8 * v4);
    if ( v9 )
    {
      if ( a3 )
      {
        v10 = (unsigned int)(*(_DWORD *)(a3 + 44) + 1);
        v11 = v10;
      }
      else
      {
        v10 = 0;
        v11 = 0;
      }
      v7 = 1;
      if ( v6 > v11 )
      {
        v12 = *(_QWORD *)(v8 + 8 * v10);
        if ( v12 )
          LOBYTE(v7) = *(_DWORD *)(v9 + 72) < *(_DWORD *)(v12 + 72);
      }
    }
  }
  return v7;
}
