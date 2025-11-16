// Function: sub_2EB3BB0
// Address: 0x2eb3bb0
//
__int64 __fastcall sub_2EB3BB0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r8
  unsigned int v4; // ecx
  unsigned int v5; // esi
  _DWORD *v6; // rax
  __int64 v7; // r8
  unsigned int v8; // ecx
  _DWORD *v9; // rdx
  _DWORD *v10; // rcx

  if ( a2 )
  {
    v3 = (unsigned int)(*(_DWORD *)(a2 + 24) + 1);
    v4 = *(_DWORD *)(a2 + 24) + 1;
  }
  else
  {
    v3 = 0;
    v4 = 0;
  }
  v5 = *(_DWORD *)(a1 + 56);
  v6 = 0;
  if ( v4 < v5 )
    v6 = *(_DWORD **)(*(_QWORD *)(a1 + 48) + 8 * v3);
  if ( a3 )
  {
    v7 = (unsigned int)(*(_DWORD *)(a3 + 24) + 1);
    v8 = *(_DWORD *)(a3 + 24) + 1;
  }
  else
  {
    v7 = 0;
    v8 = 0;
  }
  v9 = 0;
  if ( v8 < v5 )
    v9 = *(_DWORD **)(*(_QWORD *)(a1 + 48) + 8 * v7);
  while ( v6 != v9 )
  {
    if ( v9[4] > v6[4] )
    {
      v10 = v6;
      v6 = v9;
      v9 = v10;
    }
    v6 = (_DWORD *)*((_QWORD *)v6 + 1);
  }
  return *(_QWORD *)v9;
}
