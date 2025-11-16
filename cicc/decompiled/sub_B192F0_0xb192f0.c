// Function: sub_B192F0
// Address: 0xb192f0
//
__int64 __fastcall sub_B192F0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  int v4; // ecx
  unsigned int v5; // esi
  _DWORD *v6; // rax
  __int64 v7; // rcx
  __int64 v8; // r8
  unsigned int v9; // ecx
  _DWORD *v10; // rdx
  _DWORD *v11; // rcx

  result = *(_QWORD *)(*(_QWORD *)(a2 + 72) + 80LL);
  if ( result )
    result -= 24;
  if ( a2 != result && a3 != result )
  {
    v4 = *(_DWORD *)(a2 + 44);
    v5 = *(_DWORD *)(a1 + 32);
    v6 = 0;
    v7 = (unsigned int)(v4 + 1);
    if ( (unsigned int)v7 < v5 )
      v6 = *(_DWORD **)(*(_QWORD *)(a1 + 24) + 8 * v7);
    if ( a3 )
    {
      v8 = (unsigned int)(*(_DWORD *)(a3 + 44) + 1);
      v9 = *(_DWORD *)(a3 + 44) + 1;
    }
    else
    {
      v8 = 0;
      v9 = 0;
    }
    v10 = 0;
    if ( v9 < v5 )
      v10 = *(_DWORD **)(*(_QWORD *)(a1 + 24) + 8 * v8);
    while ( v6 != v10 )
    {
      if ( v10[4] > v6[4] )
      {
        v11 = v6;
        v6 = v10;
        v10 = v11;
      }
      v6 = (_DWORD *)*((_QWORD *)v6 + 1);
    }
    return *(_QWORD *)v10;
  }
  return result;
}
